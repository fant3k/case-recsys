"""Model-free outcome check of the final candidates on the selection day (docs/REPLAY-PLAN.md).

The judge reward model predicts what a user would do; here the same quantities are read from what the users
actually did in the label window: seconds listened (completion * min(train-median length, 600)), plays, and
mean completion among the recommended tracks that were played. Recommendations come from the frozen reference,
candidate A (top-50 reranking by RM v2), candidate B (RL on capped seconds, beta 0.3) and the personal-frequency
baseline. Only logged outcomes of tracks the logging system showed can be credited, so this is a replay
estimate: it ignores what users would have done with unseen tracks. Selection window only; there is no option
to read confirmation, the final window or the new cohort.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts, personal_frequency
from .full_sft import prepare_feedback
from .final_test import BETA_A, CHECKPOINTS, SECONDS, TOP_A, interval, rerank_logp, sha
from .models import make_policy
from .pilot import atomic_json
from .reward_v2 import side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]


def user_outcomes(items, recs):
    """Replay outcomes of one user's top-10: realised seconds and plays of the recommended items, completion on played ones.

    `items` maps catalog item id -> (seconds listened, number of plays, sum of completions) in the label window.
    """
    got_seconds = got_plays = got_completion = 0.0
    for item in recs:
        if item in items:
            s, n, c = items[item]
            got_seconds += s
            got_plays += n
            got_completion += c
    return {"replay_seconds": got_seconds, "replay_plays": got_plays,
            "completion_sum": got_completion}


def liked_hits(liked, recs):
    """Number of recommended tracks that the user liked in the label window."""
    return sum(1 for item in recs if item in liked)


def summarize(rows, reference_rows, draws_level=0.95):
    """Paired bootstrap of every per-user outcome against the reference, plus pooled completion on played tracks."""
    out = {}
    for key in ("replay_seconds", "replay_plays"):
        mine = np.array([r[key] for r in rows], dtype=np.float64)
        ref = np.array([r[key] for r in reference_rows], dtype=np.float64)
        out[key] = {"mean": float(mine.mean()), "reference_mean": float(ref.mean()),
                    "delta_vs_reference": interval(mine - ref, draws_level)}
    if "liked_top10" in rows[0]:
        mine = np.array([r["liked_top10"] for r in rows], dtype=np.float64)
        ref = np.array([r["liked_top10"] for r in reference_rows], dtype=np.float64)
        out["liked_top10"] = {"mean": float(mine.mean()), "reference_mean": float(ref.mean()), "liked_recommended_tracks": float(mine.sum()),
                              "users_with_a_liked_recommendation": int((mine > 0).sum()),
                              "delta_vs_reference": interval(mine - ref, draws_level)}
    plays = sum(r["replay_plays"] for r in rows)
    out["completion_on_played_recommended_tracks"] = (sum(r["completion_sum"] for r in rows) / plays) if plays else None
    out["played_recommended_tracks"] = plays
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/replay-selection-20261025-01")
    parser.add_argument("--likes", default=None, metavar="PARQUET",
                        help="Yambda likes.parquet (sequential): adds the number of recommended tracks the user liked in the label window")
    parser.add_argument("--extra-policy", action="append", default=[], metavar="NAME=CHECKPOINT",
                        help="additional frozen policy checkpoints (repeatable), e.g. RL seeds of candidate B")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = sha(proto_path)
    t = proto["time"]
    cutoff = t["query_cutoffs"]["selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan_document": "docs/REPLAY-PLAN.md", "window": "selection",
            "query_cutoff": cutoff, "label_window": [lo, hi], "protocol_sha256": proto_hash,
            "checkpoint_sha256": {n: sha(ROOT / CHECKPOINTS[n]) for n in
                                  ("reference", "B_rl_seconds_beta0.3", "rm_seed0", "rm_seed1")},
            "extra_policies": dict(spec.split("=", 1) for spec in args.extra_policy),
            "likes_sha256": sha(ROOT / args.likes) if args.likes else None,
            "scope": "selection only; confirmation, final and new cohort are not read"}
    atomic_json(out / "plan.json", plan)
    device = rl_v2.DEVICE = "cuda"
    torch.set_num_threads(4)
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    capped_length = features[:, 0] * SECONDS  # train-median length capped at 600 s, per catalog index

    def load_policy(name):
        ck = torch.load(ROOT / CHECKPOINTS[name], map_location="cpu", weights_only=True)
        if ck.get("protocol_sha256", proto_hash) != proto_hash or not np.array_equal(ck["catalog"].numpy(), data.catalog):
            raise ValueError("Catalog/protocol mismatch: " + name)
        model = make_policy(ck["settings"])
        model.load_state_dict(ck["model"])
        return model.to(device).eval().requires_grad_(False)

    reference, rl_b = load_policy("reference"), load_policy("B_rl_seconds_beta0.3")
    extra = {}
    for spec in args.extra_policy:
        name, path = spec.split("=", 1)
        CHECKPOINTS[name] = path
        extra[name] = load_policy(name)
    rms = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1")}

    uids, _, _, raw_histories = data.queries(cutoff)
    windows = []
    for uid in uids:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], cutoff))
        windows.append((u, max(0, end - data.context_size), end))
    x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
    rb, gb, cr, co = side_inputs(windows, data.context_size)
    tensors = [torch.as_tensor(v, device=device) for v in (x, mask, rb, gb, cr, co)]

    # Realised outcomes of every user in the label window, per catalog item id.
    outcomes = {}
    for u in data.users:
        left, right = np.searchsorted(u["times"], [lo, hi])
        items = {}
        for event in range(left, right):
            token = int(u["tokens"][event])
            if token < OFFSET:
                continue
            ratio = float(u["ratios"][event])
            s, n, c = items.get(int(u["items"][event]), (0.0, 0, 0.0))
            items[int(u["items"][event])] = (s + ratio * float(capped_length[token - OFFSET]), n + 1, c + ratio)
        outcomes[u["uid"]] = items

    liked = {}
    if args.likes:
        import polars as pl
        table = pl.read_parquet(ROOT / args.likes)
        for uid, stamps, item_ids in zip(table["uid"].to_list(), table["timestamp"].to_list(), table["item_id"].to_list()):
            liked[int(uid)] = {int(i) for s_, i in zip(stamps, item_ids) if lo <= s_ < hi}

    ref_logp = rl_v2.policy_logp(reference, tensors)
    with torch.no_grad():
        idx = ref_logp.topk(TOP_A, -1).indices
        scores = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            r = [topk_rm_scores(rms[n], *(v[sl] for v in tensors), idx[sl]) for n in ("rm_seed0", "rm_seed1")]
            scores.append((r[0].float() + r[1].float()) / 2)
        reranked = rerank_logp(ref_logp, idx, torch.cat(scores), BETA_A)
    top = lambda lp: data.catalog[lp.topk(10, -1).indices.cpu().numpy()].tolist()
    methods = {"reference": top(ref_logp), "A_rm_rerank_top50": top(reranked),
               "B_rl_seconds_beta0.3": top(rl_v2.policy_logp(rl_b, tensors)),
               "PersonalFrequency128": personal_frequency(raw_histories, data.catalog, 10)}
    for name, model in extra.items():
        methods[name] = top(rl_v2.policy_logp(model, tensors))
    def one(uid, recs):
        row = user_outcomes(outcomes[uid], recs)
        if args.likes:
            row["liked_top10"] = liked_hits(liked.get(uid, set()), recs)
        return row

    rows = {name: [one(uid, [int(i) for i in recs[j]]) for j, uid in enumerate(uids)] for name, recs in methods.items()}
    active = [j for j, uid in enumerate(uids) if outcomes[uid]]
    report = {"plan": plan, "warm_users": len(uids), "users_with_any_like_in_window": sum(1 for u in uids if liked.get(u)), "users_with_any_logged_listening_in_window": len(active),
              "meaning": "realised logged listening to the recommended tracks; replay estimate, not a causal effect",
              "all_warm_users": {n: summarize(r, rows["reference"]) for n, r in rows.items()},
              "users_with_listening": {n: summarize([r[j] for j in active], [rows["reference"][j] for j in active])
                                       for n, r in rows.items()},
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json(out / "report.json", report)
    atomic_json(out / "status.json", {"status": "completed"})


if __name__ == "__main__":
    main()
