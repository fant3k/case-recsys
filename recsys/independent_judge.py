"""A judge of a different family: gradient-boosted trees on tabular history features (docs/INDEPENDENT-JUDGE-PLAN.md).

The reward models that trained the policies and the seed-2 judge are neural networks on the same backbone and logs.
This judge is LightGBM regression of the logged played ratio of the target event on nine history features
(count and recency of the track in the context, pairwise past completion, user means, track length and popularity),
trained on train-period events only, with no embeddings and no GPT-2. It scores the top-10 of the reference, A, B and
the personal-frequency baseline on the selection day. Selection only; nothing is selected from the result.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch

from . import rl_v2
from .data import load_sequences, pack_contexts, personal_frequency
from .final_test import BETA_A, CHECKPOINTS, SECONDS, TOP_A, interval, rerank_logp, sha
from .full_sft import prepare_feedback
from .models import make_policy
from .pilot import atomic_json
from .reward_v2 import candidate_features, side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
POOL, SEED = 65536, 20261401


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/independent-judge-selection-20261025-01")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    t = proto["time"]
    cutoff = t["query_cutoffs"]["selection_validation"]
    device = rl_v2.DEVICE = "cuda"
    torch.set_num_threads(4)
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    rms = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1")}
    item_features = rms["rm_seed0"].item_features
    atomic_json(out / "plan.json", {"plan_document": "docs/INDEPENDENT-JUDGE-PLAN.md", "window": "selection", "pool_contexts": POOL, "seed": SEED,
                                    "protocol_sha256": sha(proto_path), "scope": "selection only; confirmation, final and new cohort are not read",
                                    "features": ["log_count", "recency", "seen", "user_track_completion", "length", "popularity",
                                                 "user_mean_completion", "organic_share", "context_length"]})

    def tensors_of(windows):
        x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
        return [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]

    # ---- train the judge on train-period logged events ----
    _, _, actions, ratios, records = data.sample(np.random.default_rng(SEED), POOL)
    windows = []
    for uid, _, event_time in records:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], event_time))
        windows.append((u, max(0, end - data.context_size), end))
    feats = []
    for s in range(0, len(windows), 4096):
        tt = tensors_of(windows[s:s + 4096])
        feats.append(candidate_features(tt[0], tt[4].float(), tt[5], torch.as_tensor(actions[s:s + 4096], device=device),
                                        item_features).cpu().numpy())
    x_all, y_all = np.concatenate(feats), np.minimum(np.asarray(ratios, dtype=np.float64), 1.0)
    split = int(len(y_all) * 0.9)
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.9, random_state=SEED, verbose=-1)
    model.fit(x_all[:split], y_all[:split])
    pred_val = model.predict(x_all[split:])
    held_out = {"events": int(len(y_all) - split), "mae": float(np.abs(pred_val - y_all[split:]).mean()),
                "mae_constant_mean": float(np.abs(y_all[:split].mean() - y_all[split:]).mean()),
                "correlation": float(np.corrcoef(pred_val, y_all[split:])[0, 1])}
    model.fit(x_all, y_all)

    # ---- policies on the selection day ----
    uids, _, _, raw = data.queries(cutoff)
    qwin = []
    for uid in uids:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], cutoff))
        qwin.append((u, max(0, end - data.context_size), end))
    q = tensors_of(qwin)

    def load_policy(name):
        ck = torch.load(ROOT / CHECKPOINTS[name], map_location="cpu", weights_only=True)
        m = make_policy(ck["settings"])
        m.load_state_dict(ck["model"])
        return m.to(device).eval().requires_grad_(False)

    reference, rl_b = load_policy("reference"), load_policy("B_rl_seconds_beta0.3")
    ref_logp = rl_v2.policy_logp(reference, q)
    idx = ref_logp.topk(TOP_A, -1).indices
    with torch.no_grad():
        parts = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            r = [topk_rm_scores(rms[n], *(v[sl] for v in q), idx[sl]).float() for n in ("rm_seed0", "rm_seed1")]
            parts.append((r[0] + r[1]) / 2)
        a_logp = rerank_logp(ref_logp, idx, torch.cat(parts), BETA_A)
    top = lambda lp: lp.topk(10, -1).indices
    index_of = {int(i): j for j, i in enumerate(data.catalog)}
    methods = {"reference": top(ref_logp), "A_rm_rerank_top50": top(a_logp), "B_rl_seconds_beta0.3": top(rl_v2.policy_logp(rl_b, q)),
               "PersonalFrequency128": torch.as_tensor([[index_of[int(i)] for i in r] for r in personal_frequency(raw, data.catalog, 10)], device=device)}
    length = item_features[:, 0].cpu().numpy() * SECONDS
    per_user = {}
    for name, ids in methods.items():
        n = len(ids)
        rep = lambda v: v.repeat_interleave(10, 0)
        f = candidate_features(rep(q[0]), rep(q[4].float()), rep(q[5]), ids.reshape(-1), item_features).cpu().numpy()
        completion = model.predict(f).reshape(n, 10)
        per_user[name] = {"completion": completion.mean(1), "seconds": (completion * length[ids.cpu().numpy()]).mean(1)}
    report = {"warm_users": len(uids), "judge_held_out": held_out, "methods": {}, "elapsed_seconds": time.perf_counter() - started}
    for name, v in per_user.items():
        entry = {"mean_judge_completion_top10": float(v["completion"].mean()), "mean_judge_seconds_top10": float(v["seconds"].mean())}
        if name != "reference":
            entry["completion_vs_reference_95"] = interval(v["completion"] - per_user["reference"]["completion"], 0.95, seed=SEED)
            entry["seconds_vs_reference_95"] = interval(v["seconds"] - per_user["reference"]["seconds"], 0.95, seed=SEED)
        report["methods"][name] = entry
    atomic_json(out / "report.json", report)
    atomic_json(out / "status.json", {"status": "completed"})
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
