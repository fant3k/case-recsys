"""PairAccuracy of RM v2 and the track mean on one basis (selection window, descriptive; nothing selected).

The global value (reward-v2 report, 0.555) and the top-50 value (review checks, 0.620) differed in four ways:
margin rule at exactly 50 points, ensemble (seeds 0-2 vs 0-1), context (history right before each listen vs
history at the query cutoff) and organic flag (actual vs 0). Here every combination is computed with the same
events, the new margin rule, and the old rule for reference. Final test untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts
from .final_test import CHECKPOINTS, TOP_A
from .full_sft import prepare_feedback
from .models import make_policy
from .pilot import atomic_json
from .reward_audit import observed_records
from .reward_v2 import candidate_features, pair_accuracy, side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
OLD_RULE = 0.5 + 1e-6  # with LABEL_TOLERANCE this reproduces the former exact `dy >= 0.5`


def pair_accuracy_distinct(predictions, labels, uids, items, margin=0.5):
    """Same as reward_v2.pair_accuracy (new rule) but without pairs of two listens of the same track.

    Cutoff-context scores tie such pairs by construction, so this separates the context effect from pair composition.
    """
    per_user, pairs = [], 0
    for u in np.unique(uids):
        m = uids == u
        p, y, it = predictions[m], labels[m], items[m]
        take = ((y[:, None] - y[None, :]) >= margin - 1e-6) & (it[:, None] != it[None, :])
        if take.any():
            dp = (p[:, None] - p[None, :])[take]
            per_user.append(float(((dp > 0) + 0.5 * (dp == 0)).mean()))
            pairs += int(take.sum())
    return {"users_with_pairs": len(per_user), "pairs": pairs, "macro_pair_accuracy": float(np.mean(per_user)) if per_user else None}


def pair_accuracy_split(predictions, labels, uids, sessions, same_session, margin=0.5):
    """New-rule PairAccuracy over pairs whose two listens are in the same session (gaps <= 20 min) or in different ones."""
    per_user, pairs = [], 0
    for u in np.unique(uids):
        m = uids == u
        p, y, ss = predictions[m], labels[m], sessions[m]
        same = ss[:, None] == ss[None, :]
        take = ((y[:, None] - y[None, :]) >= margin - 1e-6) & (same if same_session else ~same)
        if take.any():
            dp = (p[:, None] - p[None, :])[take]
            per_user.append(float(((dp > 0) + 0.5 * (dp == 0)).mean()))
            pairs += int(take.sum())
    return {"users_with_pairs": len(per_user), "pairs": pairs, "macro_pair_accuracy": float(np.mean(per_user)) if per_user else None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="artifacts/pairacc-recheck-20261023-02", help="output folder relative to the repository root; must not exist")
    out_name = parser.parse_args().out
    out = ROOT / out_name
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto = json.loads((ROOT / "research/protocol/case2-v1.json").read_text())
    t = proto["time"]
    cutoff = t["query_cutoffs"]["selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    device = rl_v2.DEVICE = "cuda"
    data, features, means, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    names = ("rm_seed0", "rm_seed1", "rm_judge_seed2")
    rms = [rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in names]
    ck = torch.load(ROOT / CHECKPOINTS["reference"], map_location="cpu", weights_only=True)
    reference = make_policy(ck["settings"]); reference.load_state_dict(ck["model"])
    reference = reference.to(device).eval().requires_grad_(False)

    records, _ = observed_records(data, lo, hi)
    uid = np.array([r[0] for r in records]); action = np.array([r[3] for r in records])
    label = np.array([r[4] for r in records]); organic = np.array([by_uid[r[0]]["organic"][r[1]] for r in records])
    session_of = {u["uid"]: np.r_[0, np.cumsum(np.diff(u["times"]) > 1200)] for u in data.users}
    session = np.array([session_of[r[0]][r[1]] for r in records])

    def tensors(windows):
        x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
        return [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]

    # Event context: history strictly before the listen's timestamp group (reward-v2 convention).
    event_scores = {flag: np.zeros((3, len(records))) for flag in ("actual", "zero")}
    with torch.inference_mode():
        for j in range(0, len(records), 256):
            sub = records[j:j + 256]
            tt = tensors([(by_uid[r[0]], max(0, r[2] - data.context_size), r[2]) for r in sub])
            act = torch.as_tensor(action[j:j + 256], device=device)
            for flag in ("actual", "zero"):
                f = torch.as_tensor(organic[j:j + 256] if flag == "actual" else np.zeros(len(sub)), dtype=torch.float32, device=device)
                for s, rm in enumerate(rms):
                    event_scores[flag][s, j:j + 256] = rm(*tt, act, f).float().cpu().numpy()

    # Cutoff context: history at the query cutoff, where A / RL choose (organic flag 0).
    users = sorted(set(uid.tolist()))
    row = {u: k for k, u in enumerate(users)}
    windows = []
    for u in users:
        user = by_uid[u]
        end = int(np.searchsorted(user["times"], cutoff))
        windows.append((user, max(0, end - data.context_size), end))
    cut_t = tensors(windows)
    ref = rl_v2.policy_logp(reference, cut_t)
    top = ref.topk(TOP_A, -1).indices
    rows = torch.as_tensor([row[u] for u in uid], device=device)
    acts = torch.as_tensor(action, device=device)
    in_top = (top[rows] == acts[:, None]).any(1).cpu().numpy()
    cut_scores = np.zeros((3, len(records)))
    personal = np.zeros(len(records))
    with torch.inference_mode():
        for j in range(0, len(records), 256):
            sl = slice(j, j + 256)
            part = [v[rows[sl]] for v in cut_t]
            for s, rm in enumerate(rms):
                cut_scores[s, sl] = topk_rm_scores(rm, *part, acts[sl][:, None]).float().squeeze(1).cpu().numpy()
            f = candidate_features(part[0], part[4].float(), part[5], acts[sl], rms[0].item_features)
            personal[sl] = torch.where(f[:, 2] > 0, f[:, 3], f[:, 6]).cpu().numpy()

    item_mean = means[action]
    predictions = {
        "event_context/actual_flag/ensemble3": event_scores["actual"].mean(0),
        "event_context/actual_flag/seeds01": event_scores["actual"][:2].mean(0),
        "event_context/flag0/seeds01": event_scores["zero"][:2].mean(0),
        "cutoff_context/flag0/seeds01": cut_scores[:2].mean(0),
        "cutoff_context/flag0/ensemble3": cut_scores.mean(0),
        "track_train_mean": item_mean,
        "cutoff_context/user_past_completion": personal}
    subsets = {"all_events": np.ones(len(records), dtype=bool), "item_in_reference_top50": in_top}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "window": "selection", "events": len(records),
              "events_in_top50": int(in_top.sum()), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rules": {"new": "label difference >= 0.5 - 1e-6", "old": "label difference >= 0.5 exactly (float32 labels)"},
              "results": {}, "final_test_scored": False}
    for sname, m in subsets.items():
        for pname, p in predictions.items():
            report["results"][f"{sname}/{pname}"] = {"new_rule": pair_accuracy(p[m], label[m], uid[m]),
                                                     "old_rule": pair_accuracy(p[m], label[m], uid[m], margin=OLD_RULE),
                                                     "new_rule_distinct_tracks": pair_accuracy_distinct(p[m], label[m], uid[m], action[m]),
                                                     "same_session": pair_accuracy_split(p[m], label[m], uid[m], session[m], True),
                                                     "different_sessions": pair_accuracy_split(p[m], label[m], uid[m], session[m], False)}
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(out / "report.json", report)
    for k, v in report["results"].items():
        d = v["new_rule_distinct_tracks"]
        ss, ds = v["same_session"], v["different_sessions"]
        print(f'{k:62s} new {v["new_rule"]["macro_pair_accuracy"]:.4f} distinct {d["macro_pair_accuracy"]:.4f} '
              f'same-session {ss["macro_pair_accuracy"]} ({ss["pairs"]}) different {ds["macro_pair_accuracy"]} ({ds["pairs"]})')


if __name__ == "__main__":
    main()
