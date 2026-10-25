"""Where does the gain of candidate A come from inside the reward model? (docs/RM-ABLATION-PLAN.md)

Candidate A reranks the reference top-50 by pi_ref * exp(r / beta). Here the same reranking is run with three
reward models, one at a time: the full RM v2 of seed 0 (with pairwise user-track features), the RM v2 trained
without those features (seed 0, same backbone and data), and the two-model ensemble used by A. Selection window
only; nothing is selected from the result.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import rl_v2
from .data import load_sequences, pack_contexts
from .final_test import BETA_A, CHECKPOINTS, TOP_A, interval, rerank_logp, sha
from .full_sft import prepare_feedback
from .metrics import ranking_metrics
from .models import make_policy
from .pilot import atomic_json
from .reward_v2 import side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
NO_FEATURES = "artifacts/reward-v2-20261022-01/v2_no_features_seed0.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/rm-ablation-selection-20261025-01")
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
    device = rl_v2.DEVICE = "cuda"
    torch.set_num_threads(4)
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    ck = torch.load(ROOT / CHECKPOINTS["reference"], map_location="cpu", weights_only=True)
    reference = make_policy(ck["settings"])
    reference.load_state_dict(ck["model"])
    reference = reference.to(device).eval().requires_grad_(False)
    paths = {"full_seed0": CHECKPOINTS["rm_seed0"], "full_seed1": CHECKPOINTS["rm_seed1"], "no_features_seed0": NO_FEATURES}
    rms = {n: rl_v2.load_rm(ROOT / p, features)[0] for n, p in paths.items()}
    atomic_json(out / "plan.json", {"plan_document": "docs/RM-ABLATION-PLAN.md", "protocol_sha256": proto_hash, "window": "selection",
                                    "label_window": [lo, hi], "beta": BETA_A, "top": TOP_A,
                                    "checkpoint_sha256": {n: sha(ROOT / p) for n, p in {"reference": CHECKPOINTS["reference"], **paths}.items()},
                                    "scope": "selection only; confirmation, final and new cohort are not read"})
    uids, _, _, _ = data.queries(cutoff)
    windows = []
    for uid in uids:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], cutoff))
        windows.append((u, max(0, end - data.context_size), end))
    x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
    tensors = [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]
    truth = data.truth(lo, hi)
    ref_logp = rl_v2.policy_logp(reference, tensors)
    idx = ref_logp.topk(TOP_A, -1).indices

    @torch.no_grad()
    def scores(names):
        parts = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            r = [topk_rm_scores(rms[n], *(v[sl] for v in tensors), idx[sl]).float() for n in names]
            parts.append(sum(r) / len(r))
        return torch.cat(parts)

    def recs(logp):
        return data.catalog[logp.topk(10, -1).indices.cpu().numpy()].tolist()

    rows = {}
    metrics = {}
    for name, lp in (("reference", ref_logp),
                     ("A_full_rm_seed0", rerank_logp(ref_logp, idx, scores(["full_seed0"]))),
                     ("A_rm_without_pairwise_features_seed0", rerank_logp(ref_logp, idx, scores(["no_features_seed0"]))),
                     ("A_ensemble_seeds01", rerank_logp(ref_logp, idx, scores(["full_seed0", "full_seed1"])))):
        m, r = ranking_metrics(uids, recs(lp), truth)
        metrics[name], rows[name] = m, r
    ndcg = {n: np.array([r["ndcg"] for r in v]) for n, v in rows.items()}
    cmp = lambda a, b: interval(ndcg[a] - ndcg[b], 0.95)
    report = {"warm_users": len(uids), "active_users": len(truth), "rows": metrics,
              "paired_95": {"A_full_seed0_minus_reference": cmp("A_full_rm_seed0", "reference"),
                            "A_without_features_minus_reference": cmp("A_rm_without_pairwise_features_seed0", "reference"),
                            "A_ensemble_minus_reference": cmp("A_ensemble_seeds01", "reference"),
                            "full_seed0_minus_without_features": cmp("A_full_rm_seed0", "A_rm_without_pairwise_features_seed0")},
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json(out / "report.json", report)
    atomic_json(out / "status.json", {"status": "completed"})
    for k, v in report["paired_95"].items():
        print(k, round(v["mean"], 4), [round(z, 4) for z in v["interval"]])


if __name__ == "__main__":
    main()
