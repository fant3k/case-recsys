"""Why A gives 0.1583 in the final dry run and 0.1585 in RL v3: compare two numerical paths on selection.

Same reference, RM v2 seeds 0,1, top-50 and beta. Path fp32: RM scores computed directly for the 50 candidates
(recsys.final_test). Path fp16: full-catalog cache rounded to half precision and gathered (recsys.rl_distill).
Selection only; nothing is chosen from the result. Final test untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import rl_v2
from .data import load_sequences, pack_contexts
from .final_test import BETA_A, CHECKPOINTS, TOP_A, rerank_logp
from .full_sft import prepare_feedback
from .metrics import ranking_metrics
from .models import make_policy
from .pilot import atomic_json
from .reward_v2 import side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()  # no options; makes --help safe
    out = ROOT / "artifacts/a-precision-check-20261023-01"
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    t = proto["time"]
    device = rl_v2.DEVICE = "cuda"
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    ck = torch.load(ROOT / CHECKPOINTS["reference"], map_location="cpu", weights_only=True)
    reference = make_policy(ck["settings"]); reference.load_state_dict(ck["model"])
    reference = reference.to(device).eval().requires_grad_(False)
    rms = [rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1")]
    cutoff = t["query_cutoffs"]["selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    uids, _, _, _ = data.queries(cutoff)
    windows = []
    for uid in uids:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], cutoff))
        windows.append((u, max(0, end - data.context_size), end))
    x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
    tensors = [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]
    truth = data.truth(lo, hi)
    ref = rl_v2.policy_logp(reference, tensors)
    idx = ref.topk(TOP_A, -1).indices
    with torch.no_grad():
        r32 = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            r = [topk_rm_scores(rm, *(v[sl] for v in tensors), idx[sl]) for rm in rms]
            r32.append((r[0].float() + r[1].float()) / 2)
        r32 = torch.cat(r32)
        cache = (rl_v2.rm_cache(rms[0], tensors).float() + rl_v2.rm_cache(rms[1], tensors).float()) / 2
        r16 = cache.gather(1, idx)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "window": "selection", "beta": BETA_A, "top_k": TOP_A,
              "max_abs_reward_difference": float((r32 - r16).abs().max()),
              "mean_abs_reward_difference": float((r32 - r16).abs().mean()), "paths": {}}
    tops = {}
    for name, r in (("fp32_direct_top50", r32), ("fp16_cache_gather", r16)):
        top = rerank_logp(ref, idx, r).topk(10, -1).indices
        tops[name] = top
        metrics, _ = ranking_metrics(uids, data.catalog[top.cpu().numpy()].tolist(), truth)
        result["paths"][name] = metrics
    differ = (tops["fp32_direct_top50"] != tops["fp16_cache_gather"]).any(1)
    as_sets = [set(a.tolist()) != set(b.tolist()) for a, b in zip(tops["fp32_direct_top50"], tops["fp16_cache_gather"])]
    result.update({"users_with_different_top10_order": int(differ.sum()), "users_with_different_top10_set": int(sum(as_sets)),
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "elapsed_seconds": time.perf_counter() - started, "final_test_scored": False})
    atomic_json(out / "report.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
