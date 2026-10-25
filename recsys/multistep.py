"""H5 for the v2 policies: five recommendations generated one by one, each appended to the context as if listened.

Descriptive diagnostic on the already used selection window; nothing is selected and final test is untouched.
The v2 policies and RM v2 read the feedback of every past event (listened ratio, gap). Generated items have no
observed feedback, so it is invented under three assumptions: fully listened, skipped, and the frozen judge RM's
prediction (a user simulator, as in Matveev et al. 2026). The same items are also scored by the judge with the
real history only, which shows how much the invented context moves the reward estimate.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts
from .final_test import BETA_A, CHECKPOINTS, SECONDS, TOP_A, rerank_logp
from .full_sft import prepare_feedback
from .models import GAP_BUCKETS, make_policy
from .pilot import atomic_json
from .reward_v2 import side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
STEPS = 5
POLICIES = {"reference": CHECKPOINTS["reference"],
            "B_rl_seconds_beta0.3": CHECKPOINTS["B_rl_seconds_beta0.3"],
            "rl_seconds_beta0.1": "artifacts/rl-v2-20261023-01/rl_seconds_beta0.1.pt"}
FEEDBACK = ("listened", "skipped", "simulated")
SEED = 20261024


def ratio_bucket(ratio):
    """Same buckets as full_sft.prepare_feedback: 1..5 for 0, <.2, <.5, <.8, >=.8."""
    r = ratio.clamp(0, 1)
    return (1 + (r > 0).long() + (r >= .2).long() + (r >= .5).long() + (r >= .8).long())


def gap_bucket(seconds):
    return 1 + torch.clamp(torch.floor(torch.log2(seconds.float() + 1)), max=GAP_BUCKETS - 2).long()


def append_event(tensors, action, ratio, gap_seconds):
    """Append one generated event (token, invented feedback, organic 0) to right-padded contexts; drop the oldest if full."""
    x, mask, rb, gb, cr, co = (t.clone() for t in tensors)
    n = mask.sum(1)
    full = n == x.shape[1]
    if full.any():
        for t in (x, mask, rb, gb, cr, co):
            t[full] = torch.roll(t[full], -1, dims=1)
    pos = torch.where(full, n - 1, n)
    rows = torch.arange(len(x), device=x.device)
    x[rows, pos] = action + OFFSET
    mask[rows, pos] = 1
    rb[rows, pos] = ratio_bucket(ratio)
    gb[rows, pos] = gap_bucket(gap_seconds)
    cr[rows, pos] = ratio.clamp(0, 1).to(cr.dtype)
    co[rows, pos] = 0
    return [x, mask, rb, gb, cr, co]


def listened_seconds(ratio, length_seconds):
    """In the logs the gap to the next event is about the listened part of the previous track."""
    return ratio.clamp(0, 1) * length_seconds


def ndcg_at(items, relevant, k=5):
    """Standard NDCG@k of the generated list as shown: positions are kept, a repeated item has zero gain."""
    seen, dcg = set(), 0.0
    for position, item in enumerate(items[:k]):
        if item in relevant and item not in seen:
            dcg += 1 / math.log2(position + 2)
        seen.add(item)
    ideal = sum(1 / math.log2(j + 2) for j in range(min(k, len(relevant))))
    return dcg / ideal


def main(out_name="artifacts/multistep-20261023-01"):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default=out_name, help="output folder relative to the repository root; must not exist")
    out_name = parser.parse_args().out
    out = ROOT / out_name
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    t = proto["time"]
    cutoff = t["query_cutoffs"]["selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "window": "selection", "steps": STEPS,
            "policies": {**POLICIES, "A_rm_rerank_top50": f"reference reweighted on top-{TOP_A} by proxy RM, beta {BETA_A}"},
            "feedback_assumptions": {"listened": "ratio 1", "skipped": "ratio 0",
                                     "simulated": "ratio = judge RM prediction for the generated item in the current context"},
            "gap": "listened part of the previous track: its simulated ratio x capped length (last real event for the first step)", "decoding": ["greedy", "sample (seed shared across policies)"],
            "seed": SEED, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "descriptive; selection already used; nothing selected; final test untouched",
            "caution": "generated items are hypothetical recommendations, not observed interactions"}
    atomic_json(out / "plan.json", plan)
    device = rl_v2.DEVICE = "cuda"
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    length_norm = torch.as_tensor(features[:, 0], device=device)

    def load_policy(path):
        ck = torch.load(ROOT / path, map_location="cpu", weights_only=True)
        m = make_policy(ck["settings"]); m.load_state_dict(ck["model"])
        return m.to(device).eval().requires_grad_(False)

    policies = {n: load_policy(p) for n, p in POLICIES.items()}
    rms = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1", "rm_judge_seed2")}
    uids, _, _, _ = data.queries(cutoff)
    windows = []
    for uid in uids:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], cutoff))
        windows.append((u, max(0, end - data.context_size), end))
    x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
    real = [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]
    real[4] = real[4].float()
    truth = data.truth(lo, hi)
    active = [j for j, uid in enumerate(uids) if uid in truth]
    last_token = real[0][torch.arange(len(uids), device=device), real[1].sum(1) - 1]
    known_last = last_token >= OFFSET
    last_ratio = real[4][torch.arange(len(uids), device=device), real[1].sum(1) - 1].float()
    last_gap = listened_seconds(last_ratio, torch.where(known_last, length_norm[(last_token - OFFSET).clamp(min=0)] * SECONDS,
                                                        torch.full_like(length_norm[:1].expand(len(uids)), 200.0)))
    recent = [set(map(int, u["tokens"][max(0, b - 128):b][u["tokens"][max(0, b - 128):b] >= OFFSET] - OFFSET)) for u, a, b in windows]

    @torch.no_grad()
    def score(name, tensors, idx):
        out_ = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            out_.append(topk_rm_scores(rms[name], *(v[sl] for v in tensors), idx[sl]).float())
        return torch.cat(out_)

    @torch.no_grad()
    def distribution(name, tensors):
        base = "reference" if name == "A_rm_rerank_top50" else name
        logp = rl_v2.policy_logp(policies[base], tensors)
        if name != "A_rm_rerank_top50":
            return logp
        idx = logp.topk(TOP_A, -1).indices
        return rerank_logp(logp, idx, (score("rm_seed0", tensors, idx) + score("rm_seed1", tensors, idx)) / 2)

    report = {"plan": plan, "warm_users": len(uids), "active_users": len(active), "rollouts": {}}
    for name in ["reference", "A_rm_rerank_top50", "B_rl_seconds_beta0.3", "rl_seconds_beta0.1"]:
        first = distribution(name, real)
        static5 = first.topk(5, -1).indices
        first10 = first.topk(10, -1).indices
        static = {"ndcg@5_active": float(np.mean([ndcg_at(data.catalog[static5[j].cpu().numpy()].tolist(), truth[uids[j]]) for j in active])),
                  "judge_completion_real_context": float(score("rm_judge_seed2", real, static5).mean())}
        for decoding in ("greedy", "sample"):
            for feedback in FEEDBACK:
                gen = torch.Generator(device=device).manual_seed(SEED)
                tensors, logp, chosen, steps, gap = real, first, [], [], last_gap
                for step in range(1, STEPS + 1):
                    if step > 1:
                        logp = distribution(name, tensors)
                    a = logp.argmax(-1) if decoding == "greedy" else torch.multinomial(logp.exp(), 1, generator=gen).squeeze(1)
                    judge = score("rm_judge_seed2", tensors, a[:, None]).squeeze(1)
                    proxy = ((score("rm_seed0", tensors, a[:, None]) + score("rm_seed1", tensors, a[:, None])) / 2).squeeze(1)
                    top10 = logp.topk(10, -1).indices
                    overlap = (top10[:, :, None] == first10[:, None, :]).any(-1).float().mean(1)
                    items = a.cpu().numpy()
                    raw = data.catalog[items]
                    steps.append({"step": step,
                                  "judge_completion_imagined": float(judge.mean()),
                                  "judge_equivalent_seconds_imagined": float((judge * length_norm[a]).mean() * SECONDS),
                                  "proxy_completion_imagined": float(proxy.mean()),
                                  "hit_rate_active": float(np.mean([int(raw[j]) in truth[uids[j]] for j in active])),
                                  "seen_in_real_recent128": float(np.mean([int(items[j]) in recent[j] for j in range(len(uids))])),
                                  "repeat_of_earlier_generated": float(np.mean([int(items[j]) in {int(c[j]) for c in chosen} for j in range(len(uids))])) if chosen else 0.0,
                                  "mean_capped_length_seconds": float(length_norm[a].mean() * SECONDS),
                                  "unique_items_across_users": int(a.unique().numel()),
                                  "top10_overlap_with_step1": float(overlap.mean())})
                    chosen.append(items)
                    ratio = {"listened": torch.ones_like(judge), "skipped": torch.zeros_like(judge), "simulated": judge}[feedback]
                    tensors = append_event(tensors, a, ratio, gap)
                    gap = listened_seconds(ratio, length_norm[a] * SECONDS)
                items = torch.as_tensor(np.stack(chosen, 1), device=device)
                real_judge = score("rm_judge_seed2", real, items)
                for s, row in enumerate(steps):
                    row["judge_completion_real_context"] = float(real_judge[:, s].mean())
                    row["imagined_minus_real_judge"] = row["judge_completion_imagined"] - row["judge_completion_real_context"]
                slates = data.catalog[np.stack(chosen, 1)]
                report["rollouts"][f"{name}/{decoding}/{feedback}"] = {
                    "steps": steps, "static_top5_from_real_context": static,
                    "generated_slate_ndcg@5_active": float(np.mean([ndcg_at(slates[j].tolist(), truth[uids[j]]) for j in active])),
                    "slates_with_duplicate": float(np.mean([len(set(s)) < STEPS for s in slates.tolist()]))}
                atomic_json(out / "report.json", report)
                print(name, decoding, feedback, json.dumps(steps[-1]), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["final_test_scored"] = False
    atomic_json(out / "report.json", report)


if __name__ == "__main__":
    main()
