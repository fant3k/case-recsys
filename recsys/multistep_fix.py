"""Does removing the self-reinforcement of generated tracks help a five-track list? (docs/MULTISTEP-FIX-PLAN.md)

In sequential generation each chosen track is appended to the context as if it had been listened to, so the PPS
repeat prior counts it and pushes it up again (docs/MULTISTEP-V2-RESULTS.md). Variants on the same policies and the
selection window, greedy decoding, simulated feedback (judge's predicted completion), no new training:
  seq        existing sequential generation
  seq_ban    seq, a track already in the list cannot be chosen again
  nopps      generated events do not enter the PPS counts (counts come from the real history only)
  nopps_ban  nopps plus the ban
  static_top5  the five best tracks of the real context (control, no generation)
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts
from .final_test import CHECKPOINTS, SECONDS, interval, sha
from .full_sft import prepare_feedback
from .models import make_policy, pps_bias
from .multistep import POLICIES, append_event, listened_seconds, ndcg_at
from .pilot import atomic_json
from .reward_v2 import side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
STEPS, SEED = 5, 20261501


def step_logits(model, tensors, real, synthetic_in_pps=True):
    """Policy logits for the next track. With synthetic_in_pps=False the PPS prior counts only the real history."""
    x, mask, ratio, gap = tensors[:4]
    hidden = model.states(x, mask, ratio, gap)
    position = mask.sum(1) - 1
    logits = F.linear(hidden[torch.arange(len(x), device=x.device), position], model.transformer.wte.weight[OFFSET:]).float()
    if model.settings["pps"]:
        source, source_position = (x, position) if synthetic_in_pps else (real[0], real[1].sum(1) - 1)
        logits = logits + pps_bias(source, source_position, logits.shape[1], model.settings["pps_epsilon"])
    return logits


def choose_next(logits, chosen, ban):
    """Greedy choice; with `ban`, tracks already in the list get -inf."""
    logits = logits.clone()
    if ban:
        for previous in chosen:
            logits[torch.arange(len(logits), device=logits.device), previous] = float("-inf")
    return logits.argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/multistep-fix-selection-20261025-01")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    t = proto["time"]
    cutoff = t["query_cutoffs"]["selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    device = rl_v2.DEVICE = "cuda"
    torch.set_num_threads(4)
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    length_norm = torch.as_tensor(features[:, 0], device=device)
    atomic_json(out / "plan.json", {"plan_document": "docs/MULTISTEP-FIX-PLAN.md", "protocol_sha256": sha(proto_path), "window": "selection",
                                    "steps": STEPS, "decoding": "greedy", "feedback": "simulated: judge (RM v2 seed 2) predicted completion",
                                    "policies": POLICIES, "scope": "descriptive; selection only; no training"})

    def load_policy(path):
        ck = torch.load(ROOT / path, map_location="cpu", weights_only=True)
        m = make_policy(ck["settings"])
        m.load_state_dict(ck["model"])
        return m.to(device).eval().requires_grad_(False)

    policies = {n: load_policy(p) for n, p in POLICIES.items() if n in ("reference", "B_rl_seconds_beta0.3")}
    judge = rl_v2.load_rm(ROOT / CHECKPOINTS["rm_judge_seed2"], features)[0]
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
    rows = torch.arange(len(uids), device=device)
    last_token = real[0][rows, real[1].sum(1) - 1]
    last_gap = listened_seconds(real[4][rows, real[1].sum(1) - 1], torch.where(
        last_token >= OFFSET, length_norm[(last_token - OFFSET).clamp(min=0)] * SECONDS, torch.full_like(length_norm[:1].expand(len(uids)), 200.0)))

    @torch.no_grad()
    def judge_score(tensors, idx):
        parts = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            parts.append(topk_rm_scores(judge, *(v[sl] for v in tensors), idx[sl]).float())
        return torch.cat(parts)

    def summarize(slates, name):
        slates = np.asarray(slates)
        per = np.array([ndcg_at(data.catalog[slates[j]].tolist(), truth[uids[j]]) for j in active])
        hits = np.array([len({int(i) for i in data.catalog[slates[j]]} & truth[uids[j]]) for j in active])
        return per, {"ndcg@5_active": float(per.mean()), "hits_in_slate_active": float(hits.mean()),
                     "users_with_a_hit_active": float((hits > 0).mean()),
                     "unique_tracks_per_slate": float(np.mean([len(set(s)) for s in slates.tolist()])),
                     "slates_with_duplicate": float(np.mean([len(set(s)) < STEPS for s in slates.tolist()])),
                     "unique_tracks_across_users": int(len(np.unique(slates)))}

    report = {"warm_users": len(uids), "active_users": len(active), "policies": {}}
    with torch.no_grad():
        for name, model in policies.items():
            entry = {}
            first = step_logits(model, real, real)
            static = first.topk(STEPS, -1).indices.cpu().numpy()
            base_per, entry["static_top5"] = summarize(static, "static")
            pers = {"static_top5": base_per}
            for variant, (synthetic_in_pps, ban) in {"seq": (True, False), "seq_ban": (True, True),
                                                      "nopps": (False, False), "nopps_ban": (False, True)}.items():
                tensors, gap, chosen = real, last_gap, []
                for step in range(STEPS):
                    logits = first if step == 0 else step_logits(model, tensors, real, synthetic_in_pps)
                    a = choose_next(logits, chosen, ban)
                    ratio = judge_score(tensors, a[:, None]).squeeze(1)
                    chosen.append(a)
                    tensors = append_event(tensors, a, ratio, gap)
                    gap = listened_seconds(ratio, length_norm[a] * SECONDS)
                slates = torch.stack(chosen, 1).cpu().numpy()
                per, entry[variant] = summarize(slates, variant)
                pers[variant] = per
                entry[variant]["ndcg@5_minus_static_top5_95"] = interval(per - base_per, 0.95, seed=SEED)
                entry[variant]["mean_overlap_with_static_top5"] = float(np.mean([len(set(slates[j]) & set(static[j])) / STEPS for j in range(len(uids))]))
            entry["paired_ndcg@5_95"] = {f"{x}_minus_{y}": interval(pers[x] - pers[y], 0.95, seed=SEED)
                                         for x, y in (("seq_ban", "seq"), ("nopps", "seq"), ("nopps_ban", "seq_ban"), ("nopps_ban", "nopps"))}
            report["policies"][name] = entry
            atomic_json(out / "report.json", report)
            for variant, v in entry.items():
                if variant == "paired_ndcg@5_95":
                    continue
                print(name, variant, round(v["ndcg@5_active"], 4), "unique", round(v["unique_tracks_per_slate"], 2), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(out / "report.json", report)
    atomic_json(out / "status.json", {"status": "completed"})


if __name__ == "__main__":
    main()
