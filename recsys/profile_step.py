"""Where does one training step of full / fullday spend its time? No checkpoints saved.

Wall-clock phases use CUDA synchronization around each phase; the torch.profiler
pass groups GPU kernel time by operator. Weights are random: timing only.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from .data import OFFSET, load_sequences
from .full_sft import (chunked_cross_entropy, chunked_weighted_cross_entropy, day_targets, group_starts_per_event,
                       prepare_day_positives, sample_targets)
from .models import Policy
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def step(model, optimizer, scaler, data, rng, day_rng, day, times, t, batch):
    def mark(name, start):
        torch.cuda.synchronize()
        now = time.perf_counter()
        times.setdefault(name, []).append(now - start)
        return now

    torch.cuda.synchronize()
    s = time.perf_counter()
    windows = []
    x, mask, owners, positions, actions = sample_targets(data, rng, batch, True, windows=windows)
    s = mark("cpu_next_targets", s)
    if day:
        do, dp, da, dw = day_targets(windows, day_rng, *t)
        s = mark("cpu_day_targets", s)
    x, mask = torch.as_tensor(x, device="cuda"), torch.as_tensor(mask, device="cuda")
    owners, positions, actions = (torch.as_tensor(v, device="cuda") for v in (owners, positions, actions))
    if day:
        do, dp, da, dw = (torch.as_tensor(v, device="cuda") for v in (do, dp, da, dw))
    s = mark("host_to_device", s)
    weight = model.transformer.wte.weight[OFFSET:]
    with torch.autocast("cuda", dtype=torch.float16):
        states = model.transformer(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
        s = mark("transformer_forward", s)
        loss = chunked_cross_entropy(states[owners, positions], weight, actions)
        if day:
            loss = 0.5 * loss + 0.5 * chunked_weighted_cross_entropy(states[do, dp], weight, da, dw)
        s = mark("logits_softmax_forward", s)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    s = mark("backward_all", s)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    mark("optimizer", s)
    return len(actions) + (len(da) if day else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    proto = json.loads((ROOT / "research/protocol/case2-v1.json").read_text())
    data, _, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], proto["time"]["train_end_exclusive"])
    t = (proto["time"]["train_end_exclusive"], proto["time"]["gap_seconds"], proto["time"]["horizon_seconds"])
    for user in data.users:
        user["group_start"] = group_starts_per_event(user)
        prepare_day_positives(user, t[0])
    report = {"device": torch.cuda.get_device_name(0), "batch": 64, "note": "random weights; timing only"}
    for arm, day in (("full", False), ("fullday", True)):
        torch.manual_seed(0)
        model = Policy(len(data.catalog), dropout=0.1).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        scaler = torch.amp.GradScaler("cuda")
        rng, day_rng = np.random.default_rng(1), np.random.default_rng(2)
        for _ in range(args.warmup):
            step(model, optimizer, scaler, data, rng, day_rng, day, {}, t, 64)
        times, targets = {}, []
        for _ in range(args.steps):
            targets.append(step(model, optimizer, scaler, data, rng, day_rng, day, times, t, 64))
        total = sum(np.median(v) for v in times.values())
        phases = {k: {"median_ms": 1000 * float(np.median(v)), "share": float(np.median(v) / total)} for k, v in times.items()}
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                step(model, optimizer, scaler, data, rng, day_rng, day, {}, t, 64)
        ops = sorted(prof.key_averages(), key=lambda e: e.device_time_total, reverse=True)
        cuda_total = sum(e.self_device_time_total for e in prof.key_averages())
        top = [{"op": e.key, "self_cuda_ms_per_step": e.self_device_time_total / 5000,
                "self_cuda_share": e.self_device_time_total / cuda_total} for e in
               sorted(prof.key_averages(), key=lambda e: e.self_device_time_total, reverse=True)[:15]]
        report[arm] = {"phases": phases, "median_step_ms": 1000 * total, "mean_targets_per_step": float(np.mean(targets)),
                       "gpu_busy_ms_per_step": cuda_total / 5000, "top_gpu_ops": top,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        del model, optimizer, ops
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    atomic_json(out / "report.json", report)


if __name__ == "__main__":
    main()
