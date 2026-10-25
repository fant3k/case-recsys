"""Bounded CUDA throughput benchmark; no heldout scores or saved trained policy."""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_sequences
from .models import Policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 64, 128, 256])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root/args.out
    if destination.exists():
        raise FileExistsError(destination)
    protocol = json.loads((root/"research/protocol/case2-v1.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.set_num_threads(4)
    data, _, _, _, _ = load_sequences(root/protocol["data"]["sample_path"], protocol["time"]["train_end_exclusive"])
    rng = np.random.default_rng(1337)
    results = []
    for batch_size in args.batches:
        torch.manual_seed(1337)
        model = Policy(len(data.catalog)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        durations, skips = [], 0
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(15):
                torch.cuda.synchronize()
                started = time.perf_counter()
                x, mask, actions, _, _ = data.sample(rng, batch_size)
                x = torch.as_tensor(x, device="cuda")
                mask = torch.as_tensor(mask, device="cuda")
                actions = torch.as_tensor(actions, device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(model(x, mask).float(), actions)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite benchmark loss")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skips += int(scaler.get_scale() < before)
                torch.cuda.synchronize()
                if step >= 5:
                    durations.append(time.perf_counter()-started)
            result = {"batch_size": batch_size, "warmup_steps": 5, "measured_steps": len(durations), "median_step_seconds": statistics.median(durations), "contexts_per_second": batch_size/statistics.median(durations), "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "optimizer_steps_skipped": skips, "status": "completed"}
        except torch.cuda.OutOfMemoryError:
            result = {"batch_size": batch_size, "status": "OOM"}
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, scaler
        torch.cuda.empty_cache()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"kind": "resource_benchmark_only", "gpu": torch.cuda.get_device_name(0), "precision": "fp16 autocast, fp32 parameters and last-position loss", "model": {"layers": 4, "width": 256, "heads": 4, "context": 128, "catalog": 50000}, "note": "Short in-process allocator peaks; not total card usage or a full-training completion guarantee. Warm caches and one device session limit timing generalization.", "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
