"""Warm-start SFT convergence run, with periodic validation/checkpoints."""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from .data import load_sequences, personal_frequency
from .metrics import ranking_metrics
from .models import Policy
from .pilot import atomic_json, source_manifest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", default="artifacts/calibration-20261021-01/sft-checkpoint.pt")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-steps", type=int, default=4000)
    args = parser.parse_args()
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.FileHandler(out/"run.log", encoding="utf-8")])
    start = time.perf_counter()
    status = {"status": "preparing", "kind": "sft_convergence_run", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args), "final_test_scored": False, "confirmation_scored": False, "new_steps_completed": 0}
    atomic_json(out/"status.json", status)
    keep_awake = False

    def progress(message):
        logging.info(message)
        status["last_event"] = message
        status["elapsed_seconds"] = time.perf_counter()-start
        atomic_json(out/"status.json", status)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if os.name == "nt":
            execution_state = ctypes.windll.kernel32.SetThreadExecutionState
            execution_state.argtypes = [ctypes.c_uint32]
            execution_state.restype = ctypes.c_uint32
            keep_awake = bool(execution_state(0x80000001))
        status["automatic_sleep_inhibited_while_running"] = keep_awake
        torch.set_num_threads(4)
        protocol_file = ROOT/"research/protocol/case2-v1.json"
        protocol = json.loads(protocol_file.read_text())
        protocol_hash = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
        status["protocol_sha256"] = protocol_hash
        status["gpu"] = torch.cuda.get_device_name(0)
        status["optimization"] = {"maximum_lr": 3e-4, "minimum_lr": 3e-5, "warmup_new_steps": 200, "schedule": "linear warmup then cosine", "weight_decay": 0.01, "gradient_clip": 1.0, "mixed_precision": "fp16 with fp32 loss", "min_validation_gain": 0.0001}
        atomic_json(out/"configuration.json", dict(status))
        with zipfile.ZipFile(out/"code-snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
            for folder in ["recsys", "tests", "scripts", "research/protocol"]:
                for p in (ROOT/folder).glob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(ROOT))
        atomic_json(out/"source-manifest.json", source_manifest())
        progress("Loading train-only catalog, sequences and SFT checkpoint")
        data, _, _, _, support = load_sequences(ROOT/protocol["data"]["sample_path"], protocol["time"]["train_end_exclusive"])
        atomic_json(out/"data-support.json", support)
        checkpoint = torch.load(ROOT/args.init, map_location="cuda", weights_only=True)
        if checkpoint["protocol_sha256"] != protocol_hash:
            raise RuntimeError("Checkpoint protocol mismatch")
        if not np.array_equal(checkpoint["catalog"].cpu().numpy(), data.catalog):
            raise RuntimeError("Checkpoint catalog mismatch")
        model = Policy(**checkpoint["settings"]).cuda()
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler = torch.amp.GradScaler("cuda")
        scaler.load_state_dict(checkpoint["scaler"])
        rng = np.random.default_rng()
        rng.bit_generator.state = checkpoint["sampler_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        initial_step = checkpoint.get("step", 0)
        initial_examples = checkpoint.get("sampled_target_examples", initial_step*16)
        del checkpoint
        status["initial_optimizer_step"] = initial_step
        status["initial_sampled_target_examples"] = initial_examples
        cutoff = protocol["time"]["query_cutoffs"]["selection_validation"]
        label_start = cutoff+protocol["time"]["gap_seconds"]
        label_end = label_start+protocol["time"]["horizon_seconds"]
        uids, query_x, query_mask, histories = data.queries(cutoff)
        truth = data.truth(label_start, label_end)
        shuffled = np.random.default_rng(20261021).permutation(len(uids))
        train_probe = data.sample(np.random.default_rng(20261022), 1024)
        val_probe = data.sample(np.random.default_rng(20261023), 2048, bounds=(label_start, label_end))
        baselines = {
            "MostPop": ranking_metrics(uids, [data.catalog[:10].tolist() for _ in uids], truth)[0],
            "PersonalFrequency128": ranking_metrics(uids, personal_frequency(histories, data.catalog, 10), truth)[0],
        }
        atomic_json(out/"baselines.json", baselines)

        @torch.no_grad()
        def evaluate(step):
            model.eval()
            recs, shuffled_recs = [], []
            for offset in range(0, len(uids), args.batch_size):
                section = slice(offset, offset+args.batch_size)
                permutation = shuffled[section]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(torch.as_tensor(query_x[section], device="cuda"), torch.as_tensor(query_mask[section], device="cuda")).float()
                    wrong_logits = model(torch.as_tensor(query_x[permutation], device="cuda"), torch.as_tensor(query_mask[permutation], device="cuda")).float()
                recs.extend(data.catalog[logits.topk(10, dim=-1).indices.cpu().numpy()].tolist())
                shuffled_recs.extend(data.catalog[wrong_logits.topk(10, dim=-1).indices.cpu().numpy()].tolist())
            metrics, per_user = ranking_metrics(uids, recs, truth)
            wrong = ranking_metrics(uids, shuffled_recs, truth)[0]
            losses = {}
            for name, probe in [("train", train_probe), ("validation", val_probe)]:
                loss_sum, hits = 0.0, 0
                for offset in range(0, len(probe[0]), args.batch_size):
                    x, mask, actions = [torch.as_tensor(a[offset:offset+args.batch_size], device="cuda") for a in probe[:3]]
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(x, mask).float()
                    loss_sum += F.cross_entropy(logits, actions, reduction="sum").item()
                    hits += (logits.argmax(-1) == actions).sum().item()
                losses[name+"_fixed_probe_nll"] = loss_sum/len(probe[0])
                losses[name+"_fixed_probe_accuracy"] = hits/len(probe[0])
            result = {"new_steps": step, "total_optimizer_steps": initial_step+step, "sampled_target_examples": initial_examples+step*args.batch_size, "ranking": metrics, "shuffled_history_ranking": wrong, "loss_probes": losses, "elapsed_seconds": time.perf_counter()-start}
            atomic_json(out/f"evaluation-{step:05d}.json", result)
            atomic_json(out/"latest-predictions.json", {"new_steps": step, "uids": uids, "rankings": recs, "per_user": per_user})
            model.train()
            return result

        def save(name, step, best_score):
            content = {"model": model.state_dict(), "settings": model.settings, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "sampler_rng": rng.bit_generator.state, "catalog": torch.from_numpy(data.catalog), "protocol_sha256": protocol_hash, "step": initial_step+step, "sampled_target_examples": initial_examples+step*args.batch_size, "best_selection_ndcg": best_score}
            temporary = out/(name+".tmp")
            torch.save(content, temporary)
            temporary.replace(out/(name+".pt"))

        history = [evaluate(0)]
        best_score = history[0]["ranking"]["NDCG@10"]
        best_step, stale = 0, 0
        save("best", 0, best_score)
        status["status"] = "running"
        status["initial_selection"] = history[0]
        status["best_selection_ndcg"] = best_score
        status["best_new_step"] = best_step
        progress(f"Initial validation NDCG@10={best_score:.6f}; starting up to {args.steps} new steps")
        torch.cuda.reset_peak_memory_stats()
        losses, times, skipped = [], [], 0
        stop_reason = "step_budget"
        for step in range(1, args.steps+1):
            if time.perf_counter()-start > args.max_seconds:
                stop_reason = "wall_time_budget"
                break
            torch.cuda.synchronize()
            t = time.perf_counter()
            if step <= 200:
                factor = 0.1+0.9*step/200
            else:
                factor = 0.1+0.45*(1+math.cos(math.pi*(step-200)/max(1, args.steps-200)))
            lr = 3e-4*factor
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = data.sample(rng, args.batch_size)
            x, mask, actions = [torch.as_tensor(a, device="cuda") for a in batch[:3]]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x, mask).float(), actions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SFT loss")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < previous_scale)
            torch.cuda.synchronize()
            times.append(time.perf_counter()-t)
            losses.append(loss.item())
            status["new_steps_completed"] = step
            status["total_sampled_target_examples"] = initial_examples+step*args.batch_size
            if step % 100 == 0:
                status["recent_mean_train_loss"] = float(np.mean(losses[-100:]))
                status["recent_median_step_seconds"] = float(np.median(times[-100:]))
                status["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                progress(f"SFT {step}/{args.steps}, loss={status['recent_mean_train_loss']:.4f}, lr={lr:.7f}")
            if step % args.eval_every == 0 or step == args.steps:
                result = evaluate(step)
                history.append(result)
                score = result["ranking"]["NDCG@10"]
                if score > best_score+0.0001:
                    best_score, best_step, stale = score, step, 0
                    save("best", step, best_score)
                else:
                    stale += 1
                save("last", step, best_score)
                atomic_json(out/"evaluations.json", history)
                status.update({"best_selection_ndcg": best_score, "best_new_step": best_step, "last_selection": result})
                progress(f"Validation at {step}: NDCG={score:.6f}, shuffled={result['shuffled_history_ranking']['NDCG@10']:.6f}, best={best_score:.6f}")
                if step >= args.min_steps and stale >= args.patience:
                    stop_reason = "validation_plateau"
                    break
        completed = status["new_steps_completed"]
        if not history or history[-1]["new_steps"] != completed:
            history.append(evaluate(completed))
        save("last", completed, best_score)
        atomic_json(out/"evaluations.json", history)
        atomic_json(out/"training-trace.json", {"loss": losses, "step_seconds": times})
        status.update({"status": "completed", "stop_reason": stop_reason, "best_selection_ndcg": best_score, "best_new_step": best_step, "baselines": baselines, "last_selection": history[-1], "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "optimizer_steps_skipped": skipped, "elapsed_seconds": time.perf_counter()-start, "finished_utc": datetime.now(timezone.utc).isoformat(), "cautions": ["Selection validation only; checkpoint selected on this window.", "Shuffle is an OOD sensitivity diagnostic, not a substitute for all personalization tests.", "Batch and LR schedule differ from the short pilot; this is baseline establishment, not an isolated batch-size ablation."]})
        atomic_json(out/"report.json", dict(status))
        progress("SFT RUN COMPLETED")
    except BaseException:
        status.update({"status": "failed", "error": traceback.format_exc(), "elapsed_seconds": time.perf_counter()-start})
        atomic_json(out/"status.json", status)
        logging.exception("SFT run failed")
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
