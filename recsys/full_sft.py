"""From-scratch equal-time comparison: last-position target vs all valid positions.

Both arms share initial weights, dropout, LR schedule, batch and anchor sampler.
The only difference: `last` trains one random event of the anchor group from the
final context position (previous recipe); `full` additionally trains every
in-window event, each predicted from the last position of the PREVIOUS
timestamp group, so no member of a tied group is ever visible to its own target.
"""
import argparse
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np
import torch
import torch.utils.checkpoint
from torch.nn import functional as F

from .data import OFFSET, load_sequences, pack_contexts, personal_frequency
from .diagnostics import paired_bootstrap, slice_metrics
from .metrics import ranking_metrics
from .models import GAP_BUCKETS, FeedbackPolicy, Policy, pps_bias
from .multitarget import logits_at_positions
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def group_starts_per_event(user):
    return np.repeat(user["starts"], user["ends"] - user["starts"])


def prepare_feedback(user):
    """Per-event input buckets known once the event itself is over: played ratio and gap to previous event.

    Ratio buckets 1..5: 0, <.2, <.5, <.8, >=.8 (ratios are clipped at 1). Gap bucket 1+floor(log2(dt+1)),
    capped, and GAP_BUCKETS for the user's first event. 0 is padding.
    """
    r = user["ratios"]
    user["ratio_bucket"] = (1 + (r > 0) + (r >= .2) + (r >= .5) + (r >= .8)).astype(np.int64)
    gaps = np.diff(user["times"], prepend=user["times"][0])
    bucket = 1 + np.minimum(GAP_BUCKETS - 2, np.floor(np.log2(gaps + 1.0))).astype(np.int64)
    bucket[0] = GAP_BUCKETS
    user["gap_bucket"] = bucket


def pack_side(slices, context_size):
    out = np.zeros((len(slices), context_size), dtype=np.int64)
    for j, values in enumerate(slices):
        values = np.asarray(values)[-context_size:]
        if not len(values):
            raise ValueError("Empty context has no feedback")
        out[j, :len(values)] = values
    return out


def window_feedback(windows, context_size):
    return (pack_side([u["ratio_bucket"][a:b] for u, a, b in windows], context_size),
            pack_side([u["gap_bucket"][a:b] for u, a, b in windows], context_size))


def sample_targets(data, rng, batch_size, full, audit=None, windows=None, positive_only=False):
    """Return x, mask and flat (owner, source position, action) target arrays.

    `audit`, if a list, receives (user, window start, source position, target event).
    `windows`, if a list, receives (user, window start, window end) per context.
    """
    contexts, owners, positions, actions = [], [], [], []
    for j in range(batch_size):
        user = data.users[int(rng.choice(data.train_users))]
        group = int(rng.choice(user["eligible"]))
        end = int(user["starts"][group])
        start = max(0, end - data.context_size)
        contexts.append(user["tokens"][start:end])
        if windows is not None:
            windows.append((user, start, end))
        anchor = user["known"][user["first"][group]:user["last"][group]]
        if positive_only:
            anchor = anchor[user["ratios"][anchor] > 0.5]
        chosen = anchor if full else ([int(anchor[rng.integers(len(anchor))])] if len(anchor) else [])
        for event in chosen:
            owners.append(j)
            positions.append(end - start - 1)
            actions.append(int(user["tokens"][event]) - OFFSET)
            if audit is not None:
                audit.append((user, start, end - start - 1, int(event)))
        if full:
            events = np.arange(start, end)
            source = user["group_start"][start:end] - 1
            valid = (source >= start) & (user["tokens"][start:end] >= OFFSET)
            if positive_only:
                valid &= user["ratios"][start:end] > 0.5
            owners.extend([j] * int(valid.sum()))
            positions.extend((source[valid] - start).tolist())
            actions.extend((user["tokens"][events[valid]] - OFFSET).tolist())
            if audit is not None:
                audit.extend((user, start, int(src - start), int(e)) for src, e in zip(source[valid], events[valid]))
    x, mask = pack_contexts(contexts, data.context_size)
    return x, mask, np.array(owners), np.array(positions), np.array(actions)


def prepare_day_positives(user, train_end):
    """Train events with played ratio >0.5 and an in-catalog token, in time order."""
    keep = (user["ratios"] > 0.5) & (user["tokens"] >= OFFSET) & (user["times"] < train_end)
    user["positive_times"] = user["times"][keep]
    user["positive_tokens"] = user["tokens"][keep]


def day_targets(windows, rng, train_end, gap, horizon, sources=8, positives=16, audit=None):
    """Next-day positive sets for sampled group-boundary positions of each context.

    A source is the last event of a timestamp group; its query time is the next
    group's timestamp c, labels are unique positives in [c+gap, c+gap+horizon),
    the whole window inside train. Each source gets total weight one.
    """
    owners, positions, actions, weights = [], [], [], []
    for j, (user, start, end) in enumerate(windows):
        times = user["times"]
        index = np.arange(start, end)
        boundary = index[(index + 1 < len(times))]
        boundary = boundary[times[boundary + 1] > times[boundary]]
        query = times[boundary + 1]
        boundary = boundary[query + gap + horizon <= train_end]
        rng.shuffle(boundary)
        used = 0
        for i in boundary:
            if used == sources:
                break
            c = times[i + 1]
            lo, hi = np.searchsorted(user["positive_times"], [c + gap, c + gap + horizon])
            items = np.unique(user["positive_tokens"][lo:hi])
            if not len(items):
                continue
            if len(items) > positives:
                items = rng.choice(items, positives, replace=False)
            used += 1
            owners.extend([j] * len(items))
            positions.extend([int(i - start)] * len(items))
            actions.extend((items - OFFSET).tolist())
            weights.extend([1 / len(items)] * len(items))
            if audit is not None:
                audit.append((user, start, int(i - start), int(c), items))
    return np.array(owners), np.array(positions), np.array(actions), np.array(weights, dtype=np.float32)


def _chunk_weighted_loss(hidden, weight, actions, weights):
    return (F.cross_entropy(F.linear(hidden, weight).float(), actions, reduction="none") * weights).sum()


def chunked_weighted_cross_entropy(hidden, weight, actions, weights, chunk=2048):
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(actions), chunk):
        s = slice(start, start + chunk)
        total = total + torch.utils.checkpoint.checkpoint(_chunk_weighted_loss, hidden[s], weight, actions[s], weights[s], use_reentrant=False)
    return total / weights.sum()


def _chunk_loss(hidden, weight, actions):
    return F.cross_entropy(F.linear(hidden, weight).float(), actions, reduction="sum")


def chunked_cross_entropy(hidden, weight, actions, chunk=2048):
    """Mean CE over all targets; logits are recomputed per chunk in backward."""
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(actions), chunk):
        s = slice(start, start + chunk)
        total = total + torch.utils.checkpoint.checkpoint(_chunk_loss, hidden[s], weight, actions[s], use_reentrant=False)
    return total / len(actions)


def _pps_chunk_loss(hidden, weight, actions, rows, positions, epsilon):
    logits = F.linear(hidden, weight).float() + pps_bias(rows, positions, weight.shape[0], epsilon)
    return F.cross_entropy(logits, actions, reduction="sum")


def chunked_pps_cross_entropy(hidden, weight, actions, rows, positions, epsilon, chunk=1024):
    """Mean CE with the fixed PPS prior added to every chunk's logits; recomputed in backward."""
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(actions), chunk):
        s = slice(start, start + chunk)
        total = total + torch.utils.checkpoint.checkpoint(_pps_chunk_loss, hidden[s], weight, actions[s], rows[s],
                                                          positions[s], epsilon, use_reentrant=False)
    return total / len(actions)


def lr_factor(fraction):
    warm = 0.03
    if fraction < warm:
        return 0.1 + 0.9 * fraction / warm
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (fraction - warm) / (1 - warm))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds-per-arm", type=float, default=1500)
    parser.add_argument("--eval-every-seconds", type=float, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--context-size", type=int, default=128,
                        help="GPT-2 context window (128 reproduces every earlier run; docs/CONTEXT-PLAN.md varies it)")
    parser.add_argument("--target-end", type=int, default=None,
                        help="train only on target groups (and day windows) before this time; catalog stays train-only")
    parser.add_argument("--arms", default="last,full",
                        help="comma list of last|full, optionally last|full:dropout:weight_decay")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": proto_hash,
            "arguments": vars(args), "init_seed": 20261101, "sampling_seed": 20261102,
            "model": f"GPT-2 4 layers, width 256, 4 heads, context {args.context_size}, tied embeddings, from scratch",
            "optimizer": "AdamW, per-arm weight decay (default 0.01), clip 1, fp16 autocast, fp32 CE, time-based warmup 3% then cosine to 10%",
            "difference": "last: one anchor event; full: all anchor-group events + every in-window event from its previous-group position; fullday: full next-item CE (weight .5) + day CE (weight .5) over unique >50% positives in [c+gap, c+gap+horizon) for up to 8 group-boundary sources per context, up to 16 sampled positives each, every source weight 1, day window inside train, separate RNG so contexts match full; +pps: fixed ln(1+c/0.01) prior on logits of items counted in the prefix up to the source position, in training and inference (Abbattista et al. RecSys 2024 PPS, softmax form); +fb: input embeddings of each past event's played-ratio bucket and log2 gap to its previous event; +pos: only targets with played ratio >0.5",
            "loss_weighting": "mean over all CE targets in the batch",
            "selection": "best day NDCG among periodic evaluations per arm (exploratory, repeatedly used selection window)",
            "scope": "selection only; confirmation and final test are not scored; no RM/RL"}
    atomic_json(out / "plan.json", plan)
    status = {"status": "running", "stage": "loading", "pid": os.getpid(), "final_test_scored": False, "confirmation_scored": False}
    atomic_json(out / "status.json", status)
    with zipfile.ZipFile(out / "code-snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for folder in ["recsys", "tests", "research/protocol"]:
            for p in (ROOT / folder).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(ROOT))
    keep_awake = False

    def progress(**fields):
        status.update(fields)
        status["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(out / "status.json", status)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if os.name == "nt":
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001)))
        torch.set_num_threads(4)
        data, _, _, _, support = load_sequences(ROOT / proto["data"]["sample_path"], proto["time"]["train_end_exclusive"], context_size=args.context_size)
        train_end = proto["time"]["train_end_exclusive"]
        if args.target_end is not None:
            if args.target_end > train_end:
                raise ValueError("target end after train end")
            train_end = args.target_end
            data.train_users = []
            for j, user in enumerate(data.users):
                user["eligible"] = user["eligible"][user["times"][user["starts"][user["eligible"]]] < train_end]
                if len(user["eligible"]):
                    data.train_users.append(j)
                assert np.all(user["times"][user["starts"][user["eligible"]]] < train_end)
        gap, horizon = proto["time"]["gap_seconds"], proto["time"]["horizon_seconds"]
        for user in data.users:
            user["group_start"] = group_starts_per_event(user)
            prepare_day_positives(user, train_end)
        cutoff = proto["time"]["query_cutoffs"]["selection_validation"]
        lo = cutoff + proto["time"]["gap_seconds"]
        hi = lo + proto["time"]["horizon_seconds"]
        uids, qx, qm, histories = data.queries(cutoff)
        truth = data.truth(lo, hi)
        val_probe = data.sample(np.random.default_rng(20261023), 2048, bounds=(lo, hi))
        for user in data.users:
            prepare_feedback(user)
        by_uid = {u["uid"]: u for u in data.users}
        query_windows = []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            query_windows.append((u, max(0, end - data.context_size), end))
        q_ratio, q_gap = window_feedback(query_windows, data.context_size)
        probe_windows = []
        for uid, event, event_time in val_probe[-1]:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], event_time))
            probe_windows.append((u, max(0, end - data.context_size), end))
        p_ratio, p_gap = window_feedback(probe_windows, data.context_size)
        assert np.array_equal(q_ratio > 0, qm > 0) and np.array_equal(p_ratio > 0, val_probe[1] > 0)
        full_seen, recent_seen = {}, {}
        for user in data.users:
            end = int(np.searchsorted(user["times"], cutoff))
            full_seen[user["uid"]] = set(map(int, user["items"][:end]))
            recent_seen[user["uid"]] = set(map(int, user["items"][max(0, end - 128):end]))
        frequency_recs = personal_frequency(histories, data.catalog, 10)
        frequency, frequency_rows = ranking_metrics(uids, frequency_recs, truth)
        report = {"plan": plan, "data_support": support, "PersonalFrequency128": frequency, "arms": {},
                  "previous_reference": {"sft-20261021-01 best (single target, ~13.5 min)": 0.05965847,
                                         "day-objective best": 0.06537783},
                  "confirmation_scored": False, "final_test_scored": False}

        for spec in args.arms.split(","):
            kind, *values = spec.split(":")
            base, *flags = kind.split("+")
            if base not in ("last", "full", "fullday") or not set(flags) <= {"pps", "fb", "pos"}:
                raise ValueError("Unknown arm "+spec)
            use_pps, use_fb, use_pos = "pps" in flags, "fb" in flags, "pos" in flags
            dropout, weight_decay = map(float, values) if values else (args.dropout, args.weight_decay)
            arm = kind if not values else f"{kind}-d{dropout:g}-wd{weight_decay:g}"
            full = base in ("full", "fullday")
            day = base == "fullday"
            arm_dir = out / arm
            arm_dir.mkdir()
            torch.manual_seed(plan["init_seed"])
            if use_pps or use_fb:
                model = FeedbackPolicy(len(data.catalog), context_size=args.context_size, dropout=dropout, pps=use_pps, feedback=use_fb).cuda()
            else:
                model = Policy(len(data.catalog), context_size=args.context_size, dropout=dropout).cuda()
            extended = isinstance(model, FeedbackPolicy)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
            scaler = torch.amp.GradScaler("cuda")
            rng = np.random.default_rng(plan["sampling_seed"])
            day_rng = np.random.default_rng(plan["sampling_seed"] + 1)
            day_targets_total = 0

            @torch.no_grad()
            def evaluate():
                model.eval()
                recs = []
                for offset in range(0, len(uids), 128):
                    s = slice(offset, offset + 128)
                    side = (torch.as_tensor(q_ratio[s], device="cuda"), torch.as_tensor(q_gap[s], device="cuda")) if extended else ()
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = model(torch.as_tensor(qx[s], device="cuda"), torch.as_tensor(qm[s], device="cuda"), *side).float()
                    recs.extend(data.catalog[logits.topk(10, dim=-1).indices.cpu().numpy()].tolist())
                nll = 0.0
                for offset in range(0, 2048, 256):
                    x, mask, a = [torch.as_tensor(v[offset:offset + 256], device="cuda") for v in val_probe[:3]]
                    side = (torch.as_tensor(p_ratio[offset:offset + 256], device="cuda"),
                            torch.as_tensor(p_gap[offset:offset + 256], device="cuda")) if extended else ()
                    with torch.autocast("cuda", dtype=torch.float16):
                        nll += F.cross_entropy(model(x, mask, *side).float(), a, reduction="sum").item()
                model.train()
                metrics, rows = ranking_metrics(uids, recs, truth)
                return metrics, rows, recs, nll / 2048

            trained, steps, targets, losses, skipped = 0.0, 0, 0, [], 0
            history, best = [], None
            next_eval = 0.0
            model.train()
            torch.cuda.reset_peak_memory_stats()
            while True:
                if trained >= next_eval:
                    metrics, rows, recs, nll = evaluate()
                    entry = {"training_seconds": trained, "steps": steps, "ce_targets": targets,
                             "NDCG@10": metrics["NDCG@10"], "Recall@10": metrics["Recall@10"],
                             "validation_probe_nll": nll,
                             "recent_train_loss": float(np.mean(losses[-200:])) if losses else None}
                    history.append(entry)
                    atomic_json(arm_dir / "evaluations.json", history)
                    if best is None or metrics["NDCG@10"] > best["NDCG@10"]:
                        best = dict(entry, rows=rows, recs=recs)
                        torch.save({"model": model.state_dict(), "settings": model.settings,
                                    "catalog": torch.from_numpy(data.catalog), "protocol_sha256": proto_hash,
                                    "steps": steps, "ce_targets": targets, "arm": arm, "target_end": train_end}, arm_dir / "best.pt")
                    progress(stage=arm, arm_training_seconds=trained, arm_steps=steps, last_eval=entry,
                             best_ndcg={**status.get("best_ndcg", {}), arm: best["NDCG@10"]})
                    next_eval += args.eval_every_seconds
                    if trained >= args.seconds_per_arm:
                        break
                torch.cuda.synchronize()
                t = time.perf_counter()
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * lr_factor(trained / args.seconds_per_arm)
                windows = []
                x, mask, owners, positions, actions = sample_targets(data, rng, args.batch_size, full, windows=windows,
                                                                     positive_only=use_pos)
                if not len(actions):
                    continue
                side = tuple(torch.as_tensor(v, device="cuda") for v in window_feedback(windows, data.context_size)) if use_fb else (None, None)
                x, mask = torch.as_tensor(x, device="cuda"), torch.as_tensor(mask, device="cuda")
                owners_t, positions_t = torch.as_tensor(owners, device="cuda"), torch.as_tensor(positions, device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    if extended:
                        day_states = model.states(x, mask, *side)
                    else:
                        day_states = model.transformer(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state
                    hidden = day_states[owners_t, positions_t]
                    if use_pps:
                        loss = chunked_pps_cross_entropy(hidden, model.transformer.wte.weight[OFFSET:],
                                                         torch.as_tensor(actions, device="cuda"), x[owners_t], positions_t,
                                                         model.settings["pps_epsilon"])
                    else:
                        loss = chunked_cross_entropy(hidden, model.transformer.wte.weight[OFFSET:],
                                                     torch.as_tensor(actions, device="cuda"))
                    if day:
                        do, dp, da, dw = day_targets(windows, day_rng, train_end, gap, horizon)
                        if len(da):
                            day_hidden = day_states[torch.as_tensor(do, device="cuda"), torch.as_tensor(dp, device="cuda")]
                            day_loss = chunked_weighted_cross_entropy(day_hidden, model.transformer.wte.weight[OFFSET:],
                                                                      torch.as_tensor(da, device="cuda"), torch.as_tensor(dw, device="cuda"))
                            loss = 0.5 * loss + 0.5 * day_loss
                            day_targets_total += len(da)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < previous)
                torch.cuda.synchronize()
                trained += time.perf_counter() - t
                steps += 1
                targets += len(actions)
                losses.append(loss.item())
                if steps % 200 == 0:
                    progress(arm_training_seconds=trained, arm_steps=steps,
                             recent_train_loss=float(np.mean(losses[-200:])))
            torch.save({"model": model.state_dict(), "settings": model.settings,
                        "catalog": torch.from_numpy(data.catalog), "protocol_sha256": proto_hash,
                        "steps": steps, "ce_targets": targets, "arm": arm, "target_end": train_end}, arm_dir / "last.pt")
            sliced, _ = slice_metrics(uids, best["recs"], truth, full_seen, recent_seen, data.catalog)
            report["arms"][arm] = {"dropout": dropout, "weight_decay": weight_decay,
                "pps": use_pps, "feedback_inputs": use_fb, "positive_targets_only": use_pos,
                "history": history, "best": {k: v for k, v in best.items() if k not in ("rows", "recs")},
                "slices_at_best": sliced, "steps": steps, "ce_targets": targets,
                "day_targets": day_targets_total, "skipped_scaler_steps": skipped, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "minus_frequency128": paired_bootstrap(best["rows"], frequency_rows)}
            report["arms"][arm]["_rows"] = best["rows"]
            del model, optimizer
            torch.cuda.empty_cache()
        if "last" in report["arms"] and "full" in report["arms"]:
            report["full_minus_last"] = paired_bootstrap(report["arms"]["full"]["_rows"], report["arms"]["last"]["_rows"])
        names = list(report["arms"])
        report["minus_first_arm"] = {name: paired_bootstrap(report["arms"][name]["_rows"], report["arms"][names[0]]["_rows"])
                                     for name in names[1:]}
        for arm in report["arms"].values():
            arm.pop("_rows")
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(out / "report.json", report)
        progress(status="completed", stage="done")
    except BaseException:
        progress(status="failed", error=traceback.format_exc())
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000000))


if __name__ == "__main__":
    main()
