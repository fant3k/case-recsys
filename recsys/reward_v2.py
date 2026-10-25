"""Reward model v2: GPT-2 backbone, repeat features, target-origin condition, ensemble.

Target (default): clipped played ratio of a logged listen, as in v1. Differences:
- history is encoded by a copy of the trained feedback GPT-2 (separate parameters
  from any policy), so past played ratios and listening gaps are visible;
- candidate features from the context strictly before the target group: count,
  position of last occurrence, user-item mean ratio, plus train item length and
  popularity and user context summaries;
- the target event's is_organic flag is an INPUT condition. A recommendation is by
  definition non-organic, so RL scores every action with flag 0. The flag is known
  at recommendation time and is not a leak of the response.
Alternative targets (continuation, quantile) come from recsys/targets.py.
Evaluation adds within-user PairAccuracy (does the RM order one user's own listens
correctly), which is what an RL policy choosing among items for one user relies on.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import OFFSET, load_sequences, pack_contexts
from .full_sft import prepare_feedback, window_feedback
from .models import FeedbackPolicy, RewardModel, make_policy
from .pilot import atomic_json
from .reward_audit import calibration, observed_records, regression
from .targets import continuation_labels, event_lengths, quantile_labels

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda"
N_FEATURES = 9


def candidate_features(rows, ratios, organic, actions, item_features):
    """Features of `actions` (n) given context rows (n x L, right padded) and their ratios/organic flags.

    Uses only context positions, i.e. history strictly before the target group.
    """
    known = rows >= OFFSET
    length = known.sum(1).clamp(min=1).float()
    match = known & ((rows - OFFSET) == actions[:, None])
    count = match.sum(1).float()
    positions = torch.arange(rows.shape[1], device=rows.device)[None].expand_as(rows)
    filled = (rows > 0).sum(1)
    last = torch.where(match, positions, torch.full_like(positions, -1)).max(1).values
    recency = torch.where(last >= 0, (filled - 1 - last).float() / rows.shape[1], torch.full_like(count, 1.5))
    ui_mean = torch.where(count > 0, (ratios * match).sum(1) / count.clamp(min=1), torch.zeros_like(count))
    user_mean = (ratios * known).sum(1) / length
    organic_share = (organic.float() * known).sum(1) / length
    item = item_features[actions]
    return torch.stack([torch.log1p(count), recency, (count > 0).float(), ui_mean, item[:, 0], item[:, 1],
                        user_mean, organic_share, torch.log1p(length) / 5], 1)


def all_candidate_features(rows, ratios, organic, item_features):
    """Same features for every catalog item, rows n x L -> n x C x N_FEATURES (memory: chunk n)."""
    n, size = len(rows), len(item_features)
    known = rows >= OFFSET
    idx = (rows - OFFSET).clamp(min=0)
    count = torch.zeros(n, size, device=rows.device).scatter_add_(1, idx, known.float())
    ratio_sum = torch.zeros(n, size, device=rows.device).scatter_add_(1, idx, ratios * known)
    filled = (rows > 0).sum(1)
    positions = torch.arange(rows.shape[1], device=rows.device)[None].expand_as(rows)
    last = torch.full((n, size), -1, device=rows.device, dtype=torch.long)
    last.scatter_reduce_(1, idx, torch.where(known, positions, torch.full_like(positions, -1)), reduce="amax")
    recency = torch.where(last >= 0, (filled[:, None] - 1 - last).float() / rows.shape[1], torch.full_like(count, 1.5))
    length = known.sum(1).clamp(min=1).float()
    user_mean = ((ratios * known).sum(1) / length)[:, None].expand(n, size)
    organic_share = ((organic.float() * known).sum(1) / length)[:, None].expand(n, size)
    item = item_features[None].expand(n, size, 2)
    return torch.stack([torch.log1p(count), recency, (count > 0).float(),
                        torch.where(count > 0, ratio_sum / count.clamp(min=1), torch.zeros_like(count)),
                        item[..., 0], item[..., 1], user_mean, organic_share,
                        (torch.log1p(length) / 5)[:, None].expand(n, size)], 2)


class RewardModelV2(nn.Module):
    def __init__(self, backbone, item_features, use_features=True, hidden=256):
        super().__init__()
        self.backbone = backbone
        self.use_features = use_features
        width = backbone.settings["width"]
        self.register_buffer("item_features", torch.as_tensor(item_features, dtype=torch.float32))
        self.head = nn.Sequential(nn.Linear(3 * width + N_FEATURES + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def encode(self, x, mask, ratio_bucket, gap_bucket):
        states = self.backbone.states(x, mask, ratio_bucket, gap_bucket)
        return states[torch.arange(len(x), device=x.device), mask.sum(1) - 1].float()

    def score(self, h, candidate, feats, organic_flag):
        if not self.use_features:
            feats = torch.zeros_like(feats)
        return torch.sigmoid(self.head(torch.cat([h, candidate, h * candidate, feats, organic_flag[:, None]], 1)).squeeze(1))

    def forward(self, x, mask, ratio_bucket, gap_bucket, ctx_ratio, ctx_organic, actions, organic_flag):
        h = self.encode(x, mask, ratio_bucket, gap_bucket)
        candidate = self.backbone.transformer.wte.weight[actions + OFFSET].float()
        feats = candidate_features(x, ctx_ratio, ctx_organic, actions, self.item_features)
        return self.score(h, candidate, feats, organic_flag)

    @torch.no_grad()
    def all_scores(self, x, mask, ratio_bucket, gap_bucket, ctx_ratio, ctx_organic, organic_flag=0.0, chunk=8):
        """Predicted response for every catalog item; organic_flag fixed (0 for recommendations)."""
        out = []
        weight = self.backbone.transformer.wte.weight[OFFSET:].float()
        for s in range(0, len(x), chunk):
            sl = slice(s, s + chunk)
            h = self.encode(x[sl], mask[sl], ratio_bucket[sl], gap_bucket[sl])
            feats = all_candidate_features(x[sl], ctx_ratio[sl], ctx_organic[sl], self.item_features)
            n, size = feats.shape[:2]
            hh = h[:, None].expand(n, size, -1)
            cand = weight[None].expand(n, size, -1)
            if not self.use_features:
                feats = torch.zeros_like(feats)
            flag = torch.full((n, size, 1), float(organic_flag), device=x.device)
            out.append(torch.sigmoid(self.head(torch.cat([hh, cand, hh * cand, feats, flag], 2)).squeeze(2)))
        return torch.cat(out)


def side_inputs(windows, context_size):
    ratio_bucket, gap_bucket = window_feedback(windows, context_size)
    ctx_ratio = np.zeros((len(windows), context_size), dtype=np.float32)
    ctx_organic = np.zeros((len(windows), context_size), dtype=np.int64)
    for j, (u, a, b) in enumerate(windows):
        r, o = u["ratios"][a:b][-context_size:], u["organic"][a:b][-context_size:]
        ctx_ratio[j, :len(r)] = r
        ctx_organic[j, :len(o)] = o
    return ratio_bucket, gap_bucket, ctx_ratio, ctx_organic


LABEL_TOLERANCE = 1e-6  # float32 labels: 0.7 - 0.2 is 0.49999998, still a 50-point pair


def pair_accuracy(predictions, labels, uids, margin=0.5):
    """Within-user pairs with |label difference| >= margin: share ordered correctly (ties count half)."""
    order = {}
    for p, y, u in zip(predictions, labels, uids):
        order.setdefault(int(u), []).append((p, y))
    per_user, pooled_hits, pooled_pairs = [], 0.0, 0
    for rows in order.values():
        p = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        dy = y[:, None] - y[None, :]
        dp = p[:, None] - p[None, :]
        take = dy >= margin - LABEL_TOLERANCE
        if not take.any():
            continue
        hits = (dp[take] > 0) + 0.5 * (dp[take] == 0)
        per_user.append(hits.mean())
        pooled_hits += hits.sum(); pooled_pairs += take.sum()
    return {"users_with_pairs": len(per_user), "pairs": int(pooled_pairs),
            "macro_pair_accuracy": float(np.mean(per_user)) if per_user else None,
            "pooled_pair_accuracy": float(pooled_hits / pooled_pairs) if pooled_pairs else None, "margin": margin}


def per_user_pair_accuracy(predictions, labels, uids, margin=0.5):
    out = {}
    for u in np.unique(uids):
        take = uids == u
        p, y = predictions[take], labels[take]
        dy = y[:, None] - y[None, :]
        sel = dy >= margin - LABEL_TOLERANCE
        if sel.any():
            dp = (p[:, None] - p[None, :])[sel]
            out[int(u)] = float(((dp > 0) + 0.5 * (dp == 0)).mean())
    return out


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="artifacts/pps-20261022-01/full+pps+fb-d0.1-wd0.01/last.pt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ensemble", type=int, default=5)
    parser.add_argument("--target", default="completion", choices=["completion", "continuation", "quantile"],
                        help="completion: clipped played ratio; continuation/quantile: see recsys/targets.py")
    parser.add_argument("--device", default="cuda", help="cpu only for smoke checks")
    parser.add_argument("--v1-rm", default="artifacts/reward-audit-20261021-01/history-last.pt")
    args = parser.parse_args()
    DEVICE = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": proto_hash,
            "backbone_init": args.backbone, "target": args.target, "steps": args.steps, "batch": args.batch_size,
            "ensemble": args.ensemble,
            "optimizer": "AdamW, backbone lr 1e-4, head lr 1e-3, wd 0.01, clip 1, fp32, fixed last step",
            "arms": ["v2 full: backbone + repeat features + organic condition, ensemble seeds 0..K-1",
                     "v2 no-features: backbone + organic condition, seed 0 (ablation)"],
            "evaluation": "all eligible selection-day in-catalog logged events (as reward audit v1); RMs fixed, nothing selected on selection",
            "pair_accuracy": "within-user pairs with label difference >= 0.5",
            "scope": "confirmation and final test untouched"}
    atomic_json(out / "plan.json", plan)
    atomic_json(out / "status.json", {"status": "running", "stage": "loading"})
    try:
        torch.set_num_threads(4)
        data, features, item_means, global_mean, _ = load_sequences(ROOT / proto["data"]["sample_path"], proto["time"]["train_end_exclusive"])
        train_end = proto["time"]["train_end_exclusive"]
        cut = proto["time"]["query_cutoffs"]["selection_validation"]
        lo = cut + proto["time"]["gap_seconds"]; hi = lo + proto["time"]["horizon_seconds"]
        mean_length_norm = float(features[:, 0].mean())
        for u in data.users:
            prepare_feedback(u)
            if args.target == "completion":
                u["train_label"] = u["eval_label"] = u["ratios"]
            elif args.target == "continuation":
                u["train_label"] = continuation_labels(u, train_end)
                u["eval_label"] = continuation_labels(u, hi)
            else:
                u["train_label"] = u["eval_label"] = quantile_labels(u, event_lengths(u, features, OFFSET, mean_length_norm))
        by_uid = {u["uid"]: u for u in data.users}
        ck = torch.load(ROOT / args.backbone, map_location="cpu", weights_only=True)
        if ck["protocol_sha256"] != proto_hash or not np.array_equal(ck["catalog"].numpy(), data.catalog):
            raise ValueError("Backbone catalog/protocol mismatch")
        base = make_policy(ck["settings"])
        base.load_state_dict(ck["model"])
        if not isinstance(base, FeedbackPolicy) or not base.settings["feedback"]:
            raise ValueError("Backbone must have feedback inputs")

        def windows_for(records):
            ws = []
            for uid, _, t in records:
                u = by_uid[uid]
                end = int(np.searchsorted(u["times"], t))
                ws.append((u, max(0, end - data.context_size), end))
            return ws

        def tensors(x, mask, windows, actions, flags):
            rb, gb, cr, co = side_inputs(windows, data.context_size)
            return [torch.as_tensor(v, device=DEVICE) for v in (x, mask, rb, gb, cr, co, actions)] + \
                   [torch.as_tensor(np.asarray(flags, dtype=np.float32), device=DEVICE)]

        arms = [(f"v2_seed{k}", True, k) for k in range(args.ensemble)] + [("v2_no_features_seed0", False, 0)]
        models, traces = {}, {}
        for name, use_features, seed in arms:
            torch.manual_seed(20261110 + seed)
            model = RewardModelV2(copy.deepcopy(base), features, use_features=use_features).to(DEVICE)
            optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 1e-4},
                                           {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
            rng = np.random.default_rng(20261120 + seed)
            trace, t0 = [], time.perf_counter()
            model.train()
            for step in range(1, args.steps + 1):
                x, mask, actions, _, records = data.sample(rng, args.batch_size)
                labels = np.array([by_uid[uid]["train_label"][event] for uid, event, _ in records], dtype=np.float32)
                flags = [by_uid[uid]["organic"][event] for uid, event, _ in records]
                inputs = tensors(x, mask, windows_for(records), actions, flags)
                loss = F.mse_loss(model(*inputs), torch.as_tensor(labels, device=DEVICE))
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite RM loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if step % 500 == 0:
                    trace.append({"step": step, "batch_mse": loss.item(), "seconds": time.perf_counter() - t0})
                    atomic_json(out / "status.json", {"status": "running", "stage": name, "step": step,
                                                      "elapsed_seconds": time.perf_counter() - started})
            model.eval().requires_grad_(False)
            torch.save({"model": model.state_dict(), "backbone_settings": base.settings, "use_features": use_features,
                        "catalog": torch.from_numpy(data.catalog), "protocol_sha256": proto_hash, "steps": args.steps,
                        "seed": seed, "target": args.target}, out / f"{name}.pt")
            models[name], traces[name] = model, trace

        old_ck = torch.load(ROOT / args.v1_rm, map_location="cpu", weights_only=True)
        if old_ck["protocol_sha256"] != proto_hash or not np.array_equal(old_ck["catalog"].numpy(), data.catalog):
            raise ValueError("v1 RM mismatch")
        v1 = RewardModel(len(data.catalog), features, width=old_ck["width"]).to(DEVICE)
        v1.load_state_dict(old_ck["model"]); v1.eval().requires_grad_(False)

        records, raw_events = observed_records(data, lo, hi)
        uid = np.array([r[0] for r in records]); action = np.array([r[3] for r in records])
        label = np.array([by_uid[r[0]]["eval_label"][r[1]] for r in records], dtype=np.float64)
        organic = np.array([by_uid[r[0]]["organic"][r[1]] for r in records])
        kinds = np.array([r[5] for r in records])
        pred = {n: [] for n in list(models) + ["v1_history_RM", "v2_seed0_flag_forced_0", "v2_seed0_flag_forced_1"]}
        with torch.inference_mode():
            for j in range(0, len(records), 256):
                sub = records[j:j + 256]
                ws = [(by_uid[r[0]], max(0, r[2] - data.context_size), r[2]) for r in sub]
                x, mask = pack_contexts([w[0]["tokens"][w[1]:w[2]] for w in ws], data.context_size)
                inputs = tensors(x, mask, ws, action[j:j + 256], organic[j:j + 256])
                for n, m in models.items():
                    pred[n].extend(m(*inputs).cpu().tolist())
                pred["v2_seed0_flag_forced_0"].extend(models["v2_seed0"](*inputs[:-1], torch.zeros_like(inputs[-1])).cpu().tolist())
                pred["v2_seed0_flag_forced_1"].extend(models["v2_seed0"](*inputs[:-1], torch.ones_like(inputs[-1])).cpu().tolist())
                pred["v1_history_RM"].extend(v1(inputs[0], inputs[1], inputs[6]).cpu().tolist())
        pred = {n: np.asarray(v) for n, v in pred.items()}
        seeds = np.stack([pred[f"v2_seed{k}"] for k in range(args.ensemble)])
        pred["v2_ensemble_mean"] = seeds.mean(0)
        if args.target != "completion":
            # Train-period item mean of the same target, catalog items only, labels censored at train end.
            sums, counts = np.zeros(len(data.catalog)), np.zeros(len(data.catalog))
            for u in data.users:
                take = (u["tokens"] >= OFFSET) & (u["times"] < train_end)
                np.add.at(sums, u["tokens"][take] - OFFSET, u["train_label"][take])
                np.add.at(counts, u["tokens"][take] - OFFSET, 1)
            global_mean = sums.sum() / counts.sum()
            item_means = np.where(counts > 0, (sums + 5 * global_mean) / (counts + 5), global_mean)
        pred["constant_train_mean"] = np.full(len(label), global_mean)
        pred["item_train_mean"] = item_means[action]
        slices = {"algorithmic": organic == 0, "organic": organic == 1,
                  **{k: kinds == k for k in ("recent128", "older_only", "never_seen")}}
        values = {}
        for n, p in pred.items():
            values[n] = {"overall": regression(p, label, uid), "calibration_gap": calibration(p, label)["weighted_absolute_gap"],
                         "pair_accuracy": pair_accuracy(p, label, uid),
                         "slices": {k: {**regression(p[t], label[t], uid[t]), "pair_accuracy": pair_accuracy(p[t], label[t], uid[t])}
                                    for k, t in slices.items() if t.any()}}
        unique = np.unique(uid)
        rng = np.random.default_rng(42); idx = rng.integers(0, len(unique), size=(2000, len(unique)))
        mae = {n: np.array([np.abs(pred[n][uid == u] - label[uid == u]).mean() for u in unique]) for n in pred}
        comparisons = {}
        for other in ("v1_history_RM", "item_train_mean", "v2_no_features_seed0"):
            d = mae["v2_ensemble_mean"] - mae[other]
            comparisons[f"macroMAE_ensemble_minus_{other}"] = {"mean": float(d.mean()), "ci95": np.quantile(d[idx].mean(1), [.025, .975]).tolist()}
            pa, pb = per_user_pair_accuracy(pred["v2_ensemble_mean"], label, uid), per_user_pair_accuracy(pred[other], label, uid)
            common = sorted(set(pa) & set(pb)); d = np.array([pa[u] - pb[u] for u in common])
            ii = np.random.default_rng(42).integers(0, len(d), size=(2000, len(d)))
            comparisons[f"pairacc_ensemble_minus_{other}"] = {"mean": float(d.mean()), "ci95": np.quantile(d[ii].mean(1), [.025, .975]).tolist(), "users": len(d)}
        report = {"plan": plan, "evaluated_events": len(records), "raw_selection_events": raw_events,
                  "label_mean": float(label.mean()), "label_zero_share": float((label == 0).mean()),
                  "algorithmic_share": float((organic == 0).mean()), "models": values, "comparisons": comparisons,
                  "ensemble_disagreement": {"mean_std": float(seeds.std(0).mean()), "p90_std": float(np.quantile(seeds.std(0), .9))},
                  "organic_condition_effect_seed0": {"mean_pred_flag0": float(pred["v2_seed0_flag_forced_0"].mean()),
                                                     "mean_pred_flag1": float(pred["v2_seed0_flag_forced_1"].mean())},
                  "traces": traces, "elapsed_seconds": time.perf_counter() - started,
                  "notes": "v1_history_RM predicts completion; for other targets it is shown only as a reference.",
                  "confirmation_scored": False, "final_test_scored": False}
        atomic_json(out / "report.json", report)
        atomic_json(out / "status.json", {"status": "completed", "elapsed_seconds": report["elapsed_seconds"]})
    except BaseException:
        atomic_json(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
