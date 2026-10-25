"""RL v2: exact-gradient KL-regularized RL on a cached RM v2 ensemble, Exp-RSFT on logged rewards.

See docs/RL-V2-PLAN.md. Selection window only; confirmation and final test are not scored.
"""
import argparse
import copy
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.utils.checkpoint
from torch.nn import functional as F

from .data import OFFSET, load_sequences, pack_contexts
from .diagnostics import slice_metrics
from .full_sft import group_starts_per_event, prepare_feedback, sample_targets, window_feedback
from .metrics import ranking_metrics
from .models import make_policy, pps_bias, state_hash
from .pilot import atomic_json
from .reward_audit import observed_records
from .reward_v2 import RewardModelV2, pair_accuracy, side_inputs
from .targets import event_lengths, quantile_labels

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda"


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda")


# ---------- objectives ----------

def kl_rl_objective(logits, ref_logp, rewards, beta):
    """Exact per-context J = E_pi[r] - beta*KL(pi||ref) summed over all actions; returns (J, E[r], KL)."""
    logp = F.log_softmax(logits.float(), -1)
    p = logp.exp()
    expected = (p * rewards.float()).sum(-1)
    kl = (p * (logp - ref_logp.float())).sum(-1)
    return expected - beta * kl, expected, kl


def analytic_logp(ref_logp, rewards, beta):
    """Optimum of the per-context objective: pi* ∝ ref * exp(r/beta)."""
    return F.log_softmax(ref_logp.float() + rewards.float() / beta, -1)


def awr_weights(reward, baseline, temperature, w_max=20.0):
    """Exp-RSFT / AWR weights min(exp((r - V)/lambda), w_max), normalised to mean one."""
    w = np.minimum(np.exp((np.asarray(reward) - np.asarray(baseline)) / temperature), w_max)
    return (w / w.mean()).astype(np.float32)


def _weighted_pps_chunk(hidden, weight, actions, rows, positions, weights, epsilon):
    logits = F.linear(hidden, weight).float() + pps_bias(rows, positions, weight.shape[0], epsilon)
    return (F.cross_entropy(logits, actions, reduction="none") * weights).sum()


def weighted_pps_cross_entropy(hidden, weight, actions, rows, positions, weights, epsilon, chunk=1024):
    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, len(actions), chunk):
        sl = slice(s, s + chunk)
        total = total + torch.utils.checkpoint.checkpoint(_weighted_pps_chunk, hidden[sl], weight, actions[sl], rows[sl],
                                                          positions[sl], weights[sl], epsilon, use_reentrant=False)
    return total / weights.sum()


def prefix_baseline(user, start, pos, key):
    """Mean of user[key] over window positions start..start+pos (state only, no target)."""
    cs = user[key + "_cumsum"]
    hi = start + pos
    lo = cs[start - 1] if start > 0 else 0.0
    return (cs[hi] - lo) / (pos + 1)


def margin_interval(rows, ref_rows, margin=0.05, draws=2000, seed=42):
    if [r["uid"] for r in rows] != [r["uid"] for r in ref_rows]:
        raise ValueError("Paired users do not match")
    d = np.array([a["ndcg"] - (1 - margin) * b["ndcg"] for a, b in zip(rows, ref_rows)])
    raw = np.array([a["ndcg"] - b["ndcg"] for a, b in zip(rows, ref_rows)])
    idx = np.random.default_rng(seed).integers(0, len(d), size=(draws, len(d)))
    return {"ndcg_difference": float(raw.mean()), "ndcg_difference_ci95": np.quantile(raw[idx].mean(1), [.025, .975]).tolist(),
            "margin_statistic": float(d.mean()), "margin_ci95": np.quantile(d[idx].mean(1), [.025, .975]).tolist(),
            "within_5pct_supported": bool(np.quantile(d[idx].mean(1), .025) >= 0)}


# ---------- loading ----------

def load_rm(path, features):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    rm = RewardModelV2(make_policy(ck["backbone_settings"]), features, use_features=ck["use_features"])
    rm.load_state_dict(ck["model"])
    return rm.to(DEVICE).eval().requires_grad_(False), ck


@torch.no_grad()
def rm_cache(rm, tensors, chunk=8):
    x, mask, rb, gb, cr, co = tensors
    out = []
    for s in range(0, len(x), 256):
        sl = slice(s, s + 256)
        out.append(rm.all_scores(x[sl], mask[sl], rb[sl], gb[sl], cr[sl], co[sl], organic_flag=0.0, chunk=chunk).half())
    return torch.cat(out)


@torch.no_grad()
def policy_logp(policy, tensors, batch=128):
    x, mask, rb, gb = tensors[:4]
    out = []
    for s in range(0, len(x), batch):
        sl = slice(s, s + batch)
        with autocast():
            logits = policy(x[sl], mask[sl], rb[sl], gb[sl])
        out.append(F.log_softmax(logits.float(), -1))
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", default="artifacts/pps-20261022-01/full+pps+fb-d0.1-wd0.01/last.pt")
    parser.add_argument("--rm-dir", default="artifacts/reward-v2-20261022-01")
    parser.add_argument("--train-contexts", type=int, default=8192)
    parser.add_argument("--rl-steps", type=int, default=2000)
    parser.add_argument("--rl-lr", type=float, default=5e-5)
    parser.add_argument("--rl-batch", type=int, default=64)
    parser.add_argument("--betas", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--rewards", default="completion,seconds,completion_lcb")
    parser.add_argument("--offline-steps", type=int, default=1500)
    parser.add_argument("--offline-lr", type=float, default=5e-5)
    parser.add_argument("--offline-arms", default="sft,exp:completion:0.1,exp:completion:0.3,exp:completion:1.0,exp:seconds:0.3")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device", default="cuda", help="cpu only for the end-to-end smoke check")
    parser.add_argument("--tag", default="", help="prefix for policy names, e.g. the RM target")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": proto_hash, "arguments": vars(args),
            "plan_document": "docs/RL-V2-PLAN.md", "proxy": "mean of RM v2 seeds 0,1", "judge": "RM v2 seed 2, never used in any objective",
            "seeds": {"contexts": 20261201, "rl_batches": 20261202, "offline": 20261203},
            "scope": "selection only; confirmation and final test untouched"}
    atomic_json(out / "plan.json", plan)
    status = {"status": "running", "stage": "loading", "pid": os.getpid()}
    atomic_json(out / "status.json", status)
    keep_awake = False

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)

    try:
        global DEVICE
        DEVICE = args.device
        if DEVICE == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if os.name == "nt":
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001)))
        torch.set_num_threads(4)
        t = proto["time"]
        data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
        length_norm = torch.as_tensor(features[:, 0], device=DEVICE)
        mean_length_norm = float(features[:, 0].mean())
        for u in data.users:
            u["group_start"] = group_starts_per_event(u)
            prepare_feedback(u)
            known = u["tokens"] >= OFFSET
            item_len = np.full(len(u["tokens"]), mean_length_norm, dtype=np.float64)
            item_len[known] = features[u["tokens"][known] - OFFSET, 0]
            u["seconds_reward"] = u["ratios"] * item_len
            u["ratios_cumsum"] = np.cumsum(u["ratios"], dtype=np.float64)
            u["seconds_reward_cumsum"] = np.cumsum(u["seconds_reward"])
            if "quantile" in args.offline_arms:
                u["quantile_label"] = quantile_labels(u, event_lengths(u, features, OFFSET, mean_length_norm)).astype(np.float64)
                u["quantile_label_cumsum"] = np.cumsum(u["quantile_label"])
        by_uid = {u["uid"]: u for u in data.users}

        ck = torch.load(ROOT / args.policy, map_location="cpu", weights_only=True)
        if ck["protocol_sha256"] != proto_hash or not np.array_equal(ck["catalog"].numpy(), data.catalog):
            raise ValueError("Policy catalog/protocol mismatch")
        base = make_policy(ck["settings"])
        base.load_state_dict(ck["model"])
        reference = copy.deepcopy(base).to(DEVICE).eval().requires_grad_(False)
        reference_hash = state_hash(reference)
        rms = {}
        for seed in (0, 1, 2):
            rm, rck = load_rm(ROOT / args.rm_dir / f"v2_seed{seed}.pt", features)
            if rck["protocol_sha256"] != proto_hash or not np.array_equal(rck["catalog"].numpy(), data.catalog):
                raise ValueError("RM catalog/protocol mismatch")
            rms[seed] = rm
        rm_hashes = {s: state_hash(m) for s, m in rms.items()}

        def window_tensors(windows):
            x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
            rb, gb, cr, co = side_inputs(windows, data.context_size)
            return [torch.as_tensor(v, device=DEVICE) for v in (x, mask, rb, gb, cr, co)]

        # Training contexts: train anchors, history strictly before the anchor group.
        progress(stage="caching train rewards")
        _, _, _, _, records = data.sample(np.random.default_rng(plan["seeds"]["contexts"]), args.train_contexts)
        train_windows = []
        for uid, _, event_time in records:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], event_time))
            train_windows.append((u, max(0, end - data.context_size), end))
        train_t = window_tensors(train_windows)
        r0, r1 = rm_cache(rms[0], train_t), rm_cache(rms[1], train_t)
        train_completion = ((r0.float() + r1.float()) / 2).half()
        train_halfdiff = ((r0.float() - r1.float()).abs() / 2).half()
        del r0, r1
        train_ref = policy_logp(reference, train_t)  # fp32: exact KL, no half-precision drift

        # Evaluation contexts: every warm user frozen at the selection cutoff.
        progress(stage="caching evaluation rewards")
        cutoff = t["query_cutoffs"]["selection_validation"]
        lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
        uids, _, _, _ = data.queries(cutoff)
        eval_windows = []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            eval_windows.append((u, max(0, end - data.context_size), end))
        eval_t = window_tensors(eval_windows)
        judge = rm_cache(rms[2], eval_t)
        e0, e1 = rm_cache(rms[0], eval_t), rm_cache(rms[1], eval_t)
        eval_completion = ((e0.float() + e1.float()) / 2).half()
        eval_halfdiff = ((e0.float() - e1.float()).abs() / 2).half()
        eval_ens3 = ((e0.float() + e1.float() + judge.float()) / 3).half()
        del e0, e1
        eval_ref = policy_logp(reference, eval_t)
        truth = data.truth(lo, hi)
        row = {uid: j for j, uid in enumerate(uids)}
        day_records, _ = observed_records(data, lo, hi)
        day_records = [r for r in day_records if r[0] in row]  # warm at the cutoff
        day_uid = np.array([r[0] for r in day_records]); day_item = np.array([r[3] for r in day_records])
        day_label = np.array([r[4] for r in day_records]); day_row = np.array([row[u] for u in day_uid])
        full_seen, recent_seen = {}, {}
        for u in data.users:
            end = int(np.searchsorted(u["times"], cutoff))
            full_seen[u["uid"]] = set(map(int, u["items"][:end]))
            recent_seen[u["uid"]] = set(map(int, u["items"][max(0, end - 128):end]))
        log_pop = torch.as_tensor(features[:, 1], device=DEVICE)

        def reward_of(kind, completion, halfdiff):
            if kind == "completion":
                return completion.float()
            if kind == "seconds":
                return completion.float() * length_norm
            if kind == "completion_lcb":
                return completion.float() - halfdiff.float()
            raise ValueError(kind)

        report = {"plan": plan, "train_contexts": len(train_windows), "eval_users": len(uids),
                  "active_users": len(truth), "observed_day_events": len(day_records),
                  "reference_sha256": reference_hash, "rm_sha256": rm_hashes, "policies": {},
                  "confirmation_scored": False, "final_test_scored": False}
        ref_rows = None

        def evaluate(name, logp, extra=None):
            nonlocal ref_rows
            p = logp.exp()
            def expect(r):
                return float((p * r).sum(-1).mean())
            kl = float((p * (logp - eval_ref)).sum(-1).mean())
            top = logp.topk(10, -1).indices
            recs = data.catalog[top.cpu().numpy()].tolist()
            metrics, rows = ranking_metrics(uids, recs, truth)
            sliced, _ = slice_metrics(uids, recs, truth, full_seen, recent_seen, data.catalog)
            policy_scores = logp[torch.as_tensor(day_row, device=DEVICE), torch.as_tensor(day_item, device=DEVICE)].cpu().numpy()
            entry = {
                "expected_reward": {
                    "proxy_completion": expect(eval_completion.float()), "proxy_seconds_norm": expect(eval_completion.float() * length_norm),
                    "judge_completion": expect(judge.float()), "judge_seconds_norm": expect(judge.float() * length_norm),
                    "ensemble3_completion": expect(eval_ens3.float()), "proxy_lcb": expect(eval_completion.float() - eval_halfdiff.float())},
                "kl_to_reference": kl,
                "top10": {"judge_completion": float(judge.float().gather(1, top).mean()),
                          "mean_length_seconds": float(length_norm[top].mean() * 600),
                          "mean_log_popularity_feature": float(log_pop[top].mean()),
                          "unique_items": int(top.unique().numel()),
                          "top_item_share": float(torch.bincount(top.flatten()).max() / top.numel()),
                          "slot_fractions": sliced["recommendation_fraction_on_active_users"]},
                "ranking": metrics, "slices": {k: v["hits"] for k, v in sliced["slices"].items()},
                "policy_pair_accuracy_on_observed_day": pair_accuracy(policy_scores, day_label, day_uid),
            }
            if ref_rows is None:
                ref_rows = rows
            else:
                entry["vs_reference"] = margin_interval(rows, ref_rows)
            if extra:
                entry.update(extra)
            report["policies"][args.tag + name] = entry
            atomic_json(out / "report.json", report)
            return entry

        evaluate("reference", eval_ref)
        for kind in args.rewards.split(","):
            for beta in map(float, args.betas.split(",")):
                evaluate(f"analytic/{kind}/beta{beta:g}", analytic_logp(eval_ref, reward_of(kind, eval_completion, eval_halfdiff), beta))

        # ---------- exact-gradient KL-RL ----------
        tx, tm, trb, tgb = train_t[:4]
        for kind in args.rewards.split(","):
            for beta in map(float, args.betas.split(",")):
                name = f"rl/{kind}/beta{beta:g}"
                progress(stage=name)
                # Dropout off: the sampled-free objective is then the exact gradient of J for these weights.
                policy = copy.deepcopy(base).to(DEVICE).eval()
                optimizer = torch.optim.AdamW(policy.parameters(), lr=args.rl_lr, weight_decay=0.0)
                rng = np.random.default_rng(plan["seeds"]["rl_batches"])
                curve, t0 = [], time.perf_counter()
                for step in range(1, args.rl_steps + 1):
                    idx = torch.as_tensor(rng.integers(0, len(train_windows), args.rl_batch), device=DEVICE)
                    with autocast():
                        logits = policy(tx[idx], tm[idx], trb[idx], tgb[idx])
                    rewards = reward_of(kind, train_completion[idx], train_halfdiff[idx])
                    objective, expected, kl = kl_rl_objective(logits, train_ref[idx], rewards, beta)
                    loss = -objective.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite RL loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    if step % args.eval_every == 0 or step == 1:
                        lp = policy_logp(policy, eval_t)
                        j_eval, e_eval, kl_eval = kl_rl_objective(lp, eval_ref, reward_of(kind, eval_completion, eval_halfdiff), beta)
                        curve.append({"step": step, "train_batch_J": objective.mean().item(), "train_batch_kl": kl.mean().item(),
                                      "eval_J": float(j_eval.mean()), "eval_expected": float(e_eval.mean()), "eval_kl": float(kl_eval.mean()),
                                      "seconds": time.perf_counter() - t0})
                        progress(stage=name, step=step, last=curve[-1])
                policy.eval()
                lp = policy_logp(policy, eval_t)
                star = analytic_logp(eval_ref, reward_of(kind, eval_completion, eval_halfdiff), beta)
                j_star = float(kl_rl_objective(star, eval_ref, reward_of(kind, eval_completion, eval_halfdiff), beta)[0].mean())
                j_ref = float(kl_rl_objective(eval_ref, eval_ref, reward_of(kind, eval_completion, eval_halfdiff), beta)[0].mean())
                evaluate(name, lp, {"curve": curve, "eval_J_reference": j_ref, "eval_J_analytic_optimum": j_star,
                                    "share_of_analytic_gain": (curve[-1]["eval_J"] - j_ref) / max(1e-12, j_star - j_ref)})
                torch.save({"model": policy.state_dict(), "settings": policy.settings, "catalog": torch.from_numpy(data.catalog),
                            "protocol_sha256": proto_hash, "method": name}, out / (name.replace("/", "_") + ".pt"))
                del policy, optimizer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
        if state_hash(reference) != reference_hash or any(state_hash(m) != rm_hashes[s] for s, m in rms.items()):
            raise RuntimeError("Reference or RM changed during RL")

        # ---------- offline: continued SFT and Exp-RSFT on logged rewards ----------
        for spec in filter(None, args.offline_arms.split(",")):
            parts = spec.split(":")
            name = "offline/sft" if parts[0] == "sft" else f"offline/exp/{parts[1]}/lambda{float(parts[2]):g}"
            progress(stage=name)
            policy = copy.deepcopy(base).to(DEVICE).train()
            optimizer = torch.optim.AdamW(policy.parameters(), lr=args.offline_lr, weight_decay=0.01)
            rng = np.random.default_rng(plan["seeds"]["offline"])
            weight_stats, t0 = [], time.perf_counter()
            for step in range(1, args.offline_steps + 1):
                windows, audit = [], []
                x, mask, owners, positions, actions = sample_targets(data, rng, 64, True, audit=audit, windows=windows)
                if parts[0] == "sft":
                    weights = np.ones(len(actions), dtype=np.float32)
                else:
                    key = {"completion": "ratios", "seconds": "seconds_reward", "quantile": "quantile_label"}[parts[1]]
                    reward = np.array([u[key][e] for u, s, p, e in audit])
                    baseline = np.array([prefix_baseline(u, s, p, key) for u, s, p, e in audit])
                    weights = awr_weights(reward, baseline, float(parts[2]))
                    weight_stats.append([float(weights.max()), float((weights >= weights.max() - 1e-6).mean())])
                rb, gb = window_feedback(windows, data.context_size)
                x, mask, rb, gb = (torch.as_tensor(v, device=DEVICE) for v in (x, mask, rb, gb))
                o, p_ = torch.as_tensor(owners, device=DEVICE), torch.as_tensor(positions, device=DEVICE)
                with autocast():
                    hidden = policy.states(x, mask, rb, gb)[o, p_]
                    loss = weighted_pps_cross_entropy(hidden, policy.transformer.wte.weight[OFFSET:], torch.as_tensor(actions, device=DEVICE),
                                                      x[o], p_, torch.as_tensor(weights, device=DEVICE), policy.settings["pps_epsilon"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite offline loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                if step % 250 == 0:
                    progress(stage=name, step=step)
            policy.eval()
            evaluate(name, policy_logp(policy, eval_t), {"training_seconds": time.perf_counter() - t0,
                                                          "max_weight_mean": float(np.mean([w[0] for w in weight_stats])) if weight_stats else 1.0})
            torch.save({"model": policy.state_dict(), "settings": policy.settings, "catalog": torch.from_numpy(data.catalog),
                        "protocol_sha256": proto_hash, "method": name}, out / (name.replace("/", "_") + ".pt"))
            del policy, optimizer
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
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
