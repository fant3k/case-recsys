"""RL v3: why does the trained policy recover only part of the analytic optimum on new contexts?

Three explanations, one factor each (docs/RL-V3-PLAN.md):
1. Overfitting to a small context pool: 8192 contexts (~16 passes) vs 65536 (~2 passes).
2. Direction of the divergence: reverse KL(pi||pi*) (= -J/beta + const, our RL objective) vs forward
   KL(pi*||pi) (distillation of the tilted distribution) vs their average (Jeffreys, as in BOND).
3. Shared item embeddings: the same arms with the tied item embedding table frozen.

Rewards are cached sparsely: RM v2 (mean of seeds 0,1) on the top-K items of pi_ref for each context;
every other item gets the context's pi_ref-weighted mean top-K reward (neutral, so the tail keeps its
relative mass). The same truncated reward defines pi* for all arms. Evaluation uses the full-catalog
RM caches and the seed-2 judge exactly as rl_v2. Selection only; confirmation and final test untouched.
"""
import argparse
import contextlib
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts
from .diagnostics import slice_metrics
from .full_sft import prepare_feedback
from .metrics import ranking_metrics
from .models import make_policy, state_hash
from .pilot import atomic_json
from .reward_audit import observed_records
from .reward_v2 import candidate_features, pair_accuracy, side_inputs

ROOT = Path(__file__).resolve().parents[1]


# ---------- truncated reward and divergences ----------

def truncated_rewards(top_index, top_reward, ref_logp):
    """Full reward vectors: top-K values where cached, pi_ref-weighted top-K mean elsewhere."""
    weights = ref_logp.gather(1, top_index).exp()
    neutral = (weights * top_reward).sum(1) / weights.sum(1).clamp(min=1e-12)
    full = neutral[:, None].expand_as(ref_logp).clone()
    return full.scatter_(1, top_index, top_reward)


def target_logp(ref_logp, rewards, beta):
    return F.log_softmax(ref_logp.float() + rewards.float() / beta, -1)


def divergence(kind, logits, target):
    """reverse: KL(pi||pi*), forward: KL(pi*||pi), jeffreys: their mean; per context."""
    logp = F.log_softmax(logits.float(), -1)
    reverse = (logp.exp() * (logp - target)).sum(-1)
    forward = (target.exp() * (target - logp)).sum(-1)
    if kind == "reverse":
        return reverse
    if kind == "forward":
        return forward
    if kind == "jeffreys":
        return 0.5 * (reverse + forward)
    raise ValueError(kind)


def train_autocast(precision):
    """Numerics of the policy forward/backward during distillation; fp16 is the historical setting."""
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=rl_v2.DEVICE == "cuda")
    if precision == "fp32":
        return contextlib.nullcontext()
    return rl_v2.autocast()  # "fp16" and "fp16scaled"


def gradient_stats(grads, truth):
    """Compare per-parameter gradients with the fp32 ones: cosine, relative L2 error, exact-zero shares.

    Returns overall numbers and the same for the tied token embedding ("wte") alone; both dicts are plain floats.
    """
    def one(names):
        a = torch.cat([grads[n].flatten().double() for n in names])
        b = torch.cat([truth[n].flatten().double() for n in names])
        norm_a, norm_b = a.norm(), b.norm()
        return {"cosine_to_fp32": float(a @ b / (norm_a * norm_b).clamp(min=1e-300)),
                "relative_l2_error": float((a - b).norm() / norm_b.clamp(min=1e-300)),
                "norm_ratio": float(norm_a / norm_b.clamp(min=1e-300)),
                "zero_fraction": float((a == 0).double().mean()), "zero_fraction_fp32": float((b == 0).double().mean()),
                "fp32_fraction_below_fp16_min_subnormal": float((b.abs() < 6e-8).double().mean())}
    names = sorted(truth)
    embedding = [n for n in names if n.endswith("wte.weight")]
    return {"all": one(names), "token_embedding": one(embedding) if embedding else None}


@torch.no_grad()
def topk_rm_scores(rm, x, mask, rb, gb, cr, co, top_index):
    """RM response for the given candidate items only; equals all_scores gathered at those items."""
    n, k = top_index.shape
    h = rm.encode(x, mask, rb, gb)
    candidate = rm.backbone.transformer.wte.weight[top_index.reshape(-1) + OFFSET].float()
    rep = lambda t: t.repeat_interleave(k, 0)
    feats = candidate_features(rep(x), rep(cr), rep(co), top_index.reshape(-1), rm.item_features)
    flag = torch.zeros(n * k, device=x.device)
    return rm.score(rep(h), candidate, feats, flag).reshape(n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", default="artifacts/pps-20261022-01/full+pps+fb-d0.1-wd0.01/last.pt")
    parser.add_argument("--rm-dir", default="artifacts/reward-v2-20261022-01")
    parser.add_argument("--pool", type=int, default=65536)
    parser.add_argument("--small-pool", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--settings", default="completion:0.03,seconds:0.3")
    parser.add_argument("--arms", default="reverse:small,reverse:pool,forward:pool,jeffreys:pool,forward:pool:frozen")
    parser.add_argument("--analytic-k", default="20,50,200,1024")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp16", "fp16scaled", "bf16", "fp32"], default="fp16",
                        help="numerics of the trained policy's forward/backward; fp16 reproduces the historical runs, "
                             "fp16scaled adds dynamic loss scaling (docs/PRECISION-PLAN.md)")
    parser.add_argument("--cache-fp32", action="store_true", help="keep the cached top-K rewards in float32 instead of float16")
    parser.add_argument("--grad-diagnostic", type=int, default=0,
                        help="number of training batches whose gradients are compared across precisions (no weight update)")
    args = parser.parse_args()
    rl_v2.DEVICE = device = args.device
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": proto_hash, "arguments": vars(args),
            "plan_document": "docs/RL-V3-PLAN.md", "proxy": "mean RM v2 seeds 0,1 on top-K of pi_ref, neutral tail",
            "judge": "RM v2 seed 2, full catalog, evaluation only", "seeds": {"pool": 20261301, "batches": 20261302},
            "scope": "selection only; confirmation and final test untouched; fixed last step reported, curves diagnostic only"}
    atomic_json(out / "plan.json", plan)
    status = {"status": "running", "stage": "loading", "pid": os.getpid()}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)

    progress()
    keep_awake = False
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if os.name == "nt":
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001)))
        torch.set_num_threads(4)
        t = proto["time"]
        data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
        for u in data.users:
            prepare_feedback(u)
        by_uid = {u["uid"]: u for u in data.users}
        length_norm = torch.as_tensor(features[:, 0], device=device)
        log_pop = torch.as_tensor(features[:, 1], device=device)
        ck = torch.load(ROOT / args.policy, map_location="cpu", weights_only=True)
        if ck["protocol_sha256"] != proto_hash or not np.array_equal(ck["catalog"].numpy(), data.catalog):
            raise ValueError("Policy catalog/protocol mismatch")
        base = make_policy(ck["settings"]); base.load_state_dict(ck["model"])
        reference = copy.deepcopy(base).to(device).eval().requires_grad_(False)
        reference_hash = state_hash(reference)
        rms = {s: rl_v2.load_rm(ROOT / args.rm_dir / f"v2_seed{s}.pt", features)[0] for s in (0, 1, 2)}
        rm_hashes = {s: state_hash(m) for s, m in rms.items()}

        def window_tensors(windows):
            x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
            rb, gb, cr, co = side_inputs(windows, data.context_size)
            return [torch.as_tensor(v, device=device) for v in (x, mask, rb, gb, cr, co)]

        # ---------- training pool: sparse top-K reward cache ----------
        progress(stage="caching pool rewards")
        _, _, _, _, records = data.sample(np.random.default_rng(plan["seeds"]["pool"]), args.pool)
        windows = []
        for uid, _, event_time in records:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], event_time))
            windows.append((u, max(0, end - data.context_size), end))
        pool_t = [torch.empty(0)] * 6
        top_index, top_comp = [], []
        for s in range(0, len(windows), 256):
            tt = window_tensors(windows[s:s + 256])
            ref = rl_v2.policy_logp(reference, tt)
            idx = ref.topk(args.top_k, -1).indices
            scores = []
            for c in range(0, len(idx), 16):
                sl = slice(c, c + 16)
                r = [topk_rm_scores(rms[seed], *(v[sl] for v in tt), idx[sl]) for seed in (0, 1)]
                scores.append((r[0] + r[1]) / 2)
            top_index.append(idx); top_comp.append(torch.cat(scores).to(torch.float32 if args.cache_fp32 else torch.float16))
            pool_t = [torch.cat([a, b]) if a.numel() else b for a, b in zip(pool_t, tt)]
            if s % 8192 == 0:
                progress(stage="caching pool rewards", contexts=s)
        top_index, top_comp = torch.cat(top_index), torch.cat(top_comp)
        px, pm, prb, pgb = pool_t[:4]

        # ---------- evaluation caches (full catalog, as rl_v2) ----------
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
        judge = rl_v2.rm_cache(rms[2], eval_t)
        eval_comp = ((rl_v2.rm_cache(rms[0], eval_t).float() + rl_v2.rm_cache(rms[1], eval_t).float()) / 2)
        eval_ref = rl_v2.policy_logp(reference, eval_t)
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
        report = {"plan": plan, "pool_contexts": len(windows), "eval_users": len(uids), "policies": {},
                  "reference_sha256": reference_hash, "rm_sha256": rm_hashes,
                  "confirmation_scored": False, "final_test_scored": False}
        ref_rows = {}

        def kind_reward(kind, completion):
            return completion * length_norm if kind == "seconds" else completion

        def evaluate(name, logp, extra=None):
            p = logp.exp()
            expect = lambda r: float((p * r).sum(-1).mean())
            top = logp.topk(10, -1).indices
            recs = data.catalog[top.cpu().numpy()].tolist()
            metrics, rows = ranking_metrics(uids, recs, truth)
            sliced, _ = slice_metrics(uids, recs, truth, full_seen, recent_seen, data.catalog)
            scores = logp[torch.as_tensor(day_row, device=device), torch.as_tensor(day_item, device=device)].cpu().numpy()
            entry = {"expected_reward": {"proxy_completion": expect(eval_comp), "proxy_seconds_norm": expect(eval_comp * length_norm),
                                         "judge_completion": expect(judge.float()), "judge_seconds_norm": expect(judge.float() * length_norm)},
                     "kl_to_reference": float((p * (logp - eval_ref)).sum(-1).mean()),
                     "top10": {"mean_length_seconds": float(length_norm[top].mean() * 600), "mean_log_popularity_feature": float(log_pop[top].mean()),
                               "unique_items": int(top.unique().numel()), "slot_fractions": sliced["recommendation_fraction_on_active_users"]},
                     "ranking": metrics, "policy_pair_accuracy_on_observed_day": pair_accuracy(scores, day_label, day_uid)}
            if "reference" in ref_rows:
                entry["vs_reference"] = rl_v2.margin_interval(rows, ref_rows["reference"])
            else:
                ref_rows["reference"] = rows
            if extra:
                entry.update(extra)
            report["policies"][name] = entry
            atomic_json(out / "report.json", report)
            return entry

        evaluate("reference", eval_ref)
        eval_top = eval_ref.topk(max(map(int, args.analytic_k.split(","))), -1).indices
        for setting in args.settings.split(","):
            kind, beta = setting.split(":"); beta = float(beta)
            full_r = kind_reward(kind, eval_comp)
            evaluate(f"analytic/{kind}/beta{beta:g}/full", target_logp(eval_ref, full_r, beta))
            for k in map(int, args.analytic_k.split(",")):
                idx = eval_top[:, :k]
                trunc = truncated_rewards(idx, full_r.gather(1, idx), eval_ref)
                evaluate(f"analytic/{kind}/beta{beta:g}/top{k}", target_logp(eval_ref, trunc, beta))

        def batch_target(idx, kind, beta):
            with torch.no_grad():
                with rl_v2.autocast():
                    ref_logp = F.log_softmax(reference(px[idx], pm[idx], prb[idx], pgb[idx]).float(), -1)
                top_r = top_comp[idx].float()
                if kind == "seconds":
                    top_r = top_r * length_norm[top_index[idx]]
                rewards = truncated_rewards(top_index[idx], top_r, ref_logp)
                return target_logp(ref_logp, rewards, beta)

        def batch_gradients(policy, idx, target, loss_kind, precision, scale=1.0):
            policy.zero_grad(set_to_none=True)
            with train_autocast(precision):
                logits = policy(px[idx], pm[idx], prb[idx], pgb[idx])
            (divergence(loss_kind, logits, target).mean() * scale).backward()
            grads = {n: q.grad.detach().float().div(scale).clone() for n, q in policy.named_parameters() if q.grad is not None}
            policy.zero_grad(set_to_none=True)
            return grads

        def gradient_diagnostic(policy, kind, beta, loss_kind):
            """Gradients of the distillation loss on the first training batches under four numerics; weights untouched."""
            modes = {"fp16": ("fp16", 1.0), "fp16_loss_scale_1024": ("fp16", 1024.0), "bf16": ("bf16", 1.0)}
            per_batch = []
            size_all = len(windows)
            for b in range(args.grad_diagnostic):
                idx = torch.arange(b * args.batch, (b + 1) * args.batch, device=device) % size_all
                target = batch_target(idx, kind, beta)
                truth_grads = batch_gradients(policy, idx, target, loss_kind, "fp32")
                per_batch.append({mode: gradient_stats(batch_gradients(policy, idx, target, loss_kind, prec, scale), truth_grads)
                                  for mode, (prec, scale) in modes.items()})
            mean = lambda key, part, mode: float(np.mean([r[mode][part][key] for r in per_batch if r[mode][part]]))
            return {"batches": args.grad_diagnostic, "loss": loss_kind,
                    "mean_over_batches": {mode: {part: {key: mean(key, part, mode) for key in per_batch[0][mode]["all"]}
                                                 for part in ("all", "token_embedding")} for mode in modes}}

        report["gradient_diagnostic"] = {}

        # ---------- trained arms ----------
        for setting in args.settings.split(","):
            kind, beta = setting.split(":"); beta = float(beta)
            if args.grad_diagnostic:
                progress(stage=f"gradient diagnostic at the reference/{kind}")
                probe = copy.deepcopy(base).to(device).eval()
                report["gradient_diagnostic"][f"init/{kind}/beta{beta:g}"] = gradient_diagnostic(probe, kind, beta, "reverse")
                del probe
                atomic_json(out / "report.json", report)
            for spec in filter(None, args.arms.split(",")):
                loss_kind, pool_kind, *flags = spec.split(":")
                frozen = "frozen" in flags
                name = f"{loss_kind}/{pool_kind}{'/frozen' if frozen else ''}/{kind}/beta{beta:g}"
                progress(stage=name)
                size = args.small_pool if pool_kind == "small" else len(windows)
                policy = copy.deepcopy(base).to(device).eval()
                policy.transformer.wte.weight.requires_grad_(not frozen)
                optimizer = torch.optim.AdamW([q for q in policy.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
                rng = np.random.default_rng(plan["seeds"]["batches"])
                order = np.concatenate([rng.permutation(size) for _ in range(args.steps * args.batch // size + 1)])
                scaler = torch.amp.GradScaler("cuda", init_scale=1024.0) if args.precision == "fp16scaled" else None
                curve, t0 = [], time.perf_counter()
                for step in range(1, args.steps + 1):
                    idx = torch.as_tensor(order[(step - 1) * args.batch: step * args.batch], device=device)
                    target = batch_target(idx, kind, beta)
                    with train_autocast(args.precision):
                        logits = policy(px[idx], pm[idx], prb[idx], pgb[idx])
                    loss = divergence(loss_kind, logits, target).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite loss")
                    optimizer.zero_grad(set_to_none=True)
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_([q for q in policy.parameters() if q.requires_grad], 1.0)
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if step % args.eval_every == 0 or step == 1:
                        lp = rl_v2.policy_logp(policy, eval_t)
                        j, e, kl = rl_v2.kl_rl_objective(lp, eval_ref, kind_reward(kind, eval_comp), beta)
                        top = lp.topk(10, -1).indices
                        ndcg = ranking_metrics(uids, data.catalog[top.cpu().numpy()].tolist(), truth)[0]["NDCG@10"]
                        curve.append({"step": step, "train_loss": loss.item(), "eval_J": float(j.mean()), "eval_expected": float(e.mean()),
                                      "eval_kl": float(kl.mean()), "eval_NDCG@10": ndcg, "seconds": time.perf_counter() - t0})
                        progress(stage=name, step=step, last=curve[-1])
                lp = rl_v2.policy_logp(policy, eval_t)
                full_r = kind_reward(kind, eval_comp)
                j_star = float(rl_v2.kl_rl_objective(target_logp(eval_ref, full_r, beta), eval_ref, full_r, beta)[0].mean())
                j_ref = float(rl_v2.kl_rl_objective(eval_ref, eval_ref, full_r, beta)[0].mean())
                # The arms are trained toward the top-K truncated optimum; report the share against it too.
                idx_k = eval_top[:, :args.top_k]
                trunc_r = truncated_rewards(idx_k, full_r.gather(1, idx_k), eval_ref)
                trunc_star = target_logp(eval_ref, trunc_r, beta)
                j_trunc_ref = float(rl_v2.kl_rl_objective(eval_ref, eval_ref, trunc_r, beta)[0].mean())
                j_trunc_star = float(rl_v2.kl_rl_objective(trunc_star, eval_ref, trunc_r, beta)[0].mean())
                j_trunc_pol = float(rl_v2.kl_rl_objective(lp, eval_ref, trunc_r, beta)[0].mean())
                if args.grad_diagnostic:
                    report["gradient_diagnostic"][f"final/{name}"] = gradient_diagnostic(policy, kind, beta, loss_kind)
                evaluate(name, lp, {"precision": args.precision, "reward_cache_fp32": args.cache_fp32,
                                    "loss_scale_at_end": scaler.get_scale() if scaler else None,
                                    "curve": curve, "pool": size, "passes": args.steps * args.batch / size, "frozen_item_embeddings": frozen,
                                    "share_of_analytic_gain": (curve[-1]["eval_J"] - j_ref) / max(1e-12, j_star - j_ref),
                                    "share_of_truncated_optimum_gain": (j_trunc_pol - j_trunc_ref) / max(1e-12, j_trunc_star - j_trunc_ref)})
                torch.save({"model": policy.state_dict(), "settings": policy.settings, "catalog": torch.from_numpy(data.catalog),
                            "protocol_sha256": proto_hash, "method": name}, out / (name.replace("/", "_") + ".pt"))
                del policy, optimizer
                if device == "cuda":
                    torch.cuda.empty_cache()
        if state_hash(reference) != reference_hash or any(state_hash(m) != rm_hashes[s] for s, m in rms.items()):
            raise RuntimeError("Reference or RM changed")
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
