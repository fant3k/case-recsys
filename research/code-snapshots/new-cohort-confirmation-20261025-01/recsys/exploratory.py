"""Exploratory experiments after the final test (docs/EXPLORATORY-PLAN.md); selection window only.

E1: a small head on top of the frozen reference adds a correction to the logits of the reference top-50, from
candidate features; trained with the RL v3 objective (reverse KL to pi* = pi_ref*exp(r/beta), RM v2 on top-50).
Variant `all` sees pairwise user-track features like the RM; `no_pairwise` sees only track/user features.
E2: REINFORCE on five-track sequences without repeats; every chosen track is appended with the training RM's
prediction as its feedback (a simulator), reward = sum of predictions, exact per-step KL to the reference.
Evaluated by the training simulator, by the judge as simulator, and by the judge on the real history.
Final test untouched.
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

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts
from .final_test import CHECKPOINTS
from .full_sft import prepare_feedback
from .metrics import ranking_metrics
from .models import make_policy
from .multistep import append_event, listened_seconds, ndcg_at
from .pilot import atomic_json
from .reward_v2 import candidate_features, side_inputs
from .rl_distill import divergence, target_logp, topk_rm_scores, train_autocast, truncated_rewards

ROOT = Path(__file__).resolve().parents[1]
TOP, BETA = 50, 0.03
FEATURES = ["log_count", "recency", "seen", "user_track_completion", "length", "popularity", "user_mean_completion",
            "organic_share", "context_length", "track_train_mean", "ref_logp", "rank"]
PAIRWISE = [0, 1, 2, 3]


class CandidateHead(nn.Module):
    """Logit correction for each candidate; zero output at start, so the policy starts as the reference."""

    def __init__(self, n_features, keep, hidden=64):
        super().__init__()
        self.register_buffer("keep", torch.as_tensor(keep, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):
        return self.net(feats * self.keep).squeeze(-1)


def head_logp(ref_logp, top_index, correction):
    return F.log_softmax(ref_logp.scatter_add(1, top_index, correction), -1)


def objective(logp, ref_logp, rewards, beta):
    p = logp.exp()
    return (p * rewards).sum(-1) - beta * (p * (logp - ref_logp)).sum(-1)


def reward_to_go(rewards):
    """rewards: steps x batch -> sum of this and all later rewards."""
    return rewards.flip(0).cumsum(0).flip(0)


def leave_one_out_advantage(returns):
    """returns: steps x batch. Baseline for each trajectory = mean of the other trajectories, so it does not depend on
    the trajectory's own action (a batch mean including it would shrink the score-function gradient by (N-1)/N)."""
    n = returns.shape[1]
    if n < 2:
        raise ValueError("leave-one-out baseline needs at least two trajectories")
    return returns - (returns.sum(1, keepdim=True) - returns) / (n - 1)


def regularised_return(rewards, penalties):
    """Return credited to action t: rewards of steps >= t minus penalties of steps > t only.

    The KL of step t depends on the state s_t, not on the action a_t, so it is not credited to a_t (its exact
    gradient is taken separately); it would only add variance.
    """
    return reward_to_go(rewards) - (reward_to_go(penalties) - penalties)


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--experiment", choices=["e1", "e2"], default="e1")
    if selector.parse_known_args()[0].experiment == "e2":
        return main_e2()  # E2 has its own options (--beta, --horizon); the E1 parser would reject them
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["e1", "e2"], default="e1")
    parser.add_argument("--out", default="artifacts/exploratory-e1-20261023-01")
    parser.add_argument("--pool", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--arms", default="all,no_pairwise",
                        help="all / no_pairwise: RL objective toward the top-50 optimum; sft_next / sft_positive: the same head "
                             "with all features trained by cross-entropy on the logged next track (all, or played >50%%), no RM; "
                             "joint: the head AND all GPT-2 weights trained together with the RL objective (docs/JOINT-PLAN.md)")
    parser.add_argument("--joint-lr", type=float, default=5e-5,
                        help="learning rate of the GPT-2 weights in the `joint` arm (head: --lr)")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16",
                        help="numerics of the GPT-2 forward/backward in the `joint` arm")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="shifts the pool, batch-order and head-init seeds by 1000*k; 0 reproduces the original run")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto = json.loads((ROOT / "research/protocol/case2-v1.json").read_text())
    t = proto["time"]
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan_document": "docs/EXPLORATORY-PLAN.md", "experiment": "E1",
            "arguments": vars(args), "top_k": TOP, "beta": BETA, "features": FEATURES, "pairwise_features": [FEATURES[i] for i in PAIRWISE],
            "seeds": {"pool": 20261301 + 1000 * args.seed_offset, "batches": 20261302 + 1000 * args.seed_offset,
                      "init": 20261303 + 1000 * args.seed_offset},
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "exploratory, selection window already used; final test untouched; fixed last step reported"}
    atomic_json(out / "plan.json", plan)
    status = {"status": "running", "stage": "loading"}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)

    progress()
    try:
        device = rl_v2.DEVICE = "cuda"
        data, features, means, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
        for u in data.users:
            prepare_feedback(u)
        by_uid = {u["uid"]: u for u in data.users}
        item_mean = torch.as_tensor(means, device=device)
        ck = torch.load(ROOT / CHECKPOINTS["reference"], map_location="cpu", weights_only=True)
        reference = make_policy(ck["settings"]); reference.load_state_dict(ck["model"])
        reference = reference.to(device).eval().requires_grad_(False)
        rms = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1", "rm_judge_seed2")}

        def window_tensors(windows):
            x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
            return [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]

        @torch.no_grad()
        def candidates(tensors):
            """Reference log-probs, top-50, RM proxy reward and the 12 raw features for every context."""
            ref = rl_v2.policy_logp(reference, tensors)
            idx = ref.topk(TOP, -1).indices
            reward, feats = [], []
            for c in range(0, len(idx), 16):
                sl = slice(c, c + 16)
                part = [v[sl] for v in tensors]
                reward.append((topk_rm_scores(rms["rm_seed0"], *part, idx[sl]).float() + topk_rm_scores(rms["rm_seed1"], *part, idx[sl]).float()) / 2)
                n = len(idx[sl])
                rep = lambda v: v.repeat_interleave(TOP, 0)
                f = candidate_features(rep(part[0]), rep(part[4].float()), rep(part[5]), idx[sl].reshape(-1), rms["rm_seed0"].item_features)
                extra = torch.stack([item_mean[idx[sl]].reshape(-1), ref[sl].gather(1, idx[sl]).reshape(-1),
                                     torch.arange(TOP, device=device).repeat(n).float() / TOP], 1)
                feats.append(torch.cat([f, extra], 1).reshape(n, TOP, -1))
            return ref, idx, torch.cat(reward), torch.cat(feats)

        progress(stage="caching pool")
        _, _, pool_actions, pool_ratios, records = data.sample(np.random.default_rng(plan["seeds"]["pool"]), args.pool)
        pool_act = torch.as_tensor(pool_actions, device=device)
        pool_positive = torch.as_tensor(pool_ratios > 0.5, device=device)
        windows = []
        for uid, _, event_time in records:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], event_time))
            windows.append((u, max(0, end - data.context_size), end))
        pool_t, pool_idx, pool_r, pool_f = [], [], [], []
        for s in range(0, len(windows), 256):
            tt = window_tensors(windows[s:s + 256])
            _, idx, r, f = candidates(tt)
            pool_t.append(tt[:4]); pool_idx.append(idx); pool_r.append(r); pool_f.append(f)
        px, pm, prb, pgb = [torch.cat([p[i] for p in pool_t]) for i in range(4)]
        pool_idx, pool_r, pool_f = torch.cat(pool_idx), torch.cat(pool_r), torch.cat(pool_f)
        mu, sd = pool_f.reshape(-1, len(FEATURES)).mean(0), pool_f.reshape(-1, len(FEATURES)).std(0).clamp(min=1e-6)
        pool_f = (pool_f - mu) / sd

        progress(stage="caching evaluation")
        cutoff = t["query_cutoffs"]["selection_validation"]
        lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
        uids, _, _, _ = data.queries(cutoff)
        eval_windows = []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            eval_windows.append((u, max(0, end - data.context_size), end))
        eval_t = window_tensors(eval_windows)
        eval_ref, eval_idx, eval_r, eval_f = candidates(eval_t)
        eval_f = (eval_f - mu) / sd
        eval_trunc = truncated_rewards(eval_idx, eval_r, eval_ref)
        eval_target = target_logp(eval_ref, eval_trunc, BETA)
        proxy_full = (rl_v2.rm_cache(rms["rm_seed0"], eval_t).float() + rl_v2.rm_cache(rms["rm_seed1"], eval_t).float()) / 2
        judge_full = rl_v2.rm_cache(rms["rm_judge_seed2"], eval_t).float()
        truth = data.truth(lo, hi)
        j_ref = float(objective(eval_ref, eval_ref, eval_trunc, BETA).mean())
        j_star = float(objective(eval_target, eval_ref, eval_trunc, BETA).mean())
        report = {"plan": plan, "pool_contexts": len(windows), "eval_users": len(uids), "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
                  "J_reference": j_ref, "J_optimum_top50": j_star, "arms": {}, "final_test_scored": False}
        rows_of = {}

        def evaluate(logp):
            p = logp.exp()
            recs = data.catalog[logp.topk(10, -1).indices.cpu().numpy()].tolist()
            metrics, rows = ranking_metrics(uids, recs, truth)
            j = float(objective(logp, eval_ref, eval_trunc, BETA).mean())
            return {"ranking": metrics, "J": j, "share_of_optimum_gain": (j - j_ref) / max(1e-12, j_star - j_ref),
                    "expected_proxy_completion": float((p * proxy_full).sum(-1).mean()),
                    "expected_judge_completion": float((p * judge_full).sum(-1).mean()),
                    "kl_to_reference": float((p * (logp - eval_ref)).sum(-1).mean())}, rows

        for name, logp in (("reference", eval_ref), ("A_optimum_top50", eval_target)):
            entry, rows = evaluate(logp)
            rows_of[name] = rows
            report["arms"][name] = entry
        report["arms"]["A_optimum_top50"]["vs_reference"] = rl_v2.margin_interval(rows_of["A_optimum_top50"], rows_of["reference"])
        atomic_json(out / "report.json", report)

        keep_all, keep_no_pairwise = [1.0] * len(FEATURES), [0.0 if i in PAIRWISE else 1.0 for i in range(len(FEATURES))]
        arm_specs = {"all": ("head_all_features", keep_all, "rl"), "no_pairwise": ("head_no_pairwise", keep_no_pairwise, "rl"),
                     "sft_next": ("head_all_features_sft_next", keep_all, "sft"),
                     "sft_positive": ("head_all_features_sft_positive", keep_all, "sft_positive"),
                     "joint": ("joint_gpt2_plus_head", keep_all, "joint")}
        for arm, keep, objective_kind in (arm_specs[a] for a in args.arms.split(",")):
            progress(stage=arm)
            torch.manual_seed(plan["seeds"]["init"])
            head = CandidateHead(len(FEATURES), keep).to(device)
            joint = objective_kind == "joint"
            if joint:
                policy = copy.deepcopy(reference).eval().requires_grad_(True)
                optimizer = torch.optim.AdamW([{"params": list(head.parameters()), "lr": args.lr, "weight_decay": 0.0},
                                               {"params": list(policy.parameters()), "lr": args.joint_lr, "weight_decay": 0.0}])
            else:
                optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

            def arm_eval_logp():
                return head_logp(rl_v2.policy_logp(policy, eval_t) if joint else eval_ref, eval_idx, head(eval_f))

            rng = np.random.default_rng(plan["seeds"]["batches"])
            order = np.concatenate([rng.permutation(len(windows)) for _ in range(args.steps * args.batch // len(windows) + 1)])
            curve = []
            for step in range(1, args.steps + 1):
                b = torch.as_tensor(order[(step - 1) * args.batch: step * args.batch], device=device)
                with torch.no_grad():
                    with rl_v2.autocast():
                        ref_logp = F.log_softmax(reference(px[b], pm[b], prb[b], pgb[b]).float(), -1)
                    target = target_logp(ref_logp, truncated_rewards(pool_idx[b], pool_r[b], ref_logp), BETA)
                if joint:
                    with train_autocast(args.precision):
                        logits = policy(px[b], pm[b], prb[b], pgb[b])
                    logp = head_logp(F.log_softmax(logits.float(), -1), pool_idx[b], head(pool_f[b]))
                else:
                    logp = head_logp(ref_logp, pool_idx[b], head(pool_f[b]))
                if objective_kind in ("rl", "joint"):
                    loss = divergence("reverse", logp, target).mean()
                else:  # supervised control: logged next track, no reward model
                    nll = -logp.gather(1, pool_act[b][:, None]).squeeze(1)
                    keep_rows = pool_positive[b] if objective_kind == "sft_positive" else torch.ones_like(nll, dtype=torch.bool)
                    loss = nll[keep_rows].mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if joint:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                if step % args.eval_every == 0 or step == 1:
                    with torch.no_grad():
                        entry, _ = evaluate(arm_eval_logp())
                    curve.append({"step": step, "train_loss": loss.item(), "eval_NDCG@10": entry["ranking"]["NDCG@10"],
                                  "eval_share": entry["share_of_optimum_gain"], "eval_kl": entry["kl_to_reference"]})
                    progress(stage=arm, step=step, last=curve[-1])
            with torch.no_grad():
                entry, rows = evaluate(arm_eval_logp())
            entry["vs_reference"] = rl_v2.margin_interval(rows, rows_of["reference"])
            entry["vs_A_optimum"] = rl_v2.margin_interval(rows, rows_of["A_optimum_top50"])
            entry["curve"] = curve
            entry["objective"] = objective_kind
            report["arms"][arm] = entry
            torch.save({"head": head.state_dict(), "keep": keep, "feature_mean": mu.cpu(), "feature_std": sd.cpu()}, out / f"{arm}.pt")
            if joint:
                torch.save({"model": policy.state_dict(), "settings": policy.settings, "catalog": torch.from_numpy(data.catalog),
                            "method": arm}, out / f"{arm}-policy.pt")
                del policy
            atomic_json(out / "report.json", report)
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(out / "report.json", report)
        progress(status="completed", stage="done")
    except BaseException:
        progress(status="failed", error=traceback.format_exc())
        raise


def main_e2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="e2")
    parser.add_argument("--out", default="artifacts/exploratory-e2-20261023-01")
    parser.add_argument("--pool", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto = json.loads((ROOT / "research/protocol/case2-v1.json").read_text())
    t = proto["time"]
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "plan_document": "docs/EXPLORATORY-PLAN.md", "experiment": "E2",
            "arguments": vars(args), "seeds": {"pool": 20261301, "batches": 20261304, "sampling": 20261305},
            "simulator": "training RM (mean seeds 0,1) prediction appended as the generated track's listened ratio",
            "reward": "sum over steps of training RM predictions; REINFORCE with reward-to-go and a leave-one-out batch baseline per step",
            "kl": "exact KL to the reference on every step, repeats masked in both; -beta*KL of later steps enters the return of each action, the current-step KL gets its exact gradient",
            "gap": "listened part of the previous track: simulated ratio x capped length",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "exploratory, selection window already used; final test untouched; fixed last step reported"}
    atomic_json(out / "plan.json", plan)
    status = {"status": "running", "stage": "loading"}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)

    progress()
    try:
        device = rl_v2.DEVICE = "cuda"
        data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
        for u in data.users:
            prepare_feedback(u)
        by_uid = {u["uid"]: u for u in data.users}
        length_norm = torch.as_tensor(features[:, 0], device=device)
        ck = torch.load(ROOT / CHECKPOINTS["reference"], map_location="cpu", weights_only=True)
        base = make_policy(ck["settings"]); base.load_state_dict(ck["model"])
        reference = base.to(device).eval().requires_grad_(False)
        policy = make_policy(ck["settings"]); policy.load_state_dict(ck["model"]); policy = policy.to(device).eval()
        rms = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1", "rm_judge_seed2")}

        def window_tensors(windows):
            x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
            return [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]

        def first_gap(tensors):
            rows = torch.arange(len(tensors[0]), device=device)
            last = tensors[0][rows, tensors[1].sum(1) - 1]
            ratio = tensors[4][rows, tensors[1].sum(1) - 1].float()
            return listened_seconds(ratio, torch.where(last >= OFFSET, length_norm[(last - OFFSET).clamp(min=0)] * 600,
                                                       torch.full_like(last, 200).float()))

        @torch.no_grad()
        def score(names, tensors, a):
            vals = [topk_rm_scores(rms[n], *tensors, a[:, None]).float().squeeze(1) for n in names]
            return sum(vals) / len(vals)

        def masked_logp(model, tensors, chosen, grad):
            with torch.set_grad_enabled(grad):
                with rl_v2.autocast():
                    logits = model(*tensors[:4]).float()
                for c in chosen:  # finite mask: keeps the exact KL and its gradient free of inf - inf
                    logits = logits.scatter(1, c[:, None], -1e9)
                return F.log_softmax(logits, -1)

        progress(stage="caching pool")
        _, _, _, _, records = data.sample(np.random.default_rng(plan["seeds"]["pool"]), args.pool)
        windows = []
        for uid, _, event_time in records:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], event_time))
            windows.append((u, max(0, end - data.context_size), end))
        pool = [torch.cat(parts) for parts in zip(*[window_tensors(windows[s:s + 1024]) for s in range(0, len(windows), 1024)])]

        cutoff = t["query_cutoffs"]["selection_validation"]
        lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
        uids, _, _, _ = data.queries(cutoff)
        eval_windows = []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            eval_windows.append((u, max(0, end - data.context_size), end))
        eval_t = window_tensors(eval_windows)
        truth = data.truth(lo, hi)
        active = [j for j, uid in enumerate(uids) if uid in truth]
        recent = [set((u["tokens"][max(0, b - 128):b][u["tokens"][max(0, b - 128):b] >= OFFSET] - OFFSET).tolist()) for u, a, b in eval_windows]

        @torch.no_grad()
        def evaluate(model):
            res = {}
            ref1 = masked_logp(reference, eval_t, [], False)
            lp1 = masked_logp(model, eval_t, [], False)
            res["step1_ndcg@10"] = ranking_metrics(uids, data.catalog[lp1.topk(10, -1).indices.cpu().numpy()].tolist(), truth)[0]["NDCG@10"]
            res["step1_kl_to_reference"] = float((lp1.exp() * (lp1 - ref1)).sum(-1).mean())
            for sim_name, sim in (("training_simulator", ("rm_seed0", "rm_seed1")), ("judge_simulator", ("rm_judge_seed2",))):
                tensors, gap, chosen, ret = eval_t, first_gap(eval_t), [], torch.zeros(len(uids), device=device)
                for step in range(args.horizon):
                    a = masked_logp(model, tensors, chosen, False).argmax(-1)
                    r = score(sim, tensors, a)
                    ret += r
                    chosen.append(a)
                    tensors = append_event(tensors, a, r, gap)
                    gap = listened_seconds(r, length_norm[a] * 600)
                items = torch.stack(chosen, 1)
                entry = {"return_by_own_simulator": float(ret.mean())}
                if sim_name == "training_simulator":
                    real_judge = torch.stack([score(("rm_judge_seed2",), eval_t, items[:, s]) for s in range(args.horizon)], 1)
                    slates = data.catalog[items.cpu().numpy()]
                    entry.update({"judge_on_real_history_return": float(real_judge.sum(1).mean()),
                                  "slate_ndcg@5_active": float(np.mean([ndcg_at(slates[j].tolist(), truth[uids[j]]) for j in active])),
                                  "mean_capped_length_seconds": float(length_norm[items].mean() * 600),
                                  "familiar_share": float(np.mean([[int(i) in recent[j] for i in items[j].tolist()] for j in range(len(uids))])),
                                  "unique_items": int(items.unique().numel())})
                res[sim_name] = entry
            return res

        report = {"plan": plan, "pool_contexts": len(windows), "eval_users": len(uids), "active_users": len(active),
                  "reference": evaluate(reference), "curve": [], "final_test_scored": False}
        atomic_json(out / "report.json", report)
        progress(stage="training")
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
        rng = np.random.default_rng(plan["seeds"]["batches"])
        gen = torch.Generator(device=device).manual_seed(plan["seeds"]["sampling"])
        order = np.concatenate([rng.permutation(len(windows)) for _ in range(args.steps * args.batch // len(windows) + 1)])
        for step in range(1, args.steps + 1):
            b = torch.as_tensor(order[(step - 1) * args.batch: step * args.batch], device=device)
            tensors = [v[b] for v in pool]
            gap, chosen, logps, kls, rewards = first_gap(tensors), [], [], [], []
            for _ in range(args.horizon):
                lp = masked_logp(policy, tensors, chosen, True)
                ref_lp = masked_logp(reference, tensors, chosen, False)
                p = lp.exp()
                kls.append((p * (lp - ref_lp)).sum(-1))
                a = torch.multinomial(p.detach(), 1, generator=gen).squeeze(1)
                logps.append(lp.gather(1, a[:, None]).squeeze(1))
                r = score(("rm_seed0", "rm_seed1"), tensors, a)
                rewards.append(r)
                chosen.append(a)
                tensors = append_event(tensors, a, r, gap)
                gap = listened_seconds(r, length_norm[a] * 600)
            # Score-function part sees the full regularised return r - beta*KL of this and later steps, so an action is
            # charged for the KL it causes downstream; the KL of the current step also gets its exact gradient below.
            rtg = regularised_return(torch.stack(rewards), args.beta * torch.stack(kls).detach())
            advantage = leave_one_out_advantage(rtg)
            loss = -(advantage * torch.stack(logps)).sum(0).mean() + args.beta * torch.stack(kls).sum(0).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            if step % args.eval_every == 0:
                entry = {"step": step, "train_return": float(torch.stack(rewards).sum(0).mean()),
                         "train_kl": float(torch.stack(kls).sum(0).mean()), "eval": evaluate(policy)}
                report["curve"].append(entry)
                atomic_json(out / "report.json", report)
                progress(stage="training", step=step, last=entry)
        report["trained"] = evaluate(policy)
        report["elapsed_seconds"] = time.perf_counter() - started
        torch.save({"model": policy.state_dict(), "settings": policy.settings, "catalog": torch.from_numpy(data.catalog)}, out / "policy.pt")
        atomic_json(out / "report.json", report)
        progress(status="completed", stage="done")
    except BaseException:
        progress(status="failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
