"""Descriptive checks requested by the external report review; selection window only, nothing selected.

1. Where does the top-50 reweighting gain come from? Same formula pi_ref*exp(r/beta) on the reference top-50 with
   r = RM v2, the track's train mean completion, the user's own past completion of the track, or a random control.
2. What does RM v2 measure inside the top-50? Within-user AUC for "this candidate is listened >50% tomorrow" versus
   PairAccuracy on listened pairs (which conditions on both tracks being played).
3. Analytic optimum for the already trained quantile RM (no RL training).
4. Five-step generation with repeats masked.
Final test untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts
from .final_test import BETA_A, CHECKPOINTS, SECONDS, TOP_A, rerank_logp
from .full_sft import prepare_feedback
from .metrics import ranking_metrics
from .models import make_policy
from .multistep import append_event, listened_seconds, ndcg_at
from .pilot import atomic_json
from .reward_audit import observed_records
from .reward_v2 import candidate_features, pair_accuracy, side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
QUANTILE_RM = "artifacts/reward-v2-quantile-20261023-01"
SEED = 20261025


def within_user_auc(scores, labels, users):
    """Macro over users with both classes: probability a positive outranks a negative (ties half)."""
    out = []
    for u in np.unique(users):
        m = users == u
        y, s = labels[m], scores[m]
        if y.all() or not y.any():
            continue
        d = s[y][:, None] - s[~y][None, :]
        out.append(float(((d > 0) + 0.5 * (d == 0)).mean()))
    return {"users": len(out), "macro_auc": float(np.mean(out)) if out else None}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", default="artifacts/review-checks-20261023-02", help="output folder relative to the repository root; must not exist")
    out_name = parser.parse_args().out
    out = ROOT / out_name
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto = json.loads((ROOT / "research/protocol/case2-v1.json").read_text())
    t = proto["time"]
    cutoff = t["query_cutoffs"]["selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "window": "selection", "beta": BETA_A, "top_k": TOP_A,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "descriptive; selection already used; nothing selected; final test untouched"}
    device = rl_v2.DEVICE = "cuda"
    data, features, means, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    for u in data.users:
        prepare_feedback(u)
    by_uid = {u["uid"]: u for u in data.users}
    length_norm = torch.as_tensor(features[:, 0], device=device)
    item_mean = torch.as_tensor(means, device=device)
    ck = torch.load(ROOT / CHECKPOINTS["reference"], map_location="cpu", weights_only=True)
    reference = make_policy(ck["settings"]); reference.load_state_dict(ck["model"])
    reference = reference.to(device).eval().requires_grad_(False)
    rm = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1", "rm_judge_seed2")}
    uids, _, _, _ = data.queries(cutoff)
    windows = []
    for uid in uids:
        u = by_uid[uid]
        end = int(np.searchsorted(u["times"], cutoff))
        windows.append((u, max(0, end - data.context_size), end))
    x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
    real = [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]
    truth = data.truth(lo, hi)
    active = [j for j, uid in enumerate(uids) if uid in truth]

    @torch.no_grad()
    def score(model, tensors, idx):
        res = []
        for c in range(0, len(idx), 16):
            sl = slice(c, c + 16)
            res.append(topk_rm_scores(model, *(v[sl] for v in tensors), idx[sl]).float())
        return torch.cat(res)

    ref = rl_v2.policy_logp(reference, real)
    idx = ref.topk(TOP_A, -1).indices
    rm_top = (score(rm["rm_seed0"], real, idx) + score(rm["rm_seed1"], real, idx)) / 2
    n, k = idx.shape
    rep = lambda v: v.repeat_interleave(k, 0)
    feats = candidate_features(rep(real[0]), rep(real[4].float()), rep(real[5]), idx.reshape(-1), rm["rm_seed0"].item_features)
    seen = feats[:, 2].reshape(n, k) > 0
    personal = torch.where(seen, feats[:, 3].reshape(n, k), feats[:, 6].reshape(n, k))
    gen = torch.Generator(device=device).manual_seed(SEED)
    def matched(r):
        """Same per-user mean and spread over the top-50 as RM v2, so beta tilts equally hard."""
        return (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True).clamp(min=1e-6) * rm_top.std(1, keepdim=True) + rm_top.mean(1, keepdim=True)

    noise = matched(torch.rand(rm_top.shape, generator=gen, device=device))

    def ranked(logp):
        recs = data.catalog[logp.topk(10, -1).indices.cpu().numpy()].tolist()
        return ranking_metrics(uids, recs, truth)

    base_metrics, base_rows = ranked(ref)
    check1 = {"reference": base_metrics}
    for name, r in (("rm_v2", rm_top), ("item_train_mean", item_mean[idx]), ("user_past_completion", personal),
                    ("item_train_mean_same_spread", matched(item_mean[idx])),
                    ("user_past_completion_same_spread", matched(personal)), ("random_same_spread", noise)):
        metrics, rows = ranked(rerank_logp(ref, idx, r))
        metrics["vs_reference"] = rl_v2.margin_interval(rows, base_rows)
        check1[name] = metrics
    report["check1_top50_reweighting_by_reward"] = check1
    atomic_json(out / "report.json", report)

    # 2. estimand: occurrence of a >50% listen tomorrow among the top-50 versus ordering of listened pairs
    labels, users, s_rm, s_item, s_personal, s_ref = [], [], [], [], [], []
    for j in active:
        cand = data.catalog[idx[j].cpu().numpy()]
        labels.append(np.array([int(c) in truth[uids[j]] for c in cand])); users.append(np.full(k, j))
        s_rm.append(rm_top[j].cpu().numpy()); s_item.append(item_mean[idx[j]].cpu().numpy())
        s_personal.append(personal[j].cpu().numpy()); s_ref.append(ref[j, idx[j]].cpu().numpy())
    labels, users = np.concatenate(labels), np.concatenate(users)
    check2 = {"top50_candidates_positive_share": float(labels.mean()),
              "auc_positive_tomorrow_within_user": {name: within_user_auc(np.concatenate(v), labels, users) for name, v in
                                                    (("rm_v2", s_rm), ("item_train_mean", s_item),
                                                     ("user_past_completion", s_personal), ("reference_logp", s_ref))}}
    row_of = {uid: j for j, uid in enumerate(uids)}
    records, _ = observed_records(data, lo, hi)
    top_sets = {j: set(idx[j].tolist()) for j in range(n)}
    keep = [r for r in records if r[0] in row_of and r[3] in top_sets[row_of[r[0]]]]
    rows_k = torch.as_tensor([row_of[r[0]] for r in keep], device=device)
    items_k = torch.as_tensor([r[3] for r in keep], device=device)
    pos = (idx[rows_k] == items_k[:, None]).float().argmax(1)
    lab_k, uid_k = np.array([r[4] for r in keep]), np.array([r[0] for r in keep])
    check2["listened_pairs_inside_top50"] = {
        "events": len(keep),
        "rm_v2": pair_accuracy(rm_top[rows_k, pos].cpu().numpy(), lab_k, uid_k),
        "item_train_mean": pair_accuracy(item_mean[items_k].cpu().numpy(), lab_k, uid_k),
        "user_past_completion": pair_accuracy(personal[rows_k, pos].cpu().numpy(), lab_k, uid_k)}
    report["check2_what_rm_measures_in_top50"] = check2
    atomic_json(out / "report.json", report)

    # 3. analytic optimum of the quantile RM (no training)
    q = {}
    for s in (0, 1, 2):
        model, qck = rl_v2.load_rm(ROOT / QUANTILE_RM / f"v2_seed{s}.pt", features)
        if not np.array_equal(qck["catalog"].numpy(), data.catalog):
            raise ValueError("Quantile RM catalog mismatch")
        q[s] = rl_v2.rm_cache(model, real).float()
        del model
    q_proxy, q_judge = (q[0] + q[1]) / 2, q[2]
    completion_judge = rl_v2.rm_cache(rm["rm_judge_seed2"], real).float()
    expect = lambda logp, r: float((logp.exp() * r).sum(-1).mean())
    check3 = {}
    for beta in (0.01, 0.03, 0.1, 0.3, 1.0):
        logp = rl_v2.analytic_logp(ref, q_proxy, beta)
        top = logp.topk(10, -1).indices
        metrics, rows = ranked(logp)
        metrics["vs_reference"] = rl_v2.margin_interval(rows, base_rows)
        check3[f"beta{beta:g}"] = {
            "ranking": metrics,
            "quantile_proxy_gain": expect(logp, q_proxy) / expect(ref, q_proxy) - 1,
            "quantile_judge_gain": expect(logp, q_judge) / expect(ref, q_judge) - 1,
            "completion_judge_gain": expect(logp, completion_judge) / expect(ref, completion_judge) - 1,
            "kl_to_reference": float((logp.exp() * (logp - ref)).sum(-1).mean()),
            "top10_mean_capped_length_seconds": float(length_norm[top].mean() * SECONDS),
            "top10_unique_items": int(top.unique().numel())}
    check3["reference_top10_mean_capped_length_seconds"] = float(length_norm[ref.topk(10, -1).indices].mean() * SECONDS)
    report["check3_quantile_rm_analytic_optimum"] = check3
    atomic_json(out / "report.json", report)
    del q, q_proxy, q_judge, completion_judge

    # 4. five-step generation with repeats masked (greedy, simulated feedback)
    last_token = real[0][torch.arange(n, device=device), real[1].sum(1) - 1]
    gap0 = listened_seconds(real[4][torch.arange(n, device=device), real[1].sum(1) - 1].float(),
                            torch.where(last_token >= OFFSET, length_norm[(last_token - OFFSET).clamp(min=0)] * SECONDS,
                                        torch.full((n,), 200.0, device=device)))
    check4 = {}
    for name in ("reference", "A_rm_rerank_top50"):
        tensors, gap, chosen = real, gap0, []
        for step in range(5):
            logp = rl_v2.policy_logp(reference, tensors)
            if name == "A_rm_rerank_top50":
                i50 = logp.topk(TOP_A, -1).indices
                logp = rerank_logp(logp, i50, (score(rm["rm_seed0"], tensors, i50) + score(rm["rm_seed1"], tensors, i50)) / 2)
            for c in chosen:
                logp[torch.arange(n, device=device), torch.as_tensor(c, device=device)] = -float("inf")
            a = logp.argmax(-1)
            sim = score(rm["rm_judge_seed2"], tensors, a[:, None]).squeeze(1)
            chosen.append(a.cpu().numpy())
            tensors = append_event(tensors, a, sim, gap)
            gap = listened_seconds(sim, length_norm[a] * SECONDS)
        slates = data.catalog[np.stack(chosen, 1)]
        check4[name] = {"masked_slate_ndcg@5_active": float(np.mean([ndcg_at(slates[j].tolist(), truth[uids[j]]) for j in active]))}
    report["check4_masked_five_step"] = check4
    report["elapsed_seconds"] = time.perf_counter() - started
    report["final_test_scored"] = False
    atomic_json(out / "report.json", report)
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    main()
