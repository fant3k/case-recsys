"""One locked final-test evaluation of the candidates fixed in docs/FINAL-CANDIDATES.md.

No fitting and no selection. Hashes of the candidate document, protocol, code and every checkpoint are written
to locked-manifest.json before any event is loaded. `--window selection` runs the identical code on the already
used selection window as a dry run (it must reproduce known selection values); `--window final` opens the
final test once and refuses to run twice.
"""
import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import lightgbm as lgb
import numpy as np
import torch

from . import rl_v2
from .data import OFFSET, load_sequences, pack_contexts, personal_frequency
from .diagnostics import slice_metrics
from .full_sft import prepare_feedback
from .hybrid import history_distribution, probability_topk
from .metrics import ranking_metrics
from .models import make_policy, state_hash
from .pilot import atomic_json
from .repeat_gate import interpretable
from .reranker import build as reranker_build, load_policy as reranker_policy, rank_groups
from .reward_audit import observed_records
from .reward_v2 import pair_accuracy, side_inputs
from .rl_distill import target_logp, topk_rm_scores, truncated_rewards

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "reference": "artifacts/pps-20261022-01/full+pps+fb-d0.1-wd0.01/last.pt",
    "B_rl_seconds_beta0.3": "artifacts/rl-v2-20261023-01/rl_seconds_beta0.3.pt",
    "GPT2_original_next_item_day": "artifacts/day-objective-20261021-01/next_item_plus_day/best.pt",
    "rm_seed0": "artifacts/reward-v2-20261022-01/v2_seed0.pt",
    "rm_seed1": "artifacts/reward-v2-20261022-01/v2_seed1.pt",
    "rm_judge_seed2": "artifacts/reward-v2-20261022-01/v2_seed2.pt",
    "reranker_booster": "artifacts/reranker-20261022-02/ranker-with_gpt.txt",
    "reranker_gpt2": "artifacts/fullsft-reg-20261022-01/full-d0.1-wd0.01/best.pt",
}
CODE = ["recsys/final_test.py", "recsys/rl_v2.py", "recsys/rl_distill.py", "recsys/reranker.py", "recsys/reward_v2.py",
        "recsys/data.py", "recsys/models.py", "recsys/metrics.py", "recsys/diagnostics.py", "recsys/full_sft.py",
        "recsys/hybrid.py", "recsys/reward_audit.py", "docs/FINAL-CANDIDATES.md", "research/protocol/case2-v1.json"]
BETA_A, TOP_A, BETA_B = 0.03, 50, 0.3
LEVEL = 0.975  # Bonferroni over the two confirmatory hypotheses
DRAWS, SEED = 10000, 20261025
SECONDS = 600  # capped_seconds reward is completion * min(length, 600) / 600; times 600 = equivalent seconds
FINAL_OUT = "artifacts/final-test-20261023-01"
FINAL_MARKER = "artifacts/FINAL-TEST-OPENED.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values, level=LEVEL, draws=DRAWS, seed=SEED):
    """Paired user bootstrap of a mean: point estimate and two-sided percentile interval."""
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(draws, len(v)))
    tail = (1 - level) / 2
    lo, hi = np.quantile(v[idx].mean(1), [tail, 1 - tail])
    return {"mean": float(v.mean()), "level": level, "interval": [float(lo), float(hi)], "users": len(v)}


def paired(rows, ref_rows, margin=None, level=LEVEL):
    if [r["uid"] for r in rows] != [r["uid"] for r in ref_rows]:
        raise ValueError("Paired users do not match")
    scale = 1.0 if margin is None else 1.0 - margin
    return interval([a["ndcg"] - scale * b["ndcg"] for a, b in zip(rows, ref_rows)], level)


def decisions(h1_delta, h2_reward, h2_margin):
    h1 = h1_delta["interval"][0] > 0
    reward, margin = h2_reward["interval"][0] > 0, h2_margin["interval"][0] >= 0
    return {"H1_rm_rerank_top50_ndcg_gain_confirmed": bool(h1),
            "H2_judge_seconds_gain_supported": bool(reward), "H2_relevance_within_5pct_supported": bool(margin),
            "H2_rl_seconds_confirmed": bool(reward and margin),
            "meaning": "Offline hypotheses on one later period of the same cohort; not causal user benefit"}


def claim_final(root, out):
    """Atomic once-only claim of the final window, independent of the output folder; kept after any failure."""
    path = Path(root) / FINAL_MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("The final test was already opened (" + FINAL_MARKER + "); it is not rerun")
    with os.fdopen(fd, "w") as f:
        json.dump({"opened_utc": datetime.now(timezone.utc).isoformat(), "out": str(out)}, f)


def rerank_logp(ref_logp, top_index, top_reward, beta=BETA_A):
    """pi_A ∝ pi_ref * exp(r/beta) with r known on top-K of pi_ref and a pi_ref-weighted neutral tail."""
    return target_logp(ref_logp, truncated_rewards(top_index, top_reward, ref_logp), beta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=["selection", "final"], required=True)
    parser.add_argument("--open-final-test", action="store_true", help="required with --window final")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.window == "final" and not args.open_final_test:
        raise SystemExit("Final test opens once: pass --open-final-test explicitly")
    final = args.window == "final"
    if final and args.out:
        raise SystemExit("The final test writes only to " + FINAL_OUT)
    out = ROOT / (FINAL_OUT if final else (args.out or "artifacts/final-dryrun-selection-20261023-01"))
    if final:
        if any((ROOT / "artifacts").glob("final-test-*")):
            raise SystemExit("A final-test output already exists; the final test is not rerun")
        claim_final(ROOT, out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = sha(proto_path)
    t = proto["time"]
    cutoff = t["query_cutoffs"]["final_test" if final else "selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    plan = {"locked_utc": datetime.now(timezone.utc).isoformat(), "window": args.window, "query_cutoff": cutoff,
            "label_window": [lo, hi], "protocol_sha256": proto_hash, "candidate_document": "docs/FINAL-CANDIDATES.md",
            "checkpoints": CHECKPOINTS, "checkpoint_sha256": {n: sha(ROOT / p) for n, p in CHECKPOINTS.items()},
            "code_sha256": {p: sha(ROOT / p) for p in CODE},
            "hypotheses": {"H1": f"A = pi_ref*exp(r/{BETA_A}) on top-{TOP_A} of pi_ref (proxy RM mean seeds 0,1): "
                                 "lower bound of mean(NDCG_A - NDCG_ref) > 0",
                           "H2": f"B = RL v2 capped seconds beta {BETA_B}: judge expected equivalent-seconds delta lower bound > 0 "
                                 "AND lower bound of mean(NDCG_B - 0.95*NDCG_ref) >= 0"},
            "bootstrap": {"unit": "paired users", "draws": DRAWS, "seed": SEED, "confirmatory_level": LEVEL,
                          "correction": "Bonferroni over H1, H2; descriptive intervals 95%"},
            "final_test_scored": final,
            "units": "equivalent seconds = RM completion * min(train-median length, 600) (protocol capped_seconds times 600)",
            "reranker_global_statistics": "all-user item popularity/response frozen at train end (protocol); personal features at cutoff",
            "scope": "no fitting, no selection; descriptive rows carry no decisions"}
    # Lock hashes and criteria before loading events or computing any score.
    atomic_json(out / "locked-manifest.json", plan)
    status = {"status": "running", "stage": "manifest_locked", "pid": os.getpid(), "final_test_scored": final}

    def progress(**fields):
        status.update(fields, elapsed_seconds=time.perf_counter() - started)
        atomic_json(out / "status.json", status)

    progress()
    keep_awake = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        if os.name == "nt":
            keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001)))
        torch.set_num_threads(4)
        device = rl_v2.DEVICE = "cuda"
        data, features, _, _, support = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
        for u in data.users:
            prepare_feedback(u)
        by_uid = {u["uid"]: u for u in data.users}
        length_norm = torch.as_tensor(features[:, 0], device=device)

        def load_policy(name):
            ck = torch.load(ROOT / CHECKPOINTS[name], map_location="cpu", weights_only=True)
            if ck.get("protocol_sha256", proto_hash) != proto_hash or not np.array_equal(ck["catalog"].numpy(), data.catalog):
                raise ValueError("Catalog/protocol mismatch: " + name)
            model = make_policy(ck["settings"])
            model.load_state_dict(ck["model"])
            return model.to(device).eval().requires_grad_(False)

        policies = {n: load_policy(n) for n in ["reference", "B_rl_seconds_beta0.3", "GPT2_original_next_item_day"]}
        rms = {}
        for seed, name in enumerate(["rm_seed0", "rm_seed1", "rm_judge_seed2"]):
            rm, rck = rl_v2.load_rm(ROOT / CHECKPOINTS[name], features)
            if rck["protocol_sha256"] != proto_hash or not np.array_equal(rck["catalog"].numpy(), data.catalog):
                raise ValueError("RM catalog/protocol mismatch")
            rms[name] = rm
        gpt_reranker, _ = reranker_policy(ROOT / CHECKPOINTS["reranker_gpt2"], proto_hash, data.catalog)
        gpt_reranker.requires_grad_(False)
        booster = lgb.Booster(model_file=str(ROOT / CHECKPOINTS["reranker_booster"]))
        frozen = {**policies, **rms, "reranker_gpt2": gpt_reranker}
        hashes = {n: state_hash(m) for n, m in frozen.items()}

        def window_tensors(windows):
            x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
            rb, gb, cr, co = side_inputs(windows, data.context_size)
            return [torch.as_tensor(v, device=device) for v in (x, mask, rb, gb, cr, co)]

        @torch.no_grad()
        def logp_of(name, tensors):
            model = policies[name]
            if "pps" in model.settings:
                return rl_v2.policy_logp(model, tensors)
            x, mask = tensors[:2]
            out_ = []
            for s in range(0, len(x), 128):
                with rl_v2.autocast():
                    out_.append(torch.log_softmax(model(x[s:s + 128], mask[s:s + 128]).float(), -1))
            return torch.cat(out_)

        @torch.no_grad()
        def rerank(ref_logp, tensors, k=TOP_A):
            idx = ref_logp.topk(k, -1).indices
            scores = []
            for c in range(0, len(idx), 16):
                sl = slice(c, c + 16)
                r = [topk_rm_scores(rms[n], *(v[sl] for v in tensors), idx[sl]) for n in ("rm_seed0", "rm_seed1")]
                scores.append((r[0].float() + r[1].float()) / 2)
            return rerank_logp(ref_logp, idx, torch.cat(scores))

        # ---------- queries at the cutoff: every warm user, history strictly before the cutoff ----------
        progress(stage="queries")
        uids, _, _, raw_histories = data.queries(cutoff)
        windows = []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            windows.append((u, max(0, end - data.context_size), end))
        tensors = window_tensors(windows)
        truth = data.truth(lo, hi)
        full_seen, recent_seen = {}, {}
        for u in data.users:
            end = int(np.searchsorted(u["times"], cutoff))
            full_seen[u["uid"]] = set(map(int, u["items"][:end]))
            recent_seen[u["uid"]] = set(map(int, u["items"][max(0, end - 128):end]))
        row_of = {uid: j for j, uid in enumerate(uids)}
        day_records, raw_day_events = observed_records(data, lo, hi)
        day_records = [r for r in day_records if r[0] in row_of]  # warm at the cutoff
        day_uid = np.array([r[0] for r in day_records])
        day_item = torch.as_tensor(np.array([r[3] for r in day_records]), device=device)
        day_label = np.array([r[4] for r in day_records])
        day_row = torch.as_tensor(np.array([row_of[u] for u in day_uid]), device=device)

        progress(stage="reward caches")
        proxy = ((rl_v2.rm_cache(rms["rm_seed0"], tensors).float() + rl_v2.rm_cache(rms["rm_seed1"], tensors).float()) / 2)
        judge = rl_v2.rm_cache(rms["rm_judge_seed2"], tensors).float()

        report = {"locked_manifest": plan, "data_support": support, "all_warm_users": len(uids),
                  "active_positive_users": len(truth), "observed_day_events_in_catalog": len(day_records),
                  "raw_day_events": raw_day_events, "rows": {}, "final_test_scored": final}
        user_rows, reward_rows = {}, {}

        def rank_row(name, recs, extra=None):
            metrics, rows = ranking_metrics(uids, recs, truth)
            metrics.update(interpretable(uids, recs, truth))
            sliced, _ = slice_metrics(uids, recs, truth, full_seen, recent_seen, data.catalog)
            index = torch.as_tensor([[index_of[int(i)] for i in r[:10]] for r in recs], device=device)
            seconds_norm = float((judge.gather(1, index) * length_norm[index]).mean())
            delivered = {"judge_completion": float(judge.gather(1, index).mean()),
                         "judge_capped_seconds_norm": seconds_norm, "judge_equivalent_seconds": seconds_norm * SECONDS,
                         "mean_capped_length_seconds": float(length_norm[index].mean() * SECONDS)}
            report["rows"][name] = {"ranking": metrics, **sliced, "delivered_top10": delivered, **(extra or {})}
            user_rows[name] = rows
            atomic_json(out / "report.json", report)

        index_of = {int(i): j for j, i in enumerate(data.catalog)}

        def policy_row(name, logp):
            p = logp.exp()
            per_user = {"proxy_completion": (p * proxy).sum(-1), "judge_completion": (p * judge).sum(-1),
                        "proxy_equivalent_seconds": (p * proxy * length_norm).sum(-1) * SECONDS,
                        "judge_equivalent_seconds": (p * judge * length_norm).sum(-1) * SECONDS,
                        "kl_to_reference": (p * (logp - ref_logp)).sum(-1)}
            reward_rows[name] = {k: v.cpu().numpy() for k, v in per_user.items()}
            scores = logp[day_row, day_item].cpu().numpy()
            extra = {"expected_all_warm": {k: float(v.mean()) for k, v in per_user.items()},
                     "policy_pair_accuracy_on_observed_day": pair_accuracy(scores, day_label, day_uid)}
            rank_row(name, data.catalog[logp.topk(10, -1).indices.cpu().numpy()].tolist(), extra)

        progress(stage="policies")
        ref_logp = logp_of("reference", tensors)
        policy_row("reference", ref_logp)
        policy_row("A_rm_rerank_top50", rerank(ref_logp, tensors))
        policy_row("A_full_catalog", rl_v2.analytic_logp(ref_logp, proxy, BETA_A))
        policy_row("B_rl_seconds_beta0.3", logp_of("B_rl_seconds_beta0.3", tensors))
        policy_row("GPT2_original_next_item_day", logp_of("GPT2_original_next_item_day", tensors))

        progress(stage="baselines")
        popularity = np.zeros(len(data.catalog), dtype=np.float64)
        for u in data.users:
            tok = u["tokens"][:int(np.searchsorted(u["times"], data.train_end))]
            popularity += np.bincount(tok[tok >= OFFSET] - OFFSET, minlength=len(popularity))
        popularity = (popularity / popularity.sum()).astype(np.float32)
        memory, recency = [], []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            first = max(0, end - 128)
            p = 0.5 * history_distribution(u["tokens"][first:end], len(data.catalog), popularity) + \
                0.5 * history_distribution(u["tokens"][:end], len(data.catalog), popularity)
            memory.append(probability_topk(p, data.catalog))
            counts, last = Counter(), {}
            for token, tt in zip(u["tokens"][first:end], u["times"][first:end]):
                if token >= OFFSET:
                    a = int(token) - OFFSET
                    counts[a] += 1
                    last[a] = int(tt)
            order = sorted(counts, key=lambda a: (-counts[a], -last[a], int(data.catalog[a])))[:10]
            used = set(order)
            order.extend(a for a in range(10) if a not in used)
            recency.append(data.catalog[order[:10]].tolist())
        rank_row("PersonalFrequency128", personal_frequency(raw_histories, data.catalog, 10))
        rank_row("Frequency128_recency_tiebreak", recency)
        rank_row("Memory_recent_full_0.5", memory)
        progress(stage="reranker")
        groups, _, group_uids = reranker_build(data, gpt_reranker, cutoff, t["gap_seconds"], t["horizon_seconds"], features[:, 0],
                                               stats_cutoff=t["train_end_exclusive"])
        if list(group_uids) != list(uids):
            raise ValueError("Reranker users differ")
        rank_row("Reranker_LightGBM", rank_groups(booster, groups, list(range(booster.num_feature())), data.catalog))

        # ---------- auxiliary NextGroup Recall@10 for policies (not used for any decision) ----------
        progress(stage="next group")
        next_windows, next_ids, next_truth = [], [], {}
        for u in data.users:
            g = int(np.searchsorted(u["times"][u["starts"]], lo))
            if g == len(u["starts"]):
                continue
            first, last = int(u["starts"][g]), int(u["ends"][g])
            if first == 0 or u["times"][first] >= hi:
                continue
            next_ids.append(u["uid"])
            next_windows.append((u, max(0, first - data.context_size), first))
            next_truth[u["uid"]] = set(map(int, u["items"][first:last]))
        nt = window_tensors(next_windows)
        next_ref = logp_of("reference", nt)
        for name, lp in [("reference", next_ref), ("A_rm_rerank_top50", rerank(next_ref, nt)),
                         ("B_rl_seconds_beta0.3", logp_of("B_rl_seconds_beta0.3", nt)),
                         ("GPT2_original_next_item_day", logp_of("GPT2_original_next_item_day", nt))]:
            recs = data.catalog[lp.topk(10, -1).indices.cpu().numpy()].tolist()
            report["rows"][name]["next_group_auxiliary"] = ranking_metrics(next_ids, recs, next_truth)[0]

        # ---------- checks, confirmatory decisions, descriptive comparisons ----------
        self_check = paired(user_rows["reference"], user_rows["reference"])
        if self_check["mean"] != 0 or self_check["interval"] != [0.0, 0.0]:
            raise ValueError("Self-comparison failed")
        if abs(report["rows"]["reference"]["expected_all_warm"]["kl_to_reference"]) > 1e-5:
            raise ValueError("Reference self KL failed")
        h1 = paired(user_rows["A_rm_rerank_top50"], user_rows["reference"])
        h2_reward = interval(reward_rows["B_rl_seconds_beta0.3"]["judge_equivalent_seconds"]
                             - reward_rows["reference"]["judge_equivalent_seconds"])
        h2_margin = paired(user_rows["B_rl_seconds_beta0.3"], user_rows["reference"], margin=0.05)
        report["confirmatory"] = {"H1_ndcg_delta_A_minus_reference": h1, "H2_judge_equivalent_seconds_delta": h2_reward,
                                  "H2_margin_B_minus_0.95_reference": h2_margin,
                                  "H2_ndcg_delta_B_minus_reference": paired(user_rows["B_rl_seconds_beta0.3"], user_rows["reference"])}
        report["decision"] = decisions(h1, h2_reward, h2_margin)
        d95 = lambda a, b: paired(user_rows[a], user_rows[b], level=0.95)
        report["descriptive_95"] = {
            "A_minus_PersonalFrequency128": d95("A_rm_rerank_top50", "PersonalFrequency128"),
            "A_minus_Reranker": d95("A_rm_rerank_top50", "Reranker_LightGBM"),
            "reference_minus_PersonalFrequency128": d95("reference", "PersonalFrequency128"),
            "Reranker_minus_Memory": d95("Reranker_LightGBM", "Memory_recent_full_0.5"),
            "Reranker_minus_PersonalFrequency128": d95("Reranker_LightGBM", "PersonalFrequency128"),
            "reference_minus_GPT2_original": d95("reference", "GPT2_original_next_item_day"),
            "B_proxy_equivalent_seconds_delta": interval(reward_rows["B_rl_seconds_beta0.3"]["proxy_equivalent_seconds"]
                                                         - reward_rows["reference"]["proxy_equivalent_seconds"], 0.95),
            "A_judge_completion_delta": interval(reward_rows["A_rm_rerank_top50"]["judge_completion"]
                                                 - reward_rows["reference"]["judge_completion"], 0.95)}
        for name, model in frozen.items():
            if state_hash(model) != hashes[name] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen component changed: " + name)
        for name, path in CHECKPOINTS.items():
            if sha(ROOT / path) != plan["checkpoint_sha256"][name]:
                raise RuntimeError("Checkpoint changed after lock: " + name)
        report.update({"self_comparison_pass": True, "all_frozen_hashes_pass": True,
                       "elapsed_seconds": time.perf_counter() - started})
        atomic_json(out / "report.json", report)
        atomic_json(out / "ranking-per-user.json", user_rows)
        atomic_json(out / "reward-per-user.json", {n: {k: v.tolist() for k, v in r.items()} for n, r in reward_rows.items()})
        progress(status="completed", stage="done", decision=report["decision"])
    except BaseException:
        progress(status="failed", error=traceback.format_exc())
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000000))


if __name__ == "__main__":
    main()
