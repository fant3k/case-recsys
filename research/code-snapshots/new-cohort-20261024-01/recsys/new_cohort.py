"""One locked evaluation on a fresh cohort of Yambda users (docs/NEW-COHORT-PLAN.md).

Everything is frozen from the original 1000-user training: catalog, track features and means, policies, RMs, E1
heads with their feature standardisation, reranker and its all-user statistics. `--cohort original --window
selection` is the dry run (must reproduce known selection values; the new cohort is not touched); `--cohort new
--window final --open-new-cohort` scores the new users once and refuses to run twice.
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
import polars as pl
import torch

from . import rl_v2
from .data import OFFSET, Sequences, load_sequences, pack_contexts, personal_frequency
from .exploratory import FEATURES, CandidateHead, head_logp
from .final_test import BETA_A, CHECKPOINTS, rerank_logp
from .full_sft import prepare_feedback
from .hybrid import history_distribution, probability_topk
from .metrics import ranking_metrics
from .models import make_policy, state_hash
from .pilot import atomic_json
from .repeat_gate import interpretable
from .reranker import build as reranker_build, load_policy as reranker_policy, rank_groups
from .reward_v2 import candidate_features, side_inputs
from .rl_distill import topk_rm_scores

ROOT = Path(__file__).resolve().parents[1]
COHORT = "data/processed/yambda-new-cohort-1000.parquet"
HEADS = {"E1_head_all_features": "artifacts/exploratory-e1-20261023-01/head_all_features.pt",
         "E1_head_no_pairwise": "artifacts/exploratory-e1-20261023-01/head_no_pairwise.pt"}
CODE = ["recsys/new_cohort.py", "recsys/final_test.py", "recsys/exploratory.py", "recsys/rl_v2.py", "recsys/rl_distill.py",
        "recsys/reranker.py", "recsys/reward_v2.py", "recsys/data.py", "recsys/models.py", "recsys/metrics.py",
        "docs/NEW-COHORT-PLAN.md", "research/protocol/case2-v1.json", "research/selection/select_new_cohort.py"]
TOP, LEVEL, DRAWS, SEED = 50, 0.975, 10000, 20261026
OUT_NEW, OUT_DRY = "artifacts/new-cohort-20261024-01", "artifacts/new-cohort-dryrun-20261024-01"
MARKER = "artifacts/NEW-COHORT-OPENED.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values, level=LEVEL, draws=DRAWS, seed=SEED):
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(draws, len(v)))
    tail = (1 - level) / 2
    lo, hi = np.quantile(v[idx].mean(1), [tail, 1 - tail])
    return {"mean": float(v.mean()), "level": level, "interval": [float(lo), float(hi)], "users": len(v)}


def paired(rows, ref_rows, level=LEVEL):
    if [r["uid"] for r in rows] != [r["uid"] for r in ref_rows]:
        raise ValueError("Paired users do not match")
    return interval([a["ndcg"] - b["ndcg"] for a, b in zip(rows, ref_rows)], level)


def decisions(n1, n2):
    return {"N1_A_beats_reference_on_new_users": bool(n1["interval"][0] > 0),
            "N2_E1_head_beats_reference_on_new_users": bool(n2["interval"][0] > 0),
            "meaning": "Offline hypotheses on new users of the same released dataset and day; not causal user benefit"}


def check_mode(cohort, window, open_flag, out):
    if (cohort, window) not in {("original", "selection"), ("new", "final")}:
        raise SystemExit("Allowed: --cohort original --window selection (dry run) or --cohort new --window final")
    if cohort == "new" and not open_flag:
        raise SystemExit("The new cohort opens once: pass --open-new-cohort explicitly")
    if cohort == "new" and out:
        raise SystemExit("The new-cohort run writes only to " + OUT_NEW)


def claim(root, out):
    path = Path(root) / MARKER
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("The new cohort was already scored (" + MARKER + "); it is not rerun")
    with os.fdopen(fd, "w") as f:
        json.dump({"opened_utc": datetime.now(timezone.utc).isoformat(), "out": str(out)}, f)


def cohort_sequences(path, catalog, train_end, context_size, original_uids):
    seq = pl.read_parquet(path)
    overlap = set(seq["uid"].to_list()) & set(original_uids)
    if overlap:
        raise ValueError(f"New cohort overlaps the original sample: {len(overlap)} users")
    return Sequences(list(seq.iter_rows(named=True)), catalog, train_end, context_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["original", "new"], required=True)
    parser.add_argument("--window", choices=["selection", "final"], required=True)
    parser.add_argument("--open-new-cohort", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    check_mode(args.cohort, args.window, args.open_new_cohort, args.out)
    new = args.cohort == "new"
    out = ROOT / (OUT_NEW if new else (args.out or OUT_DRY))
    if out.exists():
        raise SystemExit("Output already exists; not rerun: " + str(out))
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = sha(proto_path)
    t = proto["time"]
    cutoff = t["query_cutoffs"]["final_test" if args.window == "final" else "selection_validation"]
    lo, hi = cutoff + t["gap_seconds"], cutoff + t["gap_seconds"] + t["horizon_seconds"]
    checkpoints = {**CHECKPOINTS, **HEADS}
    plan = {"locked_utc": datetime.now(timezone.utc).isoformat(), "cohort": args.cohort, "window": args.window,
            "query_cutoff": cutoff, "label_window": [lo, hi], "protocol_sha256": proto_hash,
            "plan_document": "docs/NEW-COHORT-PLAN.md", "checkpoints": checkpoints,
            "checkpoint_sha256": {n: sha(ROOT / p) for n, p in checkpoints.items()},
            "code_sha256": {p: sha(ROOT / p) for p in CODE},
            "data_sha256": {"original_sample": sha(ROOT / proto["data"]["sample_path"]),
                            **({"new_cohort": sha(ROOT / COHORT)} if new else {})},
            "hypotheses": {"N1": "lower bound of mean(NDCG_A - NDCG_ref) > 0", "N2": "lower bound of mean(NDCG_E1head - NDCG_ref) > 0"},
            "bootstrap": {"draws": DRAWS, "seed": SEED, "confirmatory_level": LEVEL, "descriptive_level": 0.95},
            "new_cohort_scored": new}
    # Every hash is computed above, so a missing file fails here without claiming the one-time cohort.
    if new:
        claim(ROOT, out)
    out.mkdir(parents=True, exist_ok=False)
    # Hashes and criteria are locked before the cohort is loaded or any score is computed.
    atomic_json(out / "locked-manifest.json", plan)
    status = {"status": "running", "stage": "manifest_locked", "new_cohort_scored": new}

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
        original, features, means, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
        data = cohort_sequences(ROOT / COHORT, original.catalog, original.train_end, original.context_size,
                                [u["uid"] for u in original.users]) if new else original
        for u in data.users:
            prepare_feedback(u)
        by_uid = {u["uid"]: u for u in data.users}
        length_norm = torch.as_tensor(features[:, 0], device=device)
        item_mean = torch.as_tensor(means, device=device)

        def load_policy(path):
            ck = torch.load(ROOT / path, map_location="cpu", weights_only=True)
            if ck.get("protocol_sha256", proto_hash) != proto_hash or not np.array_equal(ck["catalog"].numpy(), data.catalog):
                raise ValueError("Catalog/protocol mismatch: " + path)
            m = make_policy(ck["settings"]); m.load_state_dict(ck["model"])
            return m.to(device).eval().requires_grad_(False)

        reference = load_policy(CHECKPOINTS["reference"])
        rl_b = load_policy(CHECKPOINTS["B_rl_seconds_beta0.3"])
        rms = {n: rl_v2.load_rm(ROOT / CHECKPOINTS[n], features)[0] for n in ("rm_seed0", "rm_seed1")}
        heads = {}
        for name, path in HEADS.items():
            hk = torch.load(ROOT / path, map_location="cpu", weights_only=True)
            head = CandidateHead(len(FEATURES), hk["keep"]); head.load_state_dict(hk["head"])
            heads[name] = (head.to(device).eval().requires_grad_(False), hk["feature_mean"].to(device), hk["feature_std"].to(device))
        gpt_reranker, _ = reranker_policy(ROOT / CHECKPOINTS["reranker_gpt2"], proto_hash, data.catalog)
        gpt_reranker.requires_grad_(False)
        booster = lgb.Booster(model_file=str(ROOT / CHECKPOINTS["reranker_booster"]))
        frozen = {"reference": reference, "B": rl_b, **rms, "reranker_gpt2": gpt_reranker, **{n: h[0] for n, h in heads.items()}}
        hashes = {n: state_hash(m) for n, m in frozen.items()}

        progress(stage="queries")
        uids, _, _, raw_histories = data.queries(cutoff)
        windows = []
        for uid in uids:
            u = by_uid[uid]
            end = int(np.searchsorted(u["times"], cutoff))
            windows.append((u, max(0, end - data.context_size), end))
        x, mask = pack_contexts([u["tokens"][a:b] for u, a, b in windows], data.context_size)
        tensors = [torch.as_tensor(v, device=device) for v in (x, mask, *side_inputs(windows, data.context_size))]
        truth = data.truth(lo, hi)

        @torch.no_grad()
        def candidates(ref):
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
            return idx, torch.cat(reward), torch.cat(feats)

        report = {"locked_manifest": plan, "warm_users": len(uids), "active_positive_users": len(truth), "rows": {}, "new_cohort_scored": new}
        user_rows = {}

        def rank_row(name, recs):
            metrics, rows = ranking_metrics(uids, recs, truth)
            metrics.update(interpretable(uids, recs, truth))
            report["rows"][name] = metrics
            user_rows[name] = rows
            atomic_json(out / "report.json", report)

        progress(stage="policies")
        ref = rl_v2.policy_logp(reference, tensors)
        idx, reward, feats = candidates(ref)
        top10 = lambda lp: data.catalog[lp.topk(10, -1).indices.cpu().numpy()].tolist()
        rank_row("reference", top10(ref))
        rank_row("A_rm_rerank_top50", top10(rerank_logp(ref, idx, reward, BETA_A)))
        for name, (head, mu, sd) in heads.items():
            with torch.no_grad():
                rank_row(name, top10(head_logp(ref, idx, head((feats - mu) / sd))))
        rank_row("B_rl_seconds_beta0.3", top10(rl_v2.policy_logp(rl_b, tensors)))

        progress(stage="baselines")
        popularity = np.zeros(len(data.catalog), dtype=np.float64)
        for u in original.users:  # frozen: train popularity of the original sample
            tok = u["tokens"][:int(np.searchsorted(u["times"], original.train_end))]
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
                                               stats_cutoff=t["train_end_exclusive"], stats_data=original)
        if list(group_uids) != list(uids):
            raise ValueError("Reranker users differ")
        rank_row("Reranker_LightGBM", rank_groups(booster, groups, list(range(booster.num_feature())), data.catalog))

        self_check = paired(user_rows["reference"], user_rows["reference"])
        if self_check["mean"] != 0 or self_check["interval"] != [0.0, 0.0]:
            raise ValueError("Self-comparison failed")
        n1 = paired(user_rows["A_rm_rerank_top50"], user_rows["reference"])
        n2 = paired(user_rows["E1_head_all_features"], user_rows["reference"])
        report["confirmatory"] = {"N1_A_minus_reference": n1, "N2_E1_head_minus_reference": n2}
        report["decision"] = decisions(n1, n2)
        d95 = lambda a, b: paired(user_rows[a], user_rows[b], level=0.95)
        report["descriptive_95"] = {
            "E1_head_minus_A": d95("E1_head_all_features", "A_rm_rerank_top50"),
            "E1_no_pairwise_minus_reference": d95("E1_head_no_pairwise", "reference"),
            "B_minus_reference": d95("B_rl_seconds_beta0.3", "reference"),
            "reference_minus_PersonalFrequency128": d95("reference", "PersonalFrequency128"),
            "Reranker_minus_PersonalFrequency128": d95("Reranker_LightGBM", "PersonalFrequency128"),
            "E1_head_minus_PersonalFrequency128": d95("E1_head_all_features", "PersonalFrequency128")}
        for name, model in frozen.items():
            if state_hash(model) != hashes[name] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Frozen component changed: " + name)
        for name, path in checkpoints.items():
            if sha(ROOT / path) != plan["checkpoint_sha256"][name]:
                raise RuntimeError("Checkpoint changed after lock: " + name)
        report.update({"self_comparison_pass": True, "all_frozen_hashes_pass": True, "elapsed_seconds": time.perf_counter() - started})
        atomic_json(out / "report.json", report)
        atomic_json(out / "ranking-per-user.json", user_rows)
        progress(status="completed", stage="done", decision=report["decision"])
    except BaseException:
        progress(status="failed", error=traceback.format_exc())
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000000))


if __name__ == "__main__":
    main()
