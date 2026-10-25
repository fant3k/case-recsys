"""Two-stage reranker: candidates from GPT-2 and history, LightGBM lambdarank on top.

Training rows come from cutoffs inside train whose next-day labels lie inside train.
Their GPT-2 features use an EARLY GPT-2 trained only on targets before the first
training cutoff, so the ranker never learns from in-sample GPT-2 scores. All
history features use events strictly before each cutoff. Hyperparameters are fixed
here; the last training cutoff is the early-stopping validation. Selection is only
scored, never used for fitting or choosing settings.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import lightgbm as lgb
import numpy as np
import torch

from .data import OFFSET, load_sequences, personal_frequency
from .diagnostics import paired_bootstrap, slice_metrics
from .hybrid import history_distribution, probability_topk
from .metrics import ranking_metrics
from .models import Policy
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DAY = 86400
FEATURES = ["gpt_logp", "gpt_log_rank", "count_recent128", "log_count_full", "count_1d", "count_7d", "count_30d",
            "log_seconds_since_last", "never_seen", "recent_position_from_end", "user_item_mean_ratio",
            "user_item_positive_count", "log_item_popularity", "log_item_popularity_7d", "item_mean_ratio",
            "item_length_norm", "log_user_history", "user_unique_fraction_recent", "user_mean_ratio",
            "user_events_1d", "share_of_recent128"]
GPT_FEATURES = {"gpt_logp", "gpt_log_rank"}
PARAMS = dict(objective="lambdarank", n_estimators=1000, learning_rate=0.05, num_leaves=63, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=20261103,
              verbose=-1)


@torch.no_grad()
def gpt_log_probs(model, x, mask, batch=128):
    out = []
    for offset in range(0, len(x), batch):
        s = slice(offset, offset + batch)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(torch.as_tensor(x[s], device="cuda"), torch.as_tensor(mask[s], device="cuda")).float()
        out.append(torch.log_softmax(logits, -1).cpu().numpy())
    return np.concatenate(out)


def global_item_stats(data, cutoff, size):
    """Popularity and mean response from all sampled users' events strictly before cutoff."""
    count, count7, ratio = (np.zeros(size) for _ in range(3))
    for user in data.users:
        end = int(np.searchsorted(user["times"], cutoff))
        tok, tm, rt = user["tokens"][:end], user["times"][:end], user["ratios"][:end]
        known = tok >= OFFSET
        idx = tok[known] - OFFSET
        count += np.bincount(idx, minlength=size)
        ratio += np.bincount(idx, weights=rt[known], minlength=size)
        count7 += np.bincount(idx[tm[known] >= cutoff - 7 * DAY], minlength=size)
    mean = ratio.sum() / max(1.0, count.sum())
    smoothed = (ratio + 5 * mean) / (count + 5)
    return count, count7, smoothed


def user_candidates(user, end, logp, cutoff, stats, length, top_gpt=50, top_full=50, context=128):
    """Candidate catalog indices and their feature rows for one user at one cutoff."""
    size = len(logp)
    count_all, count7_all, item_ratio = stats
    tok, tm, rt = user["tokens"][:end], user["times"][:end], user["ratios"][:end]
    known = tok >= OFFSET
    idx, tt, rr = tok[known] - OFFSET, tm[known], rt[known]
    recent_tok = tok[max(0, end - context):]
    recent_idx = recent_tok[recent_tok >= OFFSET] - OFFSET
    full_count = np.bincount(idx, minlength=size)
    candidates = np.unique(np.concatenate([
        np.argpartition(-logp, top_gpt)[:top_gpt],
        recent_idx,
        np.argsort(-full_count, kind="stable")[:min(top_full, int((full_count > 0).sum()))],
    ]))
    def counts(mask):
        return np.bincount(idx[mask], minlength=size)[candidates]
    recent_count = np.bincount(recent_idx, minlength=size)[candidates]
    reversed_unique, first = np.unique(idx[::-1], return_index=True)
    last_time = np.full(size, -1.0)
    last_time[reversed_unique] = tt[::-1][first]
    last = last_time[candidates]
    never = last < 0
    seconds = np.where(never, 1e9, cutoff - last)
    position = np.full(size, context + 1.0)
    rev_recent, first_recent = np.unique(recent_idx[::-1], return_index=True)
    position[rev_recent] = first_recent
    ui_sum = np.bincount(idx, weights=rr, minlength=size)[candidates]
    ui_count = full_count[candidates]
    ui_mean = np.where(ui_count > 0, ui_sum / np.maximum(ui_count, 1), np.nan)
    ui_positive = np.bincount(idx[rr > 0.5], minlength=size)[candidates]
    rank = np.empty(size)
    rank[np.argsort(-logp, kind="stable")] = np.arange(size)
    n = len(candidates)
    rows = np.column_stack([
        logp[candidates], np.log1p(rank[candidates]), recent_count, np.log1p(ui_count),
        counts(tt >= cutoff - DAY), counts(tt >= cutoff - 7 * DAY), counts(tt >= cutoff - 30 * DAY),
        np.log1p(seconds), never.astype(float), position[candidates], ui_mean, ui_positive,
        np.log1p(count_all[candidates]), np.log1p(count7_all[candidates]), item_ratio[candidates],
        length[candidates],
        np.full(n, np.log1p(len(tok))), np.full(n, len(np.unique(recent_idx)) / max(1, len(recent_idx))),
        np.full(n, rr.mean() if len(rr) else np.nan), np.full(n, float((tm >= cutoff - DAY).sum())),
        recent_count / max(1, len(recent_idx)),
    ]).astype(np.float32)
    return candidates, rows


def build(data, model, cutoff, gap, horizon, length, with_truth=True, stats_cutoff=None, stats_data=None):
    """stats_cutoff freezes the all-user item statistics (protocol: train end); personal features use cutoff.

    stats_data: users the all-user statistics come from (default: the evaluated users themselves).
    """
    uids, x, mask, _ = data.queries(cutoff)
    logp = gpt_log_probs(model, x, mask)
    stats = global_item_stats(data if stats_data is None else stats_data, cutoff if stats_cutoff is None else stats_cutoff, len(data.catalog))
    truth = data.truth(cutoff + gap, cutoff + gap + horizon)
    by_uid = {u["uid"]: u for u in data.users}
    out = []
    for uid, lp in zip(uids, logp):
        user = by_uid[uid]
        end = int(np.searchsorted(user["times"], cutoff))
        cand, rows = user_candidates(user, end, lp, cutoff, stats, length)
        target = truth.get(uid, set())
        labels = np.array([int(data.catalog[c]) in target for c in cand], dtype=np.int32)
        out.append({"uid": uid, "candidates": cand, "rows": rows, "labels": labels, "gpt_logp": lp})
    return out, truth, uids


def stack(groups, columns, positive_only=True):
    keep = [g for g in groups if not positive_only or g["labels"].any()]
    x = np.concatenate([g["rows"][:, columns] for g in keep])
    y = np.concatenate([g["labels"] for g in keep])
    return x, y, [len(g["labels"]) for g in keep]


def rank_groups(model, groups, columns, catalog, k=10):
    recs = []
    for g in groups:
        score = model.predict(g["rows"][:, columns])
        raw = catalog[g["candidates"]]
        order = np.lexsort((raw, -score))
        ranking = raw[order][:k].tolist()
        if len(ranking) < k:
            used = set(ranking)
            ranking += [int(i) for i in probability_topk(np.exp(g["gpt_logp"]), catalog, k + len(used)) if int(i) not in used][:k - len(ranking)]
        recs.append(ranking)
    return recs


def load_policy(path, proto_hash, catalog):
    ck = torch.load(path, map_location="cuda", weights_only=True)
    if ck["protocol_sha256"] != proto_hash or not np.array_equal(ck["catalog"].cpu().numpy(), catalog):
        raise ValueError("Checkpoint catalog/protocol mismatch: " + str(path))
    model = Policy(**ck["settings"]).cuda().eval()
    model.load_state_dict(ck["model"])
    return model, ck.get("target_end")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--early-gpt", default="artifacts/early-gpt-20261022-01/full-d0.1-wd0.01/last.pt")
    parser.add_argument("--main-gpt", default="artifacts/fullsft-reg-20261022-01/full-d0.1-wd0.01/best.pt")
    parser.add_argument("--train-start", type=int, default=25032400)
    parser.add_argument("--train-cutoffs", type=int, default=7)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    t = proto["time"]
    gap, horizon, train_end = t["gap_seconds"], t["horizon_seconds"], t["train_end_exclusive"]
    cutoffs = [args.train_start + k * DAY for k in range(args.train_cutoffs)]
    if cutoffs[-1] + gap + horizon > train_end:
        raise ValueError("Training label windows must lie inside train")
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": proto_hash,
            "early_gpt": args.early_gpt, "main_gpt": args.main_gpt, "training_cutoffs": cutoffs,
            "fit_cutoffs": cutoffs[:-1], "early_stopping_cutoff": cutoffs[-1], "lightgbm": PARAMS,
            "features": FEATURES, "candidates": "GPT-2 top50 + all recent128 items + top50 full-history counts",
            "primary": "reranker with GPT-2 features; selection features from main GPT-2, model fitted on early GPT-2 features",
            "secondary": ["same reranker with early GPT-2 features at selection", "reranker without GPT-2 features, same candidates"],
            "scope": "selection scored once per variant; no tuning on selection; confirmation and final test untouched"}
    atomic_json(out / "plan.json", plan)
    atomic_json(out / "status.json", {"status": "running", "stage": "loading"})
    data, features, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], train_end)
    length = features[:, 0]
    early, early_end = load_policy(ROOT / args.early_gpt, proto_hash, data.catalog)
    if early_end is None or early_end > cutoffs[0]:
        raise ValueError("Early GPT-2 must be trained only before the first training cutoff")
    main_gpt, _ = load_policy(ROOT / args.main_gpt, proto_hash, data.catalog)
    report = {"plan": plan, "confirmation_scored": False, "final_test_scored": False, "training": {}}

    train_groups = []
    for c in cutoffs:
        groups, truth, _ = build(data, early, c, gap, horizon, length)
        for g in groups:
            g["cutoff"] = c
        train_groups.append(groups)
        pairs = sum(len(v) for v in truth.values())
        covered = sum(int(g["labels"].sum()) for g in groups)
        report["training"][str(c)] = {"users": len(groups), "rows": int(sum(len(g["labels"]) for g in groups)),
                                      "truth_pairs": pairs, "candidate_recall_pooled": covered / max(1, pairs)}
        atomic_json(out / "status.json", {"status": "running", "stage": f"features {c}"})
    fit = [g for groups in train_groups[:-1] for g in groups]
    valid = train_groups[-1]

    selection_cutoff = t["query_cutoffs"]["selection_validation"]
    sel_main, truth, uids = build(data, main_gpt, selection_cutoff, gap, horizon, length)
    sel_early, _, _ = build(data, early, selection_cutoff, gap, horizon, length)
    order = [g["uid"] for g in sel_main]
    pairs = sum(len(v) for v in truth.values())
    report["selection_candidate_recall_pooled"] = sum(int(g["labels"].sum()) for g in sel_main) / pairs
    report["selection_candidate_recall_macro"] = float(np.mean(
        [g["labels"].sum() / len(truth[g["uid"]]) for g in sel_main if g["uid"] in truth]))

    atomic_json(out / "status.json", {"status": "running", "stage": "fitting rankers"})
    variants, rows_by = {}, {}
    full_cols = list(range(len(FEATURES)))
    no_gpt_cols = [j for j, f in enumerate(FEATURES) if f not in GPT_FEATURES]
    for name, columns in (("with_gpt", full_cols), ("without_gpt", no_gpt_cols)):
        x, y, g = stack(fit, columns)
        vx, vy, vg = stack(valid, columns)
        ranker = lgb.LGBMRanker(**PARAMS)
        ranker.fit(x, y, group=g, eval_set=[(vx, vy)], eval_group=[vg], eval_at=[10],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
        ranker.booster_.save_model(str(out / f"ranker-{name}.txt"))
        importance = ranker.booster_.feature_importance("gain")
        variants[name] = {"best_iteration": int(ranker.best_iteration_), "fit_rows": int(len(y)),
                          "fit_positive_rows": int(y.sum()), "fit_groups": len(g),
                          "early_stopping_ndcg@10": float(ranker.best_score_["valid_0"]["ndcg@10"]),
                          "feature_gain_share": {FEATURES[c]: float(v / importance.sum()) for c, v in zip(columns, importance)}}
        evaluations = [("main_gpt_features", sel_main)] if name == "without_gpt" else \
            [("main_gpt_features", sel_main), ("early_gpt_features", sel_early)]
        for label, groups in evaluations:
            recs = rank_groups(ranker, groups, columns, data.catalog)
            metrics, rows = ranking_metrics(order, recs, truth)
            variants[name][label] = metrics
            rows_by[f"{name}/{label}"] = (rows, recs)
    report["rerankers"] = variants

    # Baselines on the same users and truth.
    by_uid = {u["uid"]: u for u in data.users}
    histories = [by_uid[u]["items"][max(0, int(np.searchsorted(by_uid[u]["times"], selection_cutoff)) - 128):
                                    int(np.searchsorted(by_uid[u]["times"], selection_cutoff))] for u in order]
    stats_count, _, _ = global_item_stats(data, selection_cutoff, len(data.catalog))
    popularity = stats_count / stats_count.sum()
    memory = []
    for u in order:
        user = by_uid[u]
        end = int(np.searchsorted(user["times"], selection_cutoff))
        p = 0.5 * history_distribution(user["tokens"][max(0, end - 128):end], len(data.catalog), popularity) + \
            0.5 * history_distribution(user["tokens"][:end], len(data.catalog), popularity)
        memory.append(probability_topk(p, data.catalog))
    baselines = {"PersonalFrequency128": personal_frequency(histories, data.catalog, 10),
                 "Memory_recent_full_0.5": memory,
                 "GPT2_main": [probability_topk(np.exp(g["gpt_logp"]), data.catalog) for g in sel_main]}
    report["baselines"] = {}
    for name, recs in baselines.items():
        metrics, rows = ranking_metrics(order, recs, truth)
        report["baselines"][name] = metrics
        rows_by[name] = (rows, recs)
    primary = rows_by["with_gpt/main_gpt_features"][0]
    report["comparisons"] = {
        "primary_minus_frequency128": paired_bootstrap(primary, rows_by["PersonalFrequency128"][0]),
        "primary_minus_memory": paired_bootstrap(primary, rows_by["Memory_recent_full_0.5"][0]),
        "primary_minus_without_gpt": paired_bootstrap(primary, rows_by["without_gpt/main_gpt_features"][0]),
        "without_gpt_minus_memory": paired_bootstrap(rows_by["without_gpt/main_gpt_features"][0], rows_by["Memory_recent_full_0.5"][0]),
        "primary_minus_gpt2": paired_bootstrap(primary, rows_by["GPT2_main"][0])}
    full_seen, recent_seen = {}, {}
    for u in data.users:
        end = int(np.searchsorted(u["times"], selection_cutoff))
        full_seen[u["uid"]] = set(map(int, u["items"][:end]))
        recent_seen[u["uid"]] = set(map(int, u["items"][max(0, end - 128):end]))
    report["slices"] = {name: slice_metrics(order, rows_by[name][1], truth, full_seen, recent_seen, data.catalog)[0]
                        for name in ("with_gpt/main_gpt_features", "without_gpt/main_gpt_features", "PersonalFrequency128", "Memory_recent_full_0.5")}
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(out / "report.json", report)
    atomic_json(out / "status.json", {"status": "completed"})


def run():
    try:
        main()
    except BaseException:
        out = None
        for i, arg in enumerate(__import__("sys").argv):
            if arg == "--out":
                out = ROOT / __import__("sys").argv[i + 1]
        if out is not None and out.exists():
            atomic_json(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    run()
