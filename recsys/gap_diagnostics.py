"""Why does the full-position GPT-2 lose to PersonalFrequency128? No training.

1. In-sample: the same day metric on cutoffs whose label days lie inside train.
2. Ordering only: GPT-2 vs frequency ranking the SAME recent-128 items, same fill.
3. Repeat mass: GPT-2 probability on recent items vs their observed share tomorrow.
4. Where the gap sits: per-user GPT-2 minus frequency by user terciles.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import OFFSET, load_sequences, personal_frequency
from .diagnostics import paired_bootstrap
from .metrics import ranking_metrics
from .models import Policy
from .pilot import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def history_indices(tokens):
    tokens = np.asarray(tokens)
    return tokens[tokens >= OFFSET] - OFFSET


def rank_within_history(scores, history, fill_scores, k=10):
    """Top history items by `scores`, then non-history items by `fill_scores`.

    Ties break by catalog index so both orderings are deterministic.
    """
    unique = np.unique(history)
    ordered = unique[np.lexsort((unique, -scores[unique]))][:k].tolist()
    if len(ordered) < k:
        masked = fill_scores.astype(np.float64).copy()
        masked[unique] = -np.inf
        rest = np.lexsort((np.arange(len(masked)), -masked))
        ordered.extend(rest[:k - len(ordered)].tolist())
    return ordered


@torch.no_grad()
def gpt_probabilities(model, x, mask, batch=128):
    out = []
    for offset in range(0, len(x), batch):
        s = slice(offset, offset + batch)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(torch.as_tensor(x[s], device="cuda"), torch.as_tensor(mask[s], device="cuda")).float()
        out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out)


def top_ids(probabilities, catalog, k=10):
    return [catalog[np.lexsort((np.arange(len(p)), -p))[:k]].tolist() for p in probabilities]


def window(data, model, cutoff, gap, horizon):
    uids, x, mask, histories = data.queries(cutoff)
    truth = data.truth(cutoff + gap, cutoff + gap + horizon)
    probs = gpt_probabilities(model, x, mask)
    gpt = ranking_metrics(uids, top_ids(probs, data.catalog), truth)
    freq = ranking_metrics(uids, personal_frequency(histories, data.catalog, 10), truth)
    return uids, x, probs, truth, histories, gpt, freq


def summary(gpt, freq):
    return {"users": gpt[0]["users"], "GPT2_NDCG@10": gpt[0]["NDCG@10"], "Frequency128_NDCG@10": freq[0]["NDCG@10"],
            "GPT2_Recall@10": gpt[0]["Recall@10"], "Frequency128_Recall@10": freq[0]["Recall@10"],
            "gpt_minus_frequency": paired_bootstrap(gpt[1], freq[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="artifacts/fullsft-reg-20261022-01/full-d0.1-wd0.01/best.pt")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    proto_path = ROOT / "research/protocol/case2-v1.json"
    proto = json.loads(proto_path.read_text())
    proto_hash = hashlib.sha256(proto_path.read_bytes()).hexdigest()
    ck_path = ROOT / args.checkpoint
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": args.checkpoint,
              "checkpoint_sha256": hashlib.sha256(ck_path.read_bytes()).hexdigest(),
              "protocol_sha256": proto_hash, "training": False,
              "confirmation_scored": False, "final_test_scored": False}
    atomic_json(out / "status.json", {"status": "running"})
    data, _, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], proto["time"]["train_end_exclusive"])
    ck = torch.load(ck_path, map_location="cuda", weights_only=True)
    if ck["protocol_sha256"] != proto_hash or not np.array_equal(ck["catalog"].cpu().numpy(), data.catalog):
        raise ValueError("Checkpoint catalog/protocol mismatch")
    model = Policy(**ck["settings"]).cuda().eval()
    model.load_state_dict(ck["model"])
    t = proto["time"]
    gap, horizon, train_end = t["gap_seconds"], t["horizon_seconds"], t["train_end_exclusive"]

    # 1. In-sample days: label window ends at or before train_end, so the model trained on these targets.
    report["in_sample_train_days"] = {}
    for days_back in (0, 3, 7, 14):
        cutoff = train_end - gap - horizon - days_back * horizon
        *_, gpt, freq = window(data, model, cutoff, gap, horizon)
        report["in_sample_train_days"][f"cutoff_{cutoff}"] = dict(summary(gpt, freq), days_before_train_end=days_back)

    # Selection window (already repeatedly used for development).
    cutoff = t["query_cutoffs"]["selection_validation"]
    uids, x, probs, truth, histories, gpt, freq = window(data, model, cutoff, gap, horizon)
    report["selection"] = summary(gpt, freq)

    # 2. Ordering only: same candidate set (recent-128 items), same GPT-2 fill for empty slots.
    gpt_hist, freq_hist, recent_share, mass, slot_share, rows = [], [], [], [], [], []
    for u, p, tokens in zip(uids, probs, x):
        hist = history_indices(tokens)
        counts = np.bincount(hist, minlength=len(data.catalog)).astype(np.float64)
        gpt_hist.append(data.catalog[rank_within_history(p, hist, p)].tolist())
        freq_hist.append(data.catalog[rank_within_history(counts, hist, p)].tolist())
        if u in truth:
            targets = truth[u]
            recent = {int(data.catalog[i]) for i in np.unique(hist)}
            recent_share.append(len(targets & recent) / len(targets))
            mass.append(float(p[np.unique(hist)].sum()))
            top = top_ids([p], data.catalog)[0]
            slot_share.append(sum(i in recent for i in top) / 10)
            rows.append({"uid": int(u), "history_len": int((tokens > 0).sum()),
                         "unique_recent_fraction": len(np.unique(hist)) / max(1, len(hist)),
                         "target_recent_share": recent_share[-1]})
    gh = ranking_metrics(uids, gpt_hist, truth)
    fh = ranking_metrics(uids, freq_hist, truth)
    report["ordering_within_recent128"] = {
        "GPT2_order_NDCG@10": gh[0]["NDCG@10"], "Frequency_order_NDCG@10": fh[0]["NDCG@10"],
        "gpt_order_minus_frequency_order": paired_bootstrap(gh[1], fh[1]),
        "full_GPT2_minus_GPT2_order_within_history": paired_bootstrap(gpt[1], gh[1]),
        "note": "Both rank only items present in the last 128 events, then fill with GPT-2 top non-history items."}

    # 3. Repeat mass versus observed share (active users).
    report["repeat_mass"] = {
        "mean_gpt_probability_on_recent128_items": float(np.mean(mass)),
        "mean_observed_share_of_tomorrow_targets_in_recent128": float(np.mean(recent_share)),
        "mean_gpt_top10_slots_on_recent128_items": float(np.mean(slot_share)),
        "frequency_top10_slots_on_recent128_items": "1.0 whenever >=10 distinct recent items"}

    # 4. Gap by user terciles.
    delta = {r["uid"]: a["ndcg"] - b["ndcg"] for r, a, b in zip(rows, gpt[1], freq[1])}
    assert [r["uid"] for r in rows] == [a["uid"] for a in gpt[1]]
    report["gap_by_user_tercile"] = {}
    for key in ("history_len", "unique_recent_fraction", "target_recent_share"):
        values = np.array([r[key] for r in rows])
        groups = np.empty(len(values), dtype=int)
        groups[np.argsort(values, kind="stable")] = np.arange(len(values)) * 3 // len(values)
        report["gap_by_user_tercile"][key] = [
            {"tercile": g, "users": int((groups == g).sum()),
             "range": [float(values[groups == g].min()), float(values[groups == g].max())],
             "mean_gpt_minus_frequency": float(np.mean([delta[r["uid"]] for r, gg in zip(rows, groups) if gg == g])),
             "mean_frequency_ndcg": float(np.mean([b["ndcg"] for b, gg in zip(freq[1], groups) if gg == g]))}
            for g in range(3)]
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(out / "report.json", report)
    atomic_json(out / "status.json", {"status": "completed"})


if __name__ == "__main__":
    main()
