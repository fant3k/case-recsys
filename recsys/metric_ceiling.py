"""Best achievable NDCG@10 / Recall@10 / HitRate@10 under our protocol (no model; labels only).

An oracle recommends the user's own future positives that are in the fixed 50 000-track catalog; targets outside the
catalog stay misses in the denominator, exactly as for every method. Windows that are already opened only.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

from .data import Sequences, load_sequences
from .metrics import ranking_metrics

ROOT = Path(__file__).resolve().parents[1]


def oracle_recommendations(uids, truth, catalog, k=10):
    in_catalog = set(map(int, catalog))
    filler = [int(i) for i in catalog[:k * 4]]
    recs = []
    for uid in uids:
        hits = sorted(i for i in truth.get(uid, ()) if i in in_catalog)[:k]
        used = set(hits)
        recs.append(hits + [i for i in filler if i not in used and i not in truth.get(uid, ())][:k - len(hits)])
    return recs


def ceiling(data, cutoff, lo, hi):
    uids, _, _, _ = data.queries(cutoff)
    truth = data.truth(lo, hi)
    metrics, _ = ranking_metrics(uids, oracle_recommendations(uids, truth, data.catalog), truth)
    sizes = np.array([len(truth[u]) for u in uids if u in truth])
    hit = float(np.mean([any(i in set(map(int, data.catalog)) for i in truth[u]) for u in uids if u in truth]))
    return {**metrics, "HitRate@10": hit, "mean_positives_per_active_user": float(sizes.mean()),
            "median_positives": float(np.median(sizes)), "share_users_with_more_than_10": float((sizes > 10).mean())}


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()  # no options; makes --help safe
    for opened in ("FINAL-TEST-OPENED.json", "NEW-COHORT-OPENED.json"):
        if not (ROOT / "artifacts" / opened).exists() and not list((ROOT / "research/results/runs").glob("*/" + opened)):
            raise SystemExit("The ceiling reads the final window and the new cohort; they are opened only after their one-time runs (" + opened + " not found)")
    proto = json.loads((ROOT / "research/protocol/case2-v1.json").read_text())
    t = proto["time"]
    original, _, _, _, _ = load_sequences(ROOT / proto["data"]["sample_path"], t["train_end_exclusive"])
    cohort = pl.read_parquet(ROOT / "data/processed/yambda-new-cohort-1000.parquet")
    new = Sequences(list(cohort.iter_rows(named=True)), original.catalog, original.train_end, original.context_size)
    out = {}
    for name, data, window in (("original_selection", original, "selection_validation"),
                               ("original_final_test", original, "final_test"),
                               ("new_cohort_final_day", new, "final_test")):
        cutoff = t["query_cutoffs"][window]
        lo = cutoff + t["gap_seconds"]
        out[name] = ceiling(data, cutoff, lo, lo + t["horizon_seconds"])
    path = ROOT / "research/results/metric-ceiling-2026-10-25.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
