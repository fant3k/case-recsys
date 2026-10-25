"""Count support for predefined time windows without scoring a recommender."""

import hashlib
import json
import math
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("case2-v1.json")


def main():
    config = json.loads(PROTOCOL.read_text())
    source = ROOT / config["data"]["sample_path"]
    seq = pl.read_parquet(source)
    events = seq.explode([c for c in seq.columns if c != "uid"])
    train_end = config["time"]["train_end_exclusive"]
    train = events.filter(pl.col("timestamp") < train_end)
    catalog = (
        train.group_by("item_id")
        .len()
        .sort(["len", "item_id"], descending=[True, False])
        .head(config["catalog"]["size"])["item_id"]
    )
    catalog_set = set(catalog.to_list())
    assert len(catalog) == config["catalog"]["size"]
    windows = []
    previous_end = train_end
    for name, cutoff in config["time"]["query_cutoffs"].items():
        start = cutoff + config["time"]["gap_seconds"]
        end = start + config["time"]["horizon_seconds"]
        assert previous_end <= cutoff < start < end
        assert end <= events["timestamp"].max()
        history = events.filter(pl.col("timestamp") < cutoff)
        warm = history["uid"].unique()
        targets = (
            events.filter(
                (pl.col("timestamp") >= start)
                & (pl.col("timestamp") < end)
                & (pl.col("played_ratio_pct") > 50)
                & pl.col("uid").is_in(warm.implode())
            )
            .select("uid", "item_id")
            .unique()
        )
        truth = {}
        for uid, item in targets.iter_rows():
            truth.setdefault(uid, set()).add(item)
        assert truth
        counts = [(len(items), len(items & catalog_set)) for items in truth.values()]
        k = config["evaluation"]["k"]
        ideal_dcg = lambda n: sum(1 / math.log2(j + 2) for j in range(min(k, n)))
        windows.append(
            {
                "window": name,
                "query_cutoff": cutoff,
                "labels_start": start,
                "labels_end_exclusive": end,
                "eligible_warm_users": len(warm),
                "users_with_future_positive": len(truth),
                "positive_pairs": len(targets),
                "positive_pairs_in_catalog": sum(known for _, known in counts),
                "catalog_positive_pair_coverage": sum(known for _, known in counts)
                / len(targets),
                "macro_recall_at_k_catalog_ceiling": sum(
                    min(k, known) / total for total, known in counts
                )
                / len(counts),
                "macro_ndcg_at_k_catalog_ceiling": sum(
                    ideal_dcg(known) / ideal_dcg(total) for total, known in counts
                )
                / len(counts),
                "note": "Catalog-only feasibility bounds, not scores of any recommender.",
            }
        )
        previous_end = end
    result = {
        "purpose": "Support audit only; no predictions, reward scores or model comparisons.",
        "protocol": config["protocol_id"],
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_users": len(seq),
        "train_end_exclusive": train_end,
        "train_events": len(train),
        "catalog_size": len(catalog),
        "train_catalog_event_coverage": train.filter(
            pl.col("item_id").is_in(catalog.implode())
        ).height
        / len(train),
        "windows": windows,
    }
    destination = Path(__file__).with_name("time-window-support.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
