"""Select a fresh cohort of 1000 Yambda users not in the original sample (docs/NEW-COHORT-PLAN.md).

Selection uses user ids only; no event of the new users is summarised or evaluated here.
"""
from pathlib import Path
import hashlib
import json
import random

import polars as pl

root = Path(__file__).resolve().parents[2]
raw = root / "data/raw/yambda-sequential-50m-listens.parquet"
original = root / "data/processed/yambda-1000-users-full-histories.parquet"
out = root / "data/processed/yambda-new-cohort-1000.parquet"
if out.exists():
    raise SystemExit("Cohort already selected; not re-sampling")
all_uids = sorted(pl.scan_parquet(raw).select("uid").collect()["uid"].to_list())
used = set(pl.read_parquet(original, columns=["uid"])["uid"].to_list())
remaining = [u for u in all_uids if u not in used]
uids = random.Random(20261024).sample(remaining, 1000)
cohort = pl.scan_parquet(raw).filter(pl.col("uid").is_in(uids)).collect()
assert cohort.height == 1000 and not (set(cohort["uid"].to_list()) & used)
cohort.write_parquet(out)
report = {"seed": 20261024, "all_users": len(all_uids), "original_users": len(used), "remaining_users": len(remaining),
          "cohort_users": cohort.height, "overlap_with_original": 0,
          "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
          "cohort_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
          "note": "uid-only selection; no event statistics of the cohort were computed"}
(root / "research/selection/new-cohort-selection.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
