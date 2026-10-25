"""Alternative reward targets computed from logged listens only (no impressions or propensities).

continuation: completed (>50%) listens strictly later in the same session (pauses <= 20 min),
    capped at 10 and divided by 10. Events at or after `cap` are not counted (right censoring
    at the end of the available period), so training labels never look past train end.
quantile: share of the user's STRICTLY EARLIER listens in the same log2 track-length bucket
    whose played seconds are <= this listen's (OneRec-V2 duration-aware shaping). 0.5 when
    fewer than `min_prior` earlier listens exist in the bucket.
"""
from bisect import bisect_right, insort

import numpy as np

SESSION_GAP = 1200


def continuation_labels(user, cap, gap=SESSION_GAP, cap_count=10):
    times = user["times"]
    n = len(times)
    if not n:
        return np.zeros(0, dtype=np.float32)
    session = np.r_[0, np.cumsum(np.diff(times) > gap)]
    positive = ((user["ratios"] > 0.5) & (times < cap)).astype(np.int64)
    cumulative = np.cumsum(positive)
    session_last = np.r_[np.flatnonzero(np.diff(session)), n - 1][session]
    group_last = np.repeat(user["ends"], user["ends"] - user["starts"]) - 1
    later = cumulative[session_last] - cumulative[group_last]
    return (np.minimum(later, cap_count) / cap_count).astype(np.float32)


def length_bucket(seconds):
    return np.floor(np.log2(np.maximum(seconds, 0) + 1)).astype(np.int64)


def quantile_labels(user, length_seconds, min_prior=5):
    """length_seconds: per-event track length (train median for catalog items)."""
    played = user["ratios"] * length_seconds
    bucket = length_bucket(length_seconds)
    out = np.full(len(played), 0.5, dtype=np.float32)
    history = {}
    for a, b in zip(user["starts"], user["ends"]):
        for i in range(a, b):
            prior = history.get(int(bucket[i]))
            if prior is not None and len(prior) >= min_prior:
                out[i] = bisect_right(prior, played[i]) / len(prior)
        for i in range(a, b):
            insort(history.setdefault(int(bucket[i]), []), float(played[i]))
    return out


def event_lengths(user, features, offset, mean_length_norm):
    """Train-median capped length in seconds per event; catalog mean for unknown items."""
    known = user["tokens"] >= offset
    out = np.full(len(user["tokens"]), mean_length_norm * 600, dtype=np.float64)
    out[known] = features[user["tokens"][known] - offset, 0] * 600
    return out
