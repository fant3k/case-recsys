"""Past-only group sampling. Item tokens 0/1/2 are PAD/UNK/BOS."""

from collections import Counter

import numpy as np
import polars as pl

PAD, UNK, BOS, OFFSET = 0, 1, 2, 3


def build_catalog(events, train_end, size):
    train = events.filter(pl.col("timestamp") < train_end)
    top = train.group_by("item_id").len().sort(
        ["len", "item_id"], descending=[True, False]
    ).head(size)
    return top["item_id"].to_numpy().astype(np.int64)


def encode_items(items, catalog):
    order = np.argsort(catalog)
    sorted_ids = catalog[order]
    pos = np.searchsorted(sorted_ids, items)
    bounded = np.minimum(pos, len(catalog) - 1)
    valid = (pos < len(catalog)) & (sorted_ids[bounded] == items)
    result = np.full(len(items), UNK, dtype=np.int64)
    result[valid] = order[bounded[valid]] + OFFSET
    return result


def pack_contexts(contexts, context_size):
    x = np.zeros((len(contexts), context_size), dtype=np.int64)
    mask = np.zeros_like(x)
    for j, context in enumerate(contexts):
        context = np.asarray(context, dtype=np.int64)[-context_size:]
        if not len(context):
            context = np.array([BOS], dtype=np.int64)
        x[j, : len(context)] = context
        mask[j, : len(context)] = 1
    return x, mask


class Sequences:
    def __init__(self, rows, catalog, train_end, context_size=128):
        self.catalog = np.asarray(catalog, dtype=np.int64)
        self.train_end = train_end
        self.context_size = context_size
        self.users = []
        self.train_users = []
        for row in rows:
            times = np.asarray(row["timestamp"], dtype=np.int64)
            items = np.asarray(row["item_id"], dtype=np.int64)
            order = np.lexsort((items, times))
            times, items = times[order], items[order]
            tokens = encode_items(items, self.catalog)
            ends = np.r_[np.flatnonzero(np.diff(times)) + 1, len(times)]
            starts = np.r_[0, ends[:-1]]
            known = np.flatnonzero(tokens >= OFFSET)
            first = np.searchsorted(known, starts)
            last = np.searchsorted(known, ends)
            eligible = np.flatnonzero(
                (starts > 0) & (times[starts] < train_end) & (last > first)
            )
            self.users.append(
                {
                    "uid": int(row["uid"]), "times": times, "items": items,
                    "tokens": tokens, "starts": starts, "ends": ends,
                    "known": known, "first": first, "last": last,
                    "eligible": eligible,
                    "ratios": np.minimum(np.asarray(row["played_ratio_pct"])[order], 100).astype(np.float32) / 100,
                    "organic": np.asarray(row.get("is_organic", np.zeros(len(order))))[order].astype(np.int8),
                }
            )
            if len(eligible):
                self.train_users.append(len(self.users) - 1)
        if not self.train_users:
            raise ValueError("No eligible training users")

    def context(self, user, end):
        return user["tokens"][max(0, end - self.context_size):end]

    def sample(self, rng, batch_size, bounds=None):
        contexts, actions, ratios, records = [], [], [], []
        if bounds is None:
            choices = self.train_users
            groups = None
        else:
            choices, groups = [], {}
            lo, hi = bounds
            for j, user in enumerate(self.users):
                starts = user["starts"]
                eligible = np.flatnonzero(
                    (starts > 0) & (user["times"][starts] >= lo)
                    & (user["times"][starts] < hi)
                    & (user["last"] > user["first"])
                )
                if len(eligible):
                    choices.append(j)
                    groups[j] = eligible
            if not choices:
                raise ValueError("No eligible observed holdout events")
        for _ in range(batch_size):
            j = int(rng.choice(choices))
            user = self.users[j]
            group = int(rng.choice(user["eligible"] if groups is None else groups[j]))
            event = int(user["known"][rng.integers(user["first"][group], user["last"][group])])
            end = int(user["starts"][group])
            assert user["times"][end - 1] < user["times"][event]
            contexts.append(self.context(user, end))
            actions.append(int(user["tokens"][event]) - OFFSET)
            ratios.append(float(user["ratios"][event]))
            records.append((user["uid"], event, int(user["times"][event])))
        x, mask = pack_contexts(contexts, self.context_size)
        return x, mask, np.asarray(actions), np.asarray(ratios, dtype=np.float32), records

    def queries(self, cutoff):
        users, contexts, raw_contexts = [], [], []
        for user in self.users:
            end = int(np.searchsorted(user["times"], cutoff))
            if end:
                users.append(user["uid"])
                contexts.append(self.context(user, end))
                raw_contexts.append(user["items"][max(0, end-self.context_size):end])
        return users, *pack_contexts(contexts, self.context_size), raw_contexts

    def truth(self, start, end):
        result = {}
        for user in self.users:
            lo, hi = np.searchsorted(user["times"], [start, end])
            good = user["ratios"][lo:hi] > 0.5
            items = set(map(int, user["items"][lo:hi][good]))
            if items:
                result[user["uid"]] = items
        return result

    def next_group_queries(self, start, end):
        ids, contexts, truth = [], [], {}
        for user in self.users:
            g = int(np.searchsorted(user["times"][user["starts"]], start))
            if g == len(user["starts"]):
                continue
            first, last = int(user["starts"][g]), int(user["ends"][g])
            if first == 0 or user["times"][first] >= end:
                continue
            ids.append(user["uid"])
            contexts.append(self.context(user, first))
            truth[user["uid"]] = set(map(int, user["items"][first:last]))
        return ids, *pack_contexts(contexts, self.context_size), truth


def load_sequences(path, train_end, catalog_size=50000, context_size=128):
    seq = pl.read_parquet(path)
    events = seq.explode([c for c in seq.columns if c != "uid"])
    catalog = build_catalog(events, train_end, catalog_size)
    train = events.filter(pl.col("timestamp") < train_end).with_columns(
        (pl.col("played_ratio_pct").clip(0, 100) / 100).alias("response")
    )
    stats = train.group_by("item_id").agg(
        pl.len().alias("count"), pl.col("response").mean().alias("mean"),
        pl.col("track_length_seconds").median().alias("length"),
        pl.col("track_length_seconds").n_unique().alias("length_variants"),
    )
    by_item = {int(r["item_id"]): r for r in stats.iter_rows(named=True)}
    lengths = np.array([by_item[int(i)]["length"] for i in catalog], dtype=np.float32)
    counts = np.array([by_item[int(i)]["count"] for i in catalog], dtype=np.float32)
    means = np.array([by_item[int(i)]["mean"] for i in catalog], dtype=np.float32)
    features = np.stack([np.minimum(lengths, 600) / 600, np.log1p(counts) / np.log1p(counts.max())], axis=1)
    users = Sequences(list(seq.iter_rows(named=True)), catalog, train_end, context_size)
    known_train = train.filter(pl.col("item_id").is_in(pl.Series(catalog).implode()))
    report = {
        "users": len(users.users), "eligible_training_users": len(users.train_users),
        "train_events": len(train), "catalog_size": len(catalog),
        "in_catalog_train_events": len(known_train),
        "eligible_target_groups": sum(len(u["eligible"]) for u in users.users),
        "items_with_varying_recorded_length": sum(by_item[int(i)]["length_variants"] > 1 for i in catalog),
        "target_sampling": "uniform eligible user, then group, then in-catalog event",
        "context_size": context_size,
    }
    return users, features, means, float(known_train["response"].mean()), report


def personal_frequency(raw_contexts, catalog, k):
    allowed = set(map(int, catalog))
    predictions = []
    for history in raw_contexts:
        counts = Counter(int(i) for i in history if int(i) in allowed)
        ranking = sorted(counts, key=lambda i: (-counts[i], i))[:k]
        used = set(ranking)
        ranking.extend(int(i) for i in catalog[:k] if int(i) not in used)
        predictions.append(ranking[:k])
    return predictions
