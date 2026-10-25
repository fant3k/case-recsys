import math


def ranking_metrics(user_ids, recommendations, truth, k=10):
    values = []
    for uid, ranking in zip(user_ids, recommendations):
        if uid not in truth:
            continue
        ranking = list(map(int, ranking[:k]))
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate recommendation would double-count a hit")
        relevant = truth[uid]
        hits = [int(item in relevant) for item in ranking]
        recall = sum(hits) / len(relevant)
        dcg = sum(hit/math.log2(i+2) for i, hit in enumerate(hits))
        ideal = sum(1/math.log2(i+2) for i in range(min(k, len(relevant))))
        values.append({"uid": int(uid), "recall": recall, "ndcg": dcg/ideal})
    if not values:
        raise ValueError("No evaluable users")
    return {
        "users": len(values),
        f"Recall@{k}": sum(v["recall"] for v in values)/len(values),
        f"NDCG@{k}": sum(v["ndcg"] for v in values)/len(values),
    }, values
