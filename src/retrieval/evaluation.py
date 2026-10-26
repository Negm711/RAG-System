"""Retrieval metrics for explicitly labelled relevant chunk IDs."""
import math


def score_ranking(retrieved, relevant, k):
    if k < 1 or not relevant:
        raise ValueError("Positive k and nonempty relevance labels are required")
    retrieved = [str(x) for x in retrieved]
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("Retrieved chunk IDs must be unique")
    relevant = {str(x) for x in relevant}
    hits = [value in relevant for value in retrieved[:k]]
    return {
        "recall_at_k": sum(hits) / len(relevant),
        "reciprocal_rank_at_k": next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0),
    }


def evaluate(rows, k):
    if not rows:
        raise ValueError("Evaluation rows cannot be empty")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Question IDs must be unique")
    totals = {"baseline": [], "reranked": []}
    for row in rows:
        for mode in totals:
            latency = float(row[mode + "_ms"])
            if not math.isfinite(latency) or latency < 0:
                raise ValueError("Latency must be finite and nonnegative")
            metrics = score_ranking(row[mode + "_ids"], row["relevant_ids"], k)
            totals[mode].append(dict(metrics, latency_ms=latency))
    return {"questions": len(rows), "k": k, "metrics": {
        mode: {key: sum(item[key] for item in values) / len(values)
               for key in values[0]} for mode, values in totals.items()
    }}
