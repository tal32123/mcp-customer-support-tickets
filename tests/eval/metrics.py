"""Ragas-vocabulary retrieval metrics.

Pure functions over (ranked_ids, relevant_ids). No external deps beyond stdlib
+ numpy. The ragas library itself is generation-centric and pulls langchain;
the retrieval metrics it ships are ~40 LOC of arithmetic that we'd rather own
than pin around.

All metrics take:
    ranked_ids: list[str]   # what the retriever returned, best-first
    relevant_ids: set[str]  # ground-truth relevant doc ids

`@k` variants truncate `ranked_ids` to the first k entries before scoring.
"""

from __future__ import annotations
import math
from collections.abc import Iterable


def hit_rate_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant id appears in the first k results, else 0.0.

    Per-query metric; average across queries for a corpus-level number.
    """
    if k <= 0:
        return 0.0
    return 1.0 if any(rid in relevant_ids for rid in ranked_ids[:k]) else 0.0


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant hit (1-indexed). 0 if none."""
    for ix, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / ix
    return 0.0


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k that are relevant. Denominator is k, not |topk|,
    so a retriever that returns fewer than k is penalised."""
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    hits = sum(1 for rid in topk if rid in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant docs that landed in the top-k. 0 if no
    relevant docs exist (vacuous; caller should filter empty cases)."""
    if not relevant_ids or k <= 0:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k. DCG uses gain = 1 if relevant else 0, with
    log2(rank+1) discount; IDCG is the same with all relevant docs (up to k)
    packed at the top."""
    if k <= 0 or not relevant_ids:
        return 0.0
    dcg = 0.0
    for ix, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(ix + 1)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average(values: Iterable[float]) -> float:
    """Plain mean. Returns 0.0 on empty input so test reports never NaN."""
    seq = list(values)
    return sum(seq) / len(seq) if seq else 0.0
