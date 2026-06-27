"""Unit tests for the retrieval metrics. Shipping a buggy MRR would
silently invalidate every other eval number — guard that here."""

from __future__ import annotations
import math

from tests.eval.metrics import (
    average,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


RELEVANT = {"a", "b"}


def test_hit_rate_at_k_hits_when_relevant_in_topk():
    assert hit_rate_at_k(["x", "a", "y"], RELEVANT, k=3) == 1.0
    assert hit_rate_at_k(["x", "a", "y"], RELEVANT, k=1) == 0.0
    assert hit_rate_at_k([], RELEVANT, k=5) == 0.0
    assert hit_rate_at_k(["a"], RELEVANT, k=0) == 0.0


def test_mrr_uses_first_relevant_position():
    assert mrr(["a", "x"], RELEVANT) == 1.0
    assert mrr(["x", "a"], RELEVANT) == 0.5
    assert mrr(["x", "y", "a"], RELEVANT) == 1.0 / 3
    assert mrr(["x", "y"], RELEVANT) == 0.0
    # Only the first relevant hit counts.
    assert mrr(["x", "a", "b"], RELEVANT) == 0.5


def test_precision_at_k_denominator_is_k():
    assert precision_at_k(["a", "b", "x"], RELEVANT, k=3) == 2 / 3
    # Retriever returning fewer than k is penalised: 1 hit / 5 = 0.2
    assert precision_at_k(["a"], RELEVANT, k=5) == 1 / 5
    assert precision_at_k([], RELEVANT, k=3) == 0.0


def test_recall_at_k_against_full_relevant_set():
    assert recall_at_k(["a", "x"], RELEVANT, k=5) == 0.5  # 1 of 2 found
    assert recall_at_k(["a", "b"], RELEVANT, k=5) == 1.0
    assert recall_at_k(["x", "y"], RELEVANT, k=5) == 0.0
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0  # vacuous


def test_ndcg_at_k_perfect_ranking_is_one():
    # All relevant docs at the top → NDCG == 1.0
    assert ndcg_at_k(["a", "b", "x", "y"], RELEVANT, k=4) == 1.0


def test_ndcg_at_k_demoted_ranking_lt_one_gt_zero():
    score = ndcg_at_k(["x", "y", "a", "b"], RELEVANT, k=4)
    assert 0.0 < score < 1.0
    # Verify the actual value:
    # DCG = 1/log2(4) + 1/log2(5)
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3)
    expected_dcg = 1 / math.log2(4) + 1 / math.log2(5)
    expected_idcg = 1.0 + 1 / math.log2(3)
    assert math.isclose(score, expected_dcg / expected_idcg, rel_tol=1e-9)


def test_ndcg_at_k_empty_relevant_returns_zero():
    assert ndcg_at_k(["a"], set(), k=3) == 0.0


def test_average_handles_empty():
    assert average([]) == 0.0
    assert average([1.0, 1.0, 0.0, 0.0]) == 0.5
