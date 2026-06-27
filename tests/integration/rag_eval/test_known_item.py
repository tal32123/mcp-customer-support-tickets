"""Known-item retrieval metrics under the real e5-small embedder."""

from __future__ import annotations

import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder

from tests.eval.metrics import average, hit_rate_at_k, mrr, ndcg_at_k
from tests.integration.rag_eval.conftest import ranked_ids


def test_subject_known_item_hit_rate_overall(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    """Subject verbatim query: hit-rate@10 >= 0.90."""
    seeds = [s for s in eval_seeds if s["subject_query"].strip()]
    scores = [
        hit_rate_at_k(
            ranked_ids(eval_store, real_embedder.embed_queries, s["subject_query"]),
            {s["id"]},
            k=10,
        )
        for s in seeds
    ]
    avg = average(scores)
    record_summary("subject_hit_rate_10", avg)
    assert avg >= 0.90, (
        f"subject known-item hit-rate@10: expected >= 0.90, got {avg:.4f} "
        f"(n={len(scores)})"
    )


def test_subject_known_item_mrr_overall(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    """Subject verbatim query: MRR@10 >= 0.75."""
    seeds = [s for s in eval_seeds if s["subject_query"].strip()]
    scores = [
        mrr(
            ranked_ids(eval_store, real_embedder.embed_queries, s["subject_query"]),
            {s["id"]},
        )
        for s in seeds
    ]
    avg = average(scores)
    record_summary("subject_mrr_10", avg)
    assert avg >= 0.75, (
        f"subject known-item MRR@10: expected >= 0.75, got {avg:.4f} "
        f"(n={len(scores)})"
    )


def test_body_slice_known_item_hit_rate_overall(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    """Body-slice query: hit-rate@10 >= 0.80."""
    scores = [
        hit_rate_at_k(
            ranked_ids(eval_store, real_embedder.embed_queries, s["body_query"]),
            {s["id"]},
            k=10,
        )
        for s in eval_seeds
    ]
    avg = average(scores)
    record_summary("body_hit_rate_10", avg)
    assert avg >= 0.80, (
        f"body-slice known-item hit-rate@10: expected >= 0.80, got {avg:.4f} "
        f"(n={len(scores)})"
    )


def test_body_slice_known_item_mrr_overall(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    """Body-slice query: MRR@10 >= 0.60."""
    scores = [
        mrr(
            ranked_ids(eval_store, real_embedder.embed_queries, s["body_query"]),
            {s["id"]},
        )
        for s in eval_seeds
    ]
    avg = average(scores)
    record_summary("body_mrr_10", avg)
    assert avg >= 0.60, (
        f"body-slice known-item MRR@10: expected >= 0.60, got {avg:.4f} "
        f"(n={len(scores)})"
    )


def test_body_slice_known_item_ndcg_overall(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    """Body-slice query: NDCG@10 >= 0.65."""
    scores = [
        ndcg_at_k(
            ranked_ids(eval_store, real_embedder.embed_queries, s["body_query"]),
            {s["id"]},
            k=10,
        )
        for s in eval_seeds
    ]
    avg = average(scores)
    record_summary("body_ndcg_10", avg)
    assert avg >= 0.65, (
        f"body-slice known-item NDCG@10: expected >= 0.65, got {avg:.4f} "
        f"(n={len(scores)})"
    )


@pytest.mark.parametrize("language", ["en", "de", "he"])
def test_per_language_hit_rate_floor(
    language: str,
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    """Per-language body-slice hit-rate@10 >= 0.75. 'he' skips (no data)."""
    lang_seeds = [s for s in eval_seeds if s["language"] == language]
    if not lang_seeds:
        pytest.skip(
            f"no seeds for language {language!r} in the current dataset"
        )
    scores = [
        hit_rate_at_k(
            ranked_ids(eval_store, real_embedder.embed_queries, s["body_query"]),
            {s["id"]},
            k=10,
        )
        for s in lang_seeds
    ]
    avg = average(scores)
    record_summary(f"body_hit_rate_10_{language}", avg)
    assert avg >= 0.75, (
        f"per-language body-slice hit-rate@10 [{language}]: expected >= 0.75, "
        f"got {avg:.4f} (n={len(scores)})"
    )
