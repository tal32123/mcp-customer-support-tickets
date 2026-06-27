import numpy as np
import pytest
from mcp_cst.retrieval.hybrid import (
    hybrid_search,
    hybrid_search_full,
    hydrate_ids,
    reciprocal_rank_fusion,
)
from mcp_cst.data.store import TicketStore


def test_rrf_merges_by_rank():
    bm25 = ["a", "b", "c", "d"]
    vec = ["c", "a", "x", "y"]
    out = reciprocal_rank_fusion([bm25, vec], k=60)
    # `a` appears at rank 1 (bm25) and rank 2 (vec) → highest combined score
    assert out[0] == "a"
    assert "c" in out[:3]


def test_rrf_handles_disjoint_lists():
    a = ["1", "2", "3"]
    b = ["4", "5", "6"]
    out = reciprocal_rank_fusion([a, b])
    assert set(out) == {"1", "2", "3", "4", "5", "6"}


def deterministic_embedder(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower()))
        for j in range(384):
            out[i, j] = ((h >> (j % 32)) & 0xFF) / 255.0
    return out


@pytest.fixture
def store(pg_dsn, pg_schema, raw_ticket_rows):
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="r",
        rows=raw_ticket_rows,
        embedder=deterministic_embedder,
    )
    yield s
    s.close()


def test_hybrid_search_returns_ids(store):
    hits = hybrid_search(
        store,
        query="login",
        filters={},
        embedder=deterministic_embedder,
        limit=5,
    )
    assert 1 <= len(hits) <= 5
    for h in hits:
        assert "id" in h
        assert "subject" in h
        assert "snippet" in h
        assert len(h["snippet"]) <= 240


def test_hybrid_search_filters(store):
    hits = hybrid_search(
        store,
        query="login",
        filters={"language": "de"},
        embedder=deterministic_embedder,
        limit=10,
    )
    # all hits must be German (filter enforced in both BM25 and vector branches)
    # We check by re-fetching each via store
    for h in hits:
        rec = store.get(h["id"])
        assert rec.language == "de"


def test_hybrid_respects_limit(store):
    hits = hybrid_search(
        store,
        query="app",
        filters={},
        embedder=deterministic_embedder,
        limit=3,
    )
    assert len(hits) <= 3


def test_rrf_empty_input_returns_empty():
    """Both branches returning nothing must yield [] rather than crash."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_dedupes_same_id_in_both_lists():
    """A doc that appears in BM25 and vector must surface once, not twice."""
    out = reciprocal_rank_fusion([["a", "b"], ["a", "c"]])
    assert out.count("a") == 1
    assert set(out) == {"a", "b", "c"}


def test_rrf_single_list_preserves_order():
    """If one branch is empty (e.g. FTS index missing), the other drives results."""
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]
    assert reciprocal_rank_fusion([["a", "b", "c"], []]) == ["a", "b", "c"]


def test_hybrid_search_tag_filter_recall_floor(pg_dsn, pg_schema):
    """Seed N>limit tickets all tagged ['a','b']; with tags=['a','b'] and
    limit=K<N the search must return exactly K hits. Regression guard for
    the old post-filter behavior that silently shrank recall when the
    candidate window contained untagged rows."""
    rows = [
        {
            "subject": f"Package status {i}",
            "body": f"Where is my package number {i}?",
            "answer": "",
            "type": "question",
            "queue": "Shipping",
            "priority": "low",
            "language": "en",
            "version": "1.0",
            "tag_1": "a",
            "tag_2": "b",
            "tag_3": "",
            "tag_4": "",
            "tag_5": "",
            "tag_6": "",
        }
        for i in range(80)
    ]
    # Add many untagged distractors that the BM25/vector branches will
    # otherwise pull into the candidate window.
    rows.extend(
        {
            "subject": "Other ticket about package",
            "body": f"Unrelated package note {i}",
            "answer": "",
            "type": "question",
            "queue": "Other",
            "priority": "low",
            "language": "en",
            "version": "1.0",
            "tag_1": "x",
            "tag_2": "",
            "tag_3": "",
            "tag_4": "",
            "tag_5": "",
            "tag_6": "",
        }
        for i in range(120)
    )
    store = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="r",
        rows=rows,
        embedder=deterministic_embedder,
    )
    hits = hybrid_search(
        store,
        query="package",
        filters={"tags": ["a", "b"], "tags_mode": "and"},
        embedder=deterministic_embedder,
        limit=25,
    )
    assert len(hits) == 25


def test_hybrid_search_full_returns_all_fused_ids(store):
    """hybrid_search_full returns the unsliced fused id list — the basis
    for cursor pagination. Must be a superset of any limit-K slice."""
    full = hybrid_search_full(
        store,
        query="login",
        filters={},
        embedder=deterministic_embedder,
    )
    assert isinstance(full, list)
    assert all(isinstance(i, str) for i in full)
    # The thin wrapper's top-K must match the head of the full list.
    limited = hybrid_search(
        store,
        query="login",
        filters={},
        embedder=deterministic_embedder,
        limit=3,
    )
    assert [h["id"] for h in limited] == full[:3]


def test_hydrate_ids_preserves_order_and_ranks(store):
    """hydrate_ids takes a slice of ids and produces hit dicts in input
    order with score_rank offset by `rank_offset`."""
    full = hybrid_search_full(
        store,
        query="login",
        filters={},
        embedder=deterministic_embedder,
    )
    slice_ = full[2:5]
    hits = hydrate_ids(store, slice_, rank_offset=2)
    assert [h["id"] for h in hits] == slice_
    assert [h["score_rank"] for h in hits] == [3, 4, 5]


def test_hybrid_search_tags_and_vs_or(store):
    """tags_mode='and' requires every tag; tags_mode='or' requires any.

    Postcondition guarantees per-hit: AND hits contain ALL tags; OR hits
    contain ANY tag. (The previous "AND ids ⊆ OR ids" invariant held by
    accident with the old post-filter — with pushdown, the two queries see
    different candidate windows, so a global subset relation is not
    guaranteed.)
    """
    common = {"shipping", "urgent"}
    and_hits = hybrid_search(
        store,
        query="package",
        filters={"tags": list(common), "tags_mode": "and"},
        embedder=deterministic_embedder,
        limit=50,
    )
    or_hits = hybrid_search(
        store,
        query="package",
        filters={"tags": list(common), "tags_mode": "or"},
        embedder=deterministic_embedder,
        limit=50,
    )
    for h in and_hits:
        rec = store.get(h["id"])
        assert common.issubset(set(rec.tags))
    for h in or_hits:
        rec = store.get(h["id"])
        assert common & set(rec.tags)
