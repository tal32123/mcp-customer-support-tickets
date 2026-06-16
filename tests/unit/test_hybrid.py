import numpy as np
import pytest
from mcp_cst.retrieval.hybrid import reciprocal_rank_fusion, hybrid_search
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
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s", revision="r", rows=raw_ticket_rows, embedder=deterministic_embedder,
    )


def test_hybrid_search_returns_ids(store):
    hits = hybrid_search(
        store, query="login", filters={}, embedder=deterministic_embedder, limit=5,
    )
    assert 1 <= len(hits) <= 5
    for h in hits:
        assert "id" in h
        assert "subject" in h
        assert "snippet" in h
        assert len(h["snippet"]) <= 240


def test_hybrid_search_filters(store):
    hits = hybrid_search(
        store, query="login", filters={"language": "de"}, embedder=deterministic_embedder, limit=10,
    )
    # all hits must be German (filter enforced in both BM25 and vector branches)
    # We check by re-fetching each via store
    for h in hits:
        rec = store.get(h["id"])
        assert rec.language == "de"


def test_hybrid_respects_limit(store):
    hits = hybrid_search(
        store, query="app", filters={}, embedder=deterministic_embedder, limit=3,
    )
    assert len(hits) <= 3
