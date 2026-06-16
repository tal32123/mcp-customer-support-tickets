from mcp_cst.retrieval.rerank import maybe_rerank


def test_passthrough_when_disabled():
    hits = [{"id": "1"}, {"id": "2"}]
    out = maybe_rerank(query="x", hits=hits, enabled=False)
    assert out == hits


def test_enabled_but_not_implemented_returns_hits_unchanged():
    # Stub: when enabled, the function should still return hits unchanged
    # (real reranker is deferred). We just want to be sure it doesn't blow up.
    hits = [{"id": "1"}, {"id": "2"}]
    out = maybe_rerank(query="x", hits=hits, enabled=True)
    assert len(out) == len(hits)
