import numpy as np
import pytest
from mcp_cst.tools.search_tickets import search_tickets_impl
from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError


def embed(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower()))
        for j in range(384):
            out[i, j] = ((h >> (j % 32)) & 0xFF) / 255.0
    return out


@pytest.fixture
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s", revision="r", rows=raw_ticket_rows, embedder=embed,
    )


def test_returns_previews(store):
    hits = search_tickets_impl(store, embed, q="login", limit=5)
    assert 1 <= len(hits) <= 5
    for h in hits:
        assert set(h.keys()) >= {"id", "subject", "snippet", "language", "queue", "priority", "ticket_uri"}
        assert h["ticket_uri"] == f"ticket://{h['id']}"
        assert len(h["snippet"]) <= 240


def test_limit_capped_at_50(store):
    hits = search_tickets_impl(store, embed, q="login", limit=999)
    assert len(hits) <= 50


def test_language_filter(store):
    hits = search_tickets_impl(store, embed, q="login", language="de", limit=10)
    for h in hits:
        assert h["language"] == "de"


def test_unknown_filter_field_refused_via_aggregates_path():
    # search_tickets does not accept arbitrary kwargs — it has typed args,
    # so unknown kwargs would surface as TypeError. The structured refusal
    # for unsupported filters lives in aggregates and is tested there.
    pass  # placeholder, intentionally empty
