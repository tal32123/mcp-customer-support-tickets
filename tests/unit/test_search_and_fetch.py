import numpy as np
import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError
from mcp_cst.tools.search_and_fetch import search_and_fetch_impl


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
        path=tmp_path / "s",
        revision="r",
        rows=raw_ticket_rows,
        embedder=embed,
    )


def test_returns_full_rows_with_citation(store):
    hits = search_and_fetch_impl(store, embed, q="login", k=3)
    assert 1 <= len(hits) <= 3
    for h in hits:
        # Full ticket shape, not previews
        assert "subject" in h
        assert "body" in h
        assert "answer" in h
        assert h["wrapped"].startswith(f'<ticket id="{h["id"]}">')
        assert h["ticket_uri"] == f"ticket://{h['id']}"


def test_include_body_drops_answer(store):
    hits = search_and_fetch_impl(store, embed, q="login", k=3, include="body")
    assert hits
    for h in hits:
        assert "body" in h
        assert "answer" not in h


def test_include_answer_drops_body(store):
    hits = search_and_fetch_impl(store, embed, q="login", k=3, include="answer")
    assert hits
    for h in hits:
        assert "answer" in h
        assert "body" not in h


def test_k_capped_at_50(store):
    hits = search_and_fetch_impl(store, embed, q="login", k=999)
    assert len(hits) <= 50


def test_k_limit_applied(store):
    hits = search_and_fetch_impl(store, embed, q="login", k=2)
    assert len(hits) <= 2


def test_bad_include_rejected(store):
    with pytest.raises(McpCstError) as exc:
        search_and_fetch_impl(store, embed, q="login", k=3, include="garbage")  # type: ignore[arg-type]
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_short_query_rejected(store):
    with pytest.raises(McpCstError) as exc:
        search_and_fetch_impl(store, embed, q=" ", k=3)
    assert exc.value.code == ErrorCode.INVALID_INPUT
