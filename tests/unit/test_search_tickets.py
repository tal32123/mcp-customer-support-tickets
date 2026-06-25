import numpy as np
import pytest
from mcp_cst.errors import ErrorCode, McpCstError
from mcp_cst.tools.search_tickets import search_tickets_impl
from mcp_cst.data.store import TicketStore
from mcp_cst.retrieval import search_cache, hybrid as hybrid_mod


def embed(texts):
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t.lower()))
        for j in range(384):
            out[i, j] = ((h >> (j % 32)) & 0xFF) / 255.0
    return out


@pytest.fixture(autouse=True)
def _clear_cache():
    """Per-test isolation — the cache is module-level state."""
    search_cache.cache_clear()
    yield
    search_cache.cache_clear()


@pytest.fixture
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s",
        revision="r",
        rows=raw_ticket_rows,
        embedder=embed,
    )


def test_returns_previews(store):
    result = search_tickets_impl(store, embed, q="login", limit=5)
    assert set(result.keys()) == {"hits", "next_cursor", "search_id", "total_estimate"}
    hits = result["hits"]
    assert 1 <= len(hits) <= 5
    for h in hits:
        assert set(h.keys()) >= {
            "id",
            "subject",
            "snippet",
            "language",
            "queue",
            "priority",
            "ticket_uri",
            "score_rank",
        }
        assert h["ticket_uri"] == f"ticket://{h['id']}"
        assert len(h["snippet"]) <= 240


def test_limit_capped_at_50(store):
    result = search_tickets_impl(store, embed, q="login", limit=999)
    assert len(result["hits"]) <= 50


def test_language_filter(store):
    result = search_tickets_impl(store, embed, q="login", language="de", limit=10)
    for h in result["hits"]:
        assert h["language"] == "de"


def test_first_page_shape(store):
    result = search_tickets_impl(store, embed, q="login", limit=3)
    assert isinstance(result["search_id"], str) and len(result["search_id"]) == 16
    assert result["total_estimate"] >= len(result["hits"])
    # next_cursor should be set iff there are more results in the fused list
    if result["total_estimate"] > 3:
        assert result["next_cursor"] is not None
        assert result["next_cursor"].startswith(result["search_id"] + ":")
    else:
        assert result["next_cursor"] is None


def test_cursor_returns_next_slice_without_refusing(store, monkeypatch):
    """Page 2 must come from cache — RRF / fusion is not re-invoked."""
    page1 = search_tickets_impl(store, embed, q="login", limit=2)
    if page1["next_cursor"] is None:
        pytest.skip("not enough results to paginate")

    calls = {"n": 0}
    real_full = hybrid_mod.hybrid_search_full

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_full(*args, **kwargs)

    monkeypatch.setattr(
        "mcp_cst.tools.search_tickets.hybrid_search_full", spy
    )
    page2 = search_tickets_impl(
        store, embed, q="login", limit=2, cursor=page1["next_cursor"]
    )
    assert calls["n"] == 0  # cache hit, no fusion
    assert page2["search_id"] == page1["search_id"]
    page1_ids = {h["id"] for h in page1["hits"]}
    page2_ids = {h["id"] for h in page2["hits"]}
    assert page1_ids.isdisjoint(page2_ids)
    # score_rank continues from where page 1 left off
    assert page2["hits"][0]["score_rank"] == 3


def test_stable_search_id_for_same_q_and_filters(store):
    a = search_tickets_impl(store, embed, q="login", language="en", limit=3)
    b = search_tickets_impl(store, embed, q="login", language="en", limit=3)
    assert a["search_id"] == b["search_id"]
    # And different filters produce a different search_id
    c = search_tickets_impl(store, embed, q="login", language="de", limit=3)
    assert c["search_id"] != a["search_id"]


def test_garbage_cursor_raises_invalid_input(store):
    with pytest.raises(McpCstError) as exc:
        search_tickets_impl(store, embed, q="login", cursor="not-a-cursor")
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_unknown_search_id_cursor_raises_invalid_input(store):
    with pytest.raises(McpCstError) as exc:
        search_tickets_impl(
            store, embed, q="login", cursor="deadbeefdeadbeef:0"
        )
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_negative_offset_cursor_rejected(store):
    with pytest.raises(McpCstError) as exc:
        search_tickets_impl(store, embed, q="login", cursor="deadbeef:-1")
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_ttl_expiry_invalidates_cursor(store, monkeypatch):
    page1 = search_tickets_impl(store, embed, q="login", limit=2)
    if page1["next_cursor"] is None:
        pytest.skip("not enough results to paginate")
    # Jump time forward past the TTL.
    real_time = search_cache.time.time
    monkeypatch.setattr(
        search_cache.time,
        "time",
        lambda: real_time() + search_cache._SEARCH_CACHE_TTL_S + 1,
    )
    with pytest.raises(McpCstError) as exc:
        search_tickets_impl(
            store, embed, q="login", limit=2, cursor=page1["next_cursor"]
        )
    assert exc.value.code == ErrorCode.INVALID_INPUT


def test_last_page_returns_no_next_cursor(store):
    # Walk pages until next_cursor is None. Must terminate.
    total = search_tickets_impl(store, embed, q="login", limit=10)["total_estimate"]
    cursor = None
    collected = 0
    pages = 0
    while True:
        r = search_tickets_impl(store, embed, q="login", limit=10, cursor=cursor)
        collected += len(r["hits"])
        pages += 1
        if r["next_cursor"] is None:
            break
        cursor = r["next_cursor"]
        assert pages < 200  # safety belt
    assert collected == total


def test_empty_query_refused(store):
    with pytest.raises(McpCstError) as exc:
        search_tickets_impl(store, embed, q="")
    assert exc.value.code == ErrorCode.INVALID_INPUT
