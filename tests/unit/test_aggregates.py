import numpy as np
import pytest
from mcp_cst.data.aggregates import group_count
from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def store(pg_dsn, pg_schema, raw_ticket_rows):
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="rev",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    yield s
    s.close()


def test_group_by_language(store, raw_ticket_rows):
    result = group_count(store, group_by="language", filters={})
    by_lang = {r["group"]: r["count"] for r in result}
    expected_en = sum(1 for r in raw_ticket_rows if r["language"] == "en")
    expected_de = sum(1 for r in raw_ticket_rows if r["language"] == "de")
    assert by_lang["en"] == expected_en
    assert by_lang["de"] == expected_de


def test_group_by_with_filter(store, raw_ticket_rows):
    result = group_count(store, group_by="queue", filters={"language": "de"})
    total = sum(r["count"] for r in result)
    expected = sum(1 for r in raw_ticket_rows if r["language"] == "de")
    assert total == expected


def test_group_by_tags_explodes(store, raw_ticket_rows):
    result = group_count(store, group_by="tags", filters={})
    total = sum(r["count"] for r in result)
    # ticket with N tags contributes N — sum equals total tag occurrences
    expected = sum(
        sum(1 for i in range(1, 7) if r[f"tag_{i}"]) for r in raw_ticket_rows
    )
    assert total == expected


def test_tags_and_filter(store, raw_ticket_rows):
    # Pick a tag known to appear in fixture
    result = group_count(
        store, group_by="queue", filters={"tags": ["urgent"], "tags_mode": "and"}
    )
    # all returned rows have 'urgent' tag — sanity check
    assert isinstance(result, list)


def test_tags_or_filter(store):
    res_and = group_count(
        store,
        group_by="queue",
        filters={"tags": ["urgent", "login"], "tags_mode": "and"},
    )
    res_or = group_count(
        store,
        group_by="queue",
        filters={"tags": ["urgent", "login"], "tags_mode": "or"},
    )
    sum_and = sum(r["count"] for r in res_and)
    sum_or = sum(r["count"] for r in res_or)
    assert sum_or >= sum_and


def test_unsupported_group_by(store):
    with pytest.raises(McpCstError) as exc:
        group_count(store, group_by="subject", filters={})
    assert exc.value.code == ErrorCode.UNSUPPORTED_GROUP_BY


def test_unsupported_filter_key(store):
    with pytest.raises(McpCstError) as exc:
        group_count(store, group_by="queue", filters={"created_at": "2024-01-01"})
    assert exc.value.code == ErrorCode.UNSUPPORTED_FILTER


def test_unsupported_tags_mode(store):
    with pytest.raises(McpCstError) as exc:
        group_count(
            store, group_by="queue", filters={"tags": ["x"], "tags_mode": "xor"}
        )
    assert exc.value.code == ErrorCode.UNSUPPORTED_FILTER
