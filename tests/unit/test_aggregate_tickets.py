import numpy as np
import pytest
from mcp_cst.tools.aggregate_tickets import aggregate_tickets_impl
from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s",
        revision="r",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )


def test_group_by_language(store, raw_ticket_rows):
    result = aggregate_tickets_impl(store, group_by="language")
    by_lang = {r["group"]: r["count"] for r in result}
    assert by_lang["en"] == sum(1 for r in raw_ticket_rows if r["language"] == "en")


def test_with_filter(store, raw_ticket_rows):
    result = aggregate_tickets_impl(store, group_by="queue", language="de")
    total = sum(r["count"] for r in result)
    assert total == sum(1 for r in raw_ticket_rows if r["language"] == "de")


def test_unsupported_group_by(store):
    with pytest.raises(McpCstError) as exc:
        aggregate_tickets_impl(store, group_by="subject")
    assert exc.value.code == ErrorCode.UNSUPPORTED_GROUP_BY


def test_unsupported_tags_mode(store):
    with pytest.raises(McpCstError) as exc:
        aggregate_tickets_impl(store, group_by="queue", tags=["x"], tags_mode="xor")
    assert exc.value.code == ErrorCode.UNSUPPORTED_FILTER
