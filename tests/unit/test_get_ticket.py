import numpy as np
import pytest
from mcp_cst.tools.get_ticket import get_ticket_impl
from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def store(pg_dsn, pg_schema, raw_ticket_rows):
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="r",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    yield s
    s.close()


def test_returns_wrapped_ticket(store):
    first_id = store.all_ids()[0]
    out = get_ticket_impl(store, first_id)
    assert out["id"] == first_id
    assert out["wrapped"].startswith(f'<ticket id="{first_id}">')
    # Verbatim fields exposed
    assert "subject" in out
    assert "body" in out
    assert "answer" in out
    assert isinstance(out["tags"], list)
    # tag_1..tag_6 preserved
    for i in range(1, 7):
        assert f"tag_{i}" in out


def test_unknown_id_raises(store):
    with pytest.raises(McpCstError) as exc:
        get_ticket_impl(store, "doesnotexist")
    assert exc.value.code == ErrorCode.TICKET_NOT_FOUND
