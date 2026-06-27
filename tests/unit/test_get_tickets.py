import numpy as np
import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError
from mcp_cst.tools.get_tickets import HARD_CAP, get_tickets_impl


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


def test_returns_in_order_with_full_shape(store):
    ids = store.all_ids()[:3]
    out = get_tickets_impl(store, ids)
    assert [r["id"] for r in out] == ids
    for row in out:
        assert "subject" in row
        assert "body" in row
        assert "answer" in row
        assert row["wrapped"].startswith(f'<ticket id="{row["id"]}">')
        assert isinstance(row["tags"], list)
        for i in range(1, 7):
            assert f"tag_{i}" in row


def test_unknown_ids_become_none(store):
    real = store.all_ids()[0]
    out = get_tickets_impl(store, [real, "doesnotexist", real])
    assert out[0] is not None and out[0]["id"] == real
    assert out[1] is None
    assert out[2] is not None and out[2]["id"] == real


def test_empty_list_returns_empty_list(store):
    # Empty list is a valid no-op; cheaper than refusing.
    assert get_tickets_impl(store, []) == []


def test_over_cap_rejected(store):
    too_many = ["x"] * (HARD_CAP + 1)
    with pytest.raises(McpCstError) as exc:
        get_tickets_impl(store, too_many)
    assert exc.value.code == ErrorCode.INVALID_INPUT
