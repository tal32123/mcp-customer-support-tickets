import numpy as np
import pytest
from mcp_cst.data.store import TicketStore
from mcp_cst.errors import ErrorCode, McpCstError
from mcp_cst.tools.create_ticket import create_ticket_impl
from mcp_cst.tools.delete_ticket import delete_ticket_impl


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic 384-dim 'embedding' for tests — no model download."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        h = abs(hash(t))
        for j in range(384):
            out[i, j] = ((h >> (j % 32)) & 0xFF) / 255.0
    return out


@pytest.fixture
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s", revision="r", rows=raw_ticket_rows, embedder=fake_embed,
    )


def test_delete_removes_ticket(store):
    tid = create_ticket_impl(
        store, fake_embed,
        subject="To be deleted", body="This ticket will be removed.",
    )["id"]
    assert store.get(tid) is not None
    out = delete_ticket_impl(store, tid)
    assert out == {"id": tid, "deleted": True}
    assert store.get(tid) is None


def test_delete_unknown_id_raises_not_found(store):
    with pytest.raises(McpCstError) as exc:
        delete_ticket_impl(store, "deadbeef0000")
    assert exc.value.code == ErrorCode.TICKET_NOT_FOUND
