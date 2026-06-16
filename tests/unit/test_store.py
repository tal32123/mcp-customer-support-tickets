import numpy as np
import pytest
from mcp_cst.data.store import TicketStore, TicketRecord


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
    s = TicketStore.create(
        path=tmp_path / "store",
        revision="testrev",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    return s


def test_row_count(store, raw_ticket_rows):
    assert store.row_count() == len(raw_ticket_rows)


def test_ids_are_stable(store, raw_ticket_rows):
    ids = store.all_ids()
    assert len(ids) == len(raw_ticket_rows)
    assert len(set(ids)) == len(ids)  # unique
    assert all(len(i) == 12 for i in ids)
    # rebuild with same inputs → same ids
    store2 = TicketStore.create(
        path=store.path.parent / "store2",
        revision="testrev",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    assert store.all_ids() == store2.all_ids()


def test_get_ticket_verbatim(store, raw_ticket_rows):
    first_id = store.all_ids()[0]
    rec = store.get(first_id)
    assert rec is not None
    assert rec.subject == raw_ticket_rows[0]["subject"]
    assert rec.body == raw_ticket_rows[0]["body"]
    # original tag_1..tag_6 preserved
    for i in range(1, 7):
        assert getattr(rec, f"tag_{i}") == raw_ticket_rows[0][f"tag_{i}"]
    # normalized tags list: drops empties
    expected_tags = [t for t in (raw_ticket_rows[0][f"tag_{i}"] for i in range(1, 7)) if t]
    assert rec.tags == expected_tags


def test_get_missing_returns_none(store):
    assert store.get("nonexistent00") is None


def test_open_existing(store, raw_ticket_rows, tmp_path):
    reopened = TicketStore.open(path=store.path, revision="testrev")
    assert reopened.row_count() == len(raw_ticket_rows)
