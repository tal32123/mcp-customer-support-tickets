import numpy as np
import pytest
from mcp_cst.data.store import TicketStore


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
    expected_tags = [
        t for t in (raw_ticket_rows[0][f"tag_{i}"] for i in range(1, 7)) if t
    ]
    assert rec.tags == expected_tags


def test_get_missing_returns_none(store):
    assert store.get("nonexistent00") is None


def test_open_existing(store, raw_ticket_rows, tmp_path):
    reopened = TicketStore.open(path=store.path, revision="testrev")
    assert reopened.row_count() == len(raw_ticket_rows)


def test_get_escapes_quotes_in_id(store):
    """H1: a malicious id with a single quote must not corrupt the WHERE clause."""
    # Without escaping, this would expand to:  WHERE id = 'x' OR '1'='1'
    # and return an arbitrary row. With escaping, no rows match.
    assert store.get("x' OR '1'='1") is None
    assert store.get("'; DROP TABLE tickets; --") is None


def test_add_ticket_round_trip(store):
    new_id = store.add_ticket(
        subject="Fresh subject",
        body="Fresh body content",
        embedder=fake_embed,
        queue="Technical Support",
        priority="high",
        language="en",
        type="incident",
        tags=["Display", "Hardware"],
    )
    assert len(new_id) == 12
    assert all(c in "0123456789abcdef" for c in new_id)
    rec = store.get(new_id)
    assert rec is not None
    assert rec.subject == "Fresh subject"
    assert rec.body == "Fresh body content"
    assert rec.queue == "Technical Support"
    assert rec.tags == ["Display", "Hardware"]
    assert rec.tag_1 == "Display"
    assert rec.tag_2 == "Hardware"
    assert rec.tag_3 == ""


def test_add_ticket_increments_row_count(store, raw_ticket_rows):
    before = store.row_count()
    store.add_ticket(subject="s", body="b", embedder=fake_embed)
    assert store.row_count() == before + 1


def test_add_ticket_unique_ids(store):
    a = store.add_ticket(subject="alpha", body="one", embedder=fake_embed)
    b = store.add_ticket(subject="beta", body="two", embedder=fake_embed)
    assert a != b


def test_update_ticket_changes_fields(store):
    new_id = store.add_ticket(
        subject="Initial",
        body="Initial body",
        embedder=fake_embed,
        queue="Customer Service",
        priority="low",
    )
    changed = store.update_ticket(
        new_id,
        embedder=fake_embed,
        subject="Updated subject",
        priority="high",
    )
    assert changed is True
    rec = store.get(new_id)
    assert rec.subject == "Updated subject"
    assert rec.body == "Initial body"  # untouched fields preserved
    assert rec.queue == "Customer Service"
    assert rec.priority == "high"


def test_update_ticket_unknown_id_returns_false(store):
    assert (
        store.update_ticket("nonexistent0", embedder=fake_embed, subject="x") is False
    )


def test_update_ticket_replaces_tags(store):
    new_id = store.add_ticket(
        subject="s",
        body="b",
        embedder=fake_embed,
        tags=["A", "B", "C"],
    )
    store.update_ticket(new_id, embedder=fake_embed, tags=["X"])
    rec = store.get(new_id)
    assert rec.tags == ["X"]
    assert rec.tag_1 == "X"
    assert rec.tag_2 == ""


def test_delete_ticket_removes_row(store):
    new_id = store.add_ticket(subject="bye", body="bye", embedder=fake_embed)
    before = store.row_count()
    assert store.delete_ticket(new_id) is True
    assert store.row_count() == before - 1
    assert store.get(new_id) is None


def test_delete_ticket_unknown_id_returns_false(store):
    before = store.row_count()
    assert store.delete_ticket("nonexistent0") is False
    assert store.row_count() == before


def test_ingest_coerces_null_fields(tmp_path):
    """Regression: HF rows with None values must not poison the store."""
    rows = [
        {
            "subject": "ok",
            "body": "ok",
            "answer": None,
            "type": None,
            "queue": "Q",
            "priority": "low",
            "language": "en",
            "version": None,
            "tag_1": "",
            "tag_2": "",
            "tag_3": "",
            "tag_4": "",
            "tag_5": "",
            "tag_6": "",
        },
    ]
    s = TicketStore.create(
        path=tmp_path / "null-store",
        revision="r",
        rows=rows,
        embedder=fake_embed,
    )
    rec = s.get(s.all_ids()[0])
    assert rec.answer == ""
    assert rec.type == ""
    assert rec.version == ""
