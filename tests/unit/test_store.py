import re

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
def store(pg_dsn, pg_schema, raw_ticket_rows):
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="testrev",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    yield s
    s.close()


def test_row_count(store, raw_ticket_rows):
    assert store.row_count() == len(raw_ticket_rows)


def test_ids_are_uuidv7_hex(store, raw_ticket_rows):
    """Primary ids are bare 32-hex UUIDv7. Per-ingest random — no longer
    deterministic across rebuilds; provenance now lives in
    `original_system_id` (covered by test_original_system_id_populated)."""
    ids = store.all_ids()
    assert len(ids) == len(raw_ticket_rows)
    assert len(set(ids)) == len(ids)  # unique
    for i in ids:
        assert len(i) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", i)
        assert i[12] == "7"  # UUIDv7 version nibble


def test_original_system_id_populated(store, raw_ticket_rows):
    """Bulk HF rows carry deterministic `original_system_id`. User-created
    rows leave it blank."""
    from mcp_cst.data.store import derive_id

    ids = store.all_ids()
    # Bulk row: original_system_id matches the deterministic legacy scheme.
    rec0 = store.get(ids[0])
    assert rec0.original_system_id != ""
    assert len(rec0.original_system_id) == 12
    # `all_ids` is sorted by row_index, so ids[0] is row 0.
    assert rec0.original_system_id == derive_id("testrev", 0)

    # User-created row: original_system_id is empty.
    new_id = store.add_ticket(subject="user", body="row", embedder=fake_embed)
    new_rec = store.get(new_id)
    assert new_rec.original_system_id == ""


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


def test_open_existing(store, pg_dsn, raw_ticket_rows):
    reopened = TicketStore.connect(
        dsn=pg_dsn,
        schema=store.schema_name,
        revision="testrev",
    )
    assert reopened.row_count() == len(raw_ticket_rows)
    reopened.close()


def test_get_escapes_quotes_in_id(store):
    """Parameterized SQL means a literal-quote id matches nothing — psycopg
    binds it as a value, not as SQL. Regression guard against any future
    string-formatted WHERE clause."""
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
    # All ids are 32-char UUIDv7 hex (no prefix).
    assert len(new_id) == 32
    assert re.fullmatch(r"[0-9a-f]{32}", new_id)
    assert new_id[12] == "7"
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


def test_ingest_coerces_null_fields(pg_dsn, pg_schema):
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
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="r",
        rows=rows,
        embedder=fake_embed,
    )
    rec = s.get(s.all_ids()[0])
    assert rec.answer == ""
    assert rec.type == ""
    assert rec.version == ""
    s.close()


def test_is_valid_after_delete_all_stays_true(pg_dsn, pg_schema, raw_ticket_rows):
    """Regression: a user who deletes every ticket must not have the rows
    resurrected on next boot. is_valid gates on the ingest_complete marker,
    not on row count."""
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="r",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    for tid in s.all_ids():
        s.delete_ticket(tid)
    assert s.row_count() == 0
    s.close()
    assert (
        TicketStore.is_valid(dsn=pg_dsn, schema=pg_schema, revision="r") is True
    )


def test_is_valid_partial_ingest_returns_false(pg_dsn, pg_schema):
    """Crash before the ingest_complete marker commits → next boot rebuilds."""
    # Simulate partial ingest: schema + tables + schema_version row exist,
    # but no ingest_complete marker.
    store = TicketStore.connect(
        dsn=pg_dsn, schema=pg_schema, revision="r"
    )
    store.close()
    assert (
        TicketStore.is_valid(dsn=pg_dsn, schema=pg_schema, revision="r") is False
    )


def test_null_subject_body_not_poisoning_bm25(pg_dsn, pg_schema):
    """Regression #300: a None subject/body must not write the literal
    string 'None' into the BM25 text — the persisted text_search column
    must contain no spurious 'None' tokens."""
    rows = [
        {
            "subject": None,
            "body": None,
            "answer": "real answer",
            "type": "",
            "queue": "",
            "priority": "",
            "language": "",
            "version": "",
            "tag_1": "",
            "tag_2": "",
            "tag_3": "",
            "tag_4": "",
            "tag_5": "",
            "tag_6": "",
        },
    ]
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="r",
        rows=rows,
        embedder=fake_embed,
    )
    text = s.text_search_of(s.all_ids()[0])
    assert "None" not in text
    s.close()
