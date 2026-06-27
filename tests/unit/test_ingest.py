import numpy as np
import pytest
from mcp_cst.data.ingest import build_store_from_rows
from mcp_cst.data.store import TicketStore


def fake_embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 384), dtype=np.float32)


def test_build_store_from_rows(pg_dsn, pg_schema, raw_ticket_rows):
    store = build_store_from_rows(
        rows=raw_ticket_rows,
        dsn=pg_dsn,
        schema=pg_schema,
        revision="rev1",
        embedder=fake_embed,
    )
    assert isinstance(store, TicketStore)
    assert store.row_count() == len(raw_ticket_rows)
    store.close()


def test_progress_callback_called(pg_dsn, pg_schema, raw_ticket_rows):
    seen = []

    def progress(done: int, total: int) -> None:
        seen.append((done, total))

    store = build_store_from_rows(
        rows=raw_ticket_rows,
        dsn=pg_dsn,
        schema=pg_schema,
        revision="rev1",
        embedder=fake_embed,
        on_progress=progress,
    )
    assert len(seen) > 0
    assert seen[-1][0] == seen[-1][1]  # finished
    store.close()


def test_build_store_tolerates_mixed_type_string_columns(pg_dsn, pg_schema):
    # Regression: the HF dataset ships multiple CSV shards, and one of them
    # has a numeric `version` cell while others are blank. Force-stringify in
    # store ingest fixed it; this guards the regression.
    rows = [
        {"subject": "a", "body": "b", "version": 1, "language": "en"},
        {"subject": "c", "body": "d", "version": "", "language": "en"},
    ]
    store = build_store_from_rows(
        rows=rows,
        dsn=pg_dsn,
        schema=pg_schema,
        revision="rev1",
        embedder=fake_embed,
    )
    assert store.row_count() == 2
    store.close()
