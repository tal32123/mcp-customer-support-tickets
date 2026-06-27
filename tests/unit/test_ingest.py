import numpy as np
from mcp_cst.data.ingest import build_store_from_rows
from mcp_cst.data.store import TicketStore


def fake_embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 384), dtype=np.float32)


def test_build_store_from_rows(tmp_path, raw_ticket_rows):
    store = build_store_from_rows(
        rows=raw_ticket_rows,
        path=tmp_path / "store",
        revision="rev1",
        embedder=fake_embed,
    )
    assert isinstance(store, TicketStore)
    assert store.row_count() == len(raw_ticket_rows)


def test_progress_callback_called(tmp_path, raw_ticket_rows):
    seen = []

    def progress(done: int, total: int) -> None:
        seen.append((done, total))

    build_store_from_rows(
        rows=raw_ticket_rows,
        path=tmp_path / "store",
        revision="rev1",
        embedder=fake_embed,
        on_progress=progress,
    )
    assert len(seen) > 0
    assert seen[-1][0] == seen[-1][1]  # finished


def test_build_store_tolerates_mixed_type_string_columns(tmp_path):
    # Regression: the HF dataset ships multiple CSV shards, and one of them
    # has a numeric `version` cell while others are blank. LanceDB infers
    # pyarrow types from the records (ignoring the schema we pass), so the
    # first numeric value pinned the column to int64 and the next "" raised
    # `ArrowInvalid: Could not convert '' with type str`. Force-stringify in
    # store._create fixed it; this guards the regression.
    rows = [
        {"subject": "a", "body": "b", "version": 1, "language": "en"},
        {"subject": "c", "body": "d", "version": "", "language": "en"},
    ]
    store = build_store_from_rows(
        rows=rows,
        path=tmp_path / "store",
        revision="rev1",
        embedder=fake_embed,
    )
    assert store.row_count() == 2
