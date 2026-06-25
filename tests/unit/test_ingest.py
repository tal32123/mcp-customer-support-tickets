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
