"""Build the Postgres store from Hugging Face Parquet or in-memory rows."""

from __future__ import annotations
from typing import Callable

import numpy as np

from .store import TicketStore


ProgressFn = Callable[[int, int], None]

_BATCH_SIZE = 256


def build_store_from_rows(
    *,
    rows: list[dict],
    dsn: str,
    schema: str,
    revision: str,
    embedder: Callable[[list[str]], np.ndarray],
    embedding_dim: int = 384,
    on_progress: ProgressFn | None = None,
    batch_size: int = _BATCH_SIZE,
) -> TicketStore:
    """Build a fresh store from a list of dict rows.

    Wraps the embedder so we can fire ``on_progress`` per batch without
    pushing that concern into the store layer.
    """
    total = len(rows)
    done = [0]

    def progress_embedder(texts: list[str]) -> np.ndarray:
        vec = embedder(texts)
        done[0] += len(texts)
        if on_progress is not None:
            on_progress(done[0], total)
        return vec

    return TicketStore.create_with_rows(
        dsn=dsn,
        schema=schema,
        revision=revision,
        rows=rows,
        embedder=progress_embedder,
        embedding_dim=embedding_dim,
    )


def build_store_from_huggingface(
    *,
    dsn: str,
    schema: str,
    dataset_id: str,
    revision: str,
    embedder: Callable[[list[str]], np.ndarray],
    embedding_dim: int = 384,
    on_progress: ProgressFn | None = None,
    batch_size: int = _BATCH_SIZE,
) -> TicketStore:
    """Download the HF dataset at ``revision`` and build the store.

    Used at server startup if no cached store exists.
    """
    from datasets import load_dataset  # local import: heavy

    ds = load_dataset(dataset_id, revision=revision, split="train")
    rows = [dict(r) for r in ds]
    return build_store_from_rows(
        rows=rows,
        dsn=dsn,
        schema=schema,
        revision=revision,
        embedder=embedder,
        embedding_dim=embedding_dim,
        on_progress=on_progress,
        batch_size=batch_size,
    )
