"""Build the LanceDB store from Hugging Face Parquet or in-memory rows."""

from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np

from .store import TicketStore


ProgressFn = Callable[[int, int], None]

_BATCH_SIZE = 64


def build_store_from_rows(
    *,
    rows: list[dict],
    path: Path,
    revision: str,
    embedder: Callable[[list[str]], np.ndarray],
    on_progress: ProgressFn | None = None,
) -> TicketStore:
    """Build a fresh store from a list of dict rows.

    The embedder is called in batches so progress can be reported.
    """
    total = len(rows)
    done = [0]

    def batched(texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            chunk = embedder(texts[i : i + _BATCH_SIZE])
            chunks.append(chunk)
            done[0] += chunk.shape[0]
            if on_progress is not None:
                on_progress(done[0], total)
        return np.vstack(chunks) if chunks else np.zeros((0, 384), dtype=np.float32)

    return TicketStore.create(
        path=path,
        revision=revision,
        rows=rows,
        embedder=batched,
    )


def build_store_from_huggingface(
    *,
    path: Path,
    dataset_id: str,
    revision: str,
    embedder: Callable[[list[str]], np.ndarray],
    on_progress: ProgressFn | None = None,
) -> TicketStore:
    """Download the HF dataset at `revision` and build the store.

    Not unit-tested here; verified manually during integration. Used at
    server startup if no cached store exists.
    """
    from datasets import load_dataset  # local import: heavy

    ds = load_dataset(dataset_id, revision=revision, split="train")
    rows = [dict(r) for r in ds]
    return build_store_from_rows(
        rows=rows,
        path=path,
        revision=revision,
        embedder=embedder,
        on_progress=on_progress,
    )
