"""Shared fixtures for the RAG eval tier.

The CI tier uses a hash-based deterministic embedder over the 200-row synthetic
fixture. BM25 carries the signal; the vector branch is noise. This is enough
to catch wiring regressions (filter pushdown, pagination, language honoring)
and to produce ragas-vocabulary numbers in `tests/eval/test_known_item.py`.

Semantic quality is the slow tier's job:
`tests/integration/test_rag_full_eval.py` runs the real `multilingual-e5-small`
embedder over a 2k stratified subsample of the live Hugging Face dataset,
gated by `MCP_CST_EVAL_FULL=1`.
"""

from __future__ import annotations
import hashlib
from typing import Callable

import numpy as np
import pytest

from mcp_cst.data.store import TicketStore


def deterministic_embed(texts: list[str]) -> np.ndarray:
    """Hash-based 384-d vectors. Same inputs → same outputs across Python runs
    because we use sha1, not Python's salted `hash`.

    L2-normalised so dot == cosine (mirrors the real embedder contract;
    `draft_reply.select_grounding` assumes this when comparing to the 0.70
    similarity threshold)."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, t in enumerate(texts):
        h = int(hashlib.sha1(t.lower().encode("utf-8")).hexdigest(), 16)
        for j in range(384):
            out[i, j] = ((h >> (j % 256)) & 0xFF) / 255.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


@pytest.fixture(scope="session")
def eval_store_session(raw_ticket_rows, tmp_path_factory):
    """Session-scoped store, built once across all eval tests. Revision is
    hardcoded to "eval" — known-item tests pass this to `derive_id` to compute
    gold ids without coupling to the store's internal id scheme."""
    path = tmp_path_factory.mktemp("eval-store")
    return TicketStore.create(
        path=path / "s",
        revision="eval",
        rows=raw_ticket_rows,
        embedder=deterministic_embed,
    )


@pytest.fixture
def eval_embedder() -> Callable[[list[str]], np.ndarray]:
    """Query-side embedder. Same as the passage-side embedder because the
    deterministic hash has no passage/query split."""
    return deterministic_embed
