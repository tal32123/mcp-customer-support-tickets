"""Cross-encoder reranking. DEFERRED: returns hits unchanged for now.

When implemented:
- Load BAAI/bge-reranker-base on first call (lazy import).
- Score each (query, hit.body) pair, sort hits by score descending.
- Cache the model on disk via sentence-transformers' default cache.

Until then, this module exists so the call site can stay stable.
"""

from __future__ import annotations
import logging


log = logging.getLogger(__name__)
_WARNED = False


def maybe_rerank(*, query: str, hits: list[dict], enabled: bool) -> list[dict]:
    """No-op when disabled (and currently no-op when enabled either).

    When `enabled=True` we emit a one-time warning so an operator who set
    `RERANK=true` is not silently surprised that nothing changed.
    """
    global _WARNED
    if not enabled:
        return hits
    if not _WARNED:
        log.warning(
            "RERANK=true was set but the cross-encoder reranker is not yet "
            "implemented; returning hits unchanged. Track this in the spec.",
        )
        _WARNED = True
    return hits
