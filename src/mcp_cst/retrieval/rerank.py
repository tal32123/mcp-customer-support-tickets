"""Cross-encoder reranking. DEFERRED: returns hits unchanged for now.

When implemented:
- Load BAAI/bge-reranker-base on first call (lazy import).
- Score each (query, hit.body) pair, sort hits by score descending.
- Cache the model on disk via sentence-transformers' default cache.

Until then, this module exists so the call site can stay stable.
"""

from __future__ import annotations


def maybe_rerank(*, query: str, hits: list[dict], enabled: bool) -> list[dict]:
    """No-op when disabled (and currently no-op when enabled).

    Returning hits unchanged keeps the surface stable; the only behavioural
    difference of enabling rerank today is the log line below.
    """
    if not enabled:
        return hits
    # TODO: load BAAI/bge-reranker-base and re-score hits.
    return hits
