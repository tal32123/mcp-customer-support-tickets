"""search_tickets tool — hybrid retrieval with cursor pagination."""

from __future__ import annotations
from typing import Callable, Literal

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..retrieval.hybrid import build_filters, hybrid_search_full, hydrate_ids
from ..retrieval.search_cache import (
    cache_get,
    cache_put,
    compute_search_id,
    decode_cursor,
    encode_cursor,
)


HARD_CAP = 50

DESCRIPTION = make_description(
    summary="Find tickets matching a free-text query using hybrid BM25 + vector retrieval. Returns up to `limit` previews per page, plus a `next_cursor` for paging deeper.",
    use_for=(
        "Use this for: 'find tickets about login problems', 'tickets mentioning error 500', "
        "'tickets similar to: app crashes on startup', narrowing by language/queue/priority/type/tags. "
        "For long sweeps, pass the returned `next_cursor` back to fetch the next page."
    ),
    not_for=(
        "Do NOT use this for: counting tickets (use aggregate_tickets), fetching a specific ticket id "
        "(use get_ticket), date-range filtering (the dataset has no timestamps; will be refused)."
    ),
    output=(
        "Output: {hits: [{id, subject, snippet (<=240 chars), language, queue, priority, score_rank, "
        "ticket_uri}], next_cursor: str|None, search_id: str, total_estimate: int}. "
        "`next_cursor` is None on the last page. `search_id` is stable across calls with the same q+filters."
    ),
    include_g4=True,
)


def search_tickets_impl(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    *,
    q: str,
    queue: str | None = None,
    priority: str | None = None,
    language: Literal["en", "de", "he"] | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    tags_mode: Literal["and", "or"] = "and",
    limit: int = 10,
    cursor: str | None = None,
) -> dict:
    """Search with optional cursor pagination.

    First call: cursor=None → run hybrid fusion, cache the full id list,
    return page 1 + a `next_cursor` pointing at the next slice. Subsequent
    calls: pass the cursor back to read the next slice from the cache
    without re-fusing. Stale/garbage cursors raise INVALID_INPUT.
    """
    if not q.strip() or len(q.strip()) < 2:
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            "query must be at least 2 non-whitespace characters",
        )
    filters = build_filters(
        queue=queue,
        priority=priority,
        language=language,
        type=type,
        tags=tags,
        tags_mode=tags_mode,
    )
    capped = max(1, min(limit, HARD_CAP))

    if cursor is None:
        search_id = compute_search_id(q, dict(filters))
        fused_ids = hybrid_search_full(
            store, query=q, filters=filters, embedder=embedder
        )
        cache_put(search_id, fused_ids)
        offset = 0
    else:
        search_id, offset = decode_cursor(cursor)
        fused_ids = cache_get(search_id)
        if fused_ids is None:
            # Either evicted, expired, or just garbage. Same response either
            # way — caller must restart the search.
            raise McpCstError(
                ErrorCode.INVALID_INPUT,
                "cursor is stale or unknown; rerun search_tickets without cursor",
            )

    page_ids = fused_ids[offset : offset + capped]
    hits = hydrate_ids(store, page_ids, rank_offset=offset)
    for h in hits:
        h["ticket_uri"] = f"ticket://{h['id']}"

    next_offset = offset + capped
    next_cursor = encode_cursor(search_id, next_offset) if next_offset < len(fused_ids) else None

    return {
        "hits": hits,
        "next_cursor": next_cursor,
        "search_id": search_id,
        "total_estimate": len(fused_ids),
    }
