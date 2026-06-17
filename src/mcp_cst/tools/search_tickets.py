"""search_tickets tool — hybrid retrieval entry point."""

from __future__ import annotations
from typing import Callable, Literal

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..retrieval.hybrid import hybrid_search


HARD_CAP = 50

DESCRIPTION = make_description(
    summary="Find tickets matching a free-text query using hybrid BM25 + vector retrieval. Returns up to `limit` previews.",
    use_for=(
        "Use this for: 'find tickets about login problems', 'tickets mentioning error 500', "
        "'tickets similar to: app crashes on startup', narrowing by language/queue/priority/type/tags."
    ),
    not_for=(
        "Do NOT use this for: counting tickets (use aggregate_tickets), fetching a specific ticket id "
        "(use get_ticket), date-range filtering (the dataset has no timestamps; will be refused)."
    ),
    output=(
        "Output: list of {id, subject, snippet (<=240 chars of body), language, queue, priority, ticket_uri}. "
        "The ticket_uri is the citation handle suitable for attaching to chat."
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
    language: Literal["en", "de"] | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    tags_mode: Literal["and", "or"] = "and",
    limit: int = 10,
) -> list[dict]:
    filters: dict = {}
    if queue is not None: filters["queue"] = queue
    if priority is not None: filters["priority"] = priority
    if language is not None: filters["language"] = language
    if type is not None: filters["type"] = type
    if tags: filters["tags"] = tags
    filters["tags_mode"] = tags_mode

    capped = max(1, min(limit, HARD_CAP))
    hits = hybrid_search(store, query=q, filters=filters, embedder=embedder, limit=capped)
    for h in hits:
        h["ticket_uri"] = f"ticket://{h['id']}"
    return hits
