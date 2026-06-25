"""search_and_fetch tool — hybrid search + full-row hydration in one call."""

from __future__ import annotations
from typing import Callable, Literal

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..retrieval.hybrid import build_filters, hybrid_search
from .get_tickets import _to_dict


HARD_CAP = 50

DESCRIPTION = make_description(
    summary=(
        "Search by free-text query and return the FULL ticket rows for the top hits in one call "
        "(no follow-up get_ticket needed)."
    ),
    use_for=(
        "Use this for: 'show me the actual content of tickets about login problems', any time "
        "the agent needs bodies/answers + citations without an N+1 fetch loop."
    ),
    not_for=(
        "Do NOT use this for: previews only (use search_tickets — cheaper), counting "
        "(use aggregate_tickets), or single known id (use get_ticket)."
    ),
    output=(
        "Output: list of full ticket dicts (every column, normalized tags, `wrapped`, plus "
        "`ticket_uri` citation handle). `include='body'` drops `answer`; `include='answer'` "
        "drops `body`; `include='all'` returns both."
    ),
    include_g4=True,
)


_DROP = {
    "body": ("answer",),
    "answer": ("body",),
    "all": (),
}


def search_and_fetch_impl(
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
    k: int = 10,
    include: Literal["body", "answer", "all"] = "all",
) -> list[dict]:
    if not q.strip() or len(q.strip()) < 2:
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            "query must be at least 2 non-whitespace characters",
        )
    if include not in _DROP:
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            f"include must be 'body', 'answer', or 'all', got {include!r}",
        )
    filters = build_filters(
        queue=queue,
        priority=priority,
        language=language,
        type=type,
        tags=tags,
        tags_mode=tags_mode,
    )
    capped = max(1, min(k, HARD_CAP))
    hits = hybrid_search(store, query=q, filters=filters, embedder=embedder, limit=capped)

    drop_fields = _DROP[include]
    out: list[dict] = []
    for h in hits:
        rec = store.get(h["id"])
        if rec is None:
            continue  # search returned id that vanished; skip rather than poison batch
        row = _to_dict(rec)
        row["ticket_uri"] = f"ticket://{rec.id}"
        for f in drop_fields:
            row.pop(f, None)
        out.append(row)
    return out
