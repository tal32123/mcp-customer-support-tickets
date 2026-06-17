"""aggregate_tickets tool — group-by counts with filters."""

from __future__ import annotations
from typing import Literal

from ..data.aggregates import group_count
from ..data.store import TicketStore
from ..docs import make_description


DESCRIPTION = make_description(
    summary="Count tickets grouped by queue, priority, language, type, or tags. Same filter args as search_tickets.",
    use_for=(
        "Use this for: 'how many tickets per queue?', 'how many German billing tickets?', "
        "'most common priorities for type=incident', any 'count' or 'distribution' question."
    ),
    not_for=(
        "Do NOT use this for: returning ticket content (use search_tickets), fetching one ticket "
        "(use get_ticket), date filters (refused — no timestamp column)."
    ),
    output="Output: list of {group: str, count: int}, sorted by count descending.",
    include_g4=False,
)


def aggregate_tickets_impl(
    store: TicketStore,
    *,
    group_by: str,
    queue: str | None = None,
    priority: str | None = None,
    language: Literal["en", "de", "he"] | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    tags_mode: Literal["and", "or"] = "and",
) -> list[dict]:
    filters: dict = {"tags_mode": tags_mode}
    if queue is not None: filters["queue"] = queue
    if priority is not None: filters["priority"] = priority
    if language is not None: filters["language"] = language
    if type is not None: filters["type"] = type
    if tags: filters["tags"] = tags
    return group_count(store, group_by=group_by, filters=filters)
