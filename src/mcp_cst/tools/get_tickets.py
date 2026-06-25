"""get_tickets tool — batch fetch to collapse N+1 round-trips."""

from __future__ import annotations

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import wrap_ticket


HARD_CAP = 50

DESCRIPTION = make_description(
    summary="Fetch many tickets by id in one call. Preserves input order; unknown ids become null.",
    use_for=(
        "Use this for: hydrating a batch of ids from search_tickets in one round-trip, "
        "or any workflow where you already have several ticket ids."
    ),
    not_for=(
        "Do NOT use this for: a single id (use get_ticket), discovery by topic "
        "(use search_tickets or search_and_fetch)."
    ),
    output=(
        "Output: JSON list, same length and order as `ids`. Each entry is either the "
        "full get_ticket payload (every column, normalized tags, `wrapped`) or null "
        "when no ticket with that id exists."
    ),
    include_g4=True,
)


def _to_dict(rec) -> dict:
    wrapped = wrap_ticket(
        ticket_id=rec.id,
        subject=rec.subject,
        body=rec.body,
        extra={"language": rec.language, "queue": rec.queue, "priority": rec.priority},
    )
    return {
        "id": rec.id,
        "subject": rec.subject,
        "body": rec.body,
        "answer": rec.answer,
        "type": rec.type,
        "queue": rec.queue,
        "priority": rec.priority,
        "language": rec.language,
        "version": rec.version,
        **{f"tag_{i}": getattr(rec, f"tag_{i}") for i in range(1, 7)},
        "tags": rec.tags,
        "wrapped": wrapped,
    }


def get_tickets_impl(store: TicketStore, ids: list[str]) -> list[dict | None]:
    if len(ids) > HARD_CAP:
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            f"too many ids: {len(ids)} > hard cap {HARD_CAP}",
        )
    # ponytail: per-id store.get loop, swap for a single id-IN-list query if profiling shows it.
    out: list[dict | None] = []
    for tid in ids:
        rec = store.get(tid)
        out.append(_to_dict(rec) if rec is not None else None)
    return out
