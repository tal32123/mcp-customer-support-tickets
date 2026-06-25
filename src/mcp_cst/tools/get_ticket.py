"""get_ticket tool — verbatim row fetch with <ticket> wrapping."""

from __future__ import annotations

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import wrap_ticket


DESCRIPTION = make_description(
    summary="Fetch one ticket by id and return every column, verbatim.",
    use_for="Use this for: any time the user identifies a ticket by id (e.g. 'show me ticket abc123', 'what's in ticket xyz789'), inspecting a ticket before drafting a reply.",
    not_for="Do NOT use this for: finding tickets by topic (use search_tickets), counting (use aggregate_tickets).",
    output="Output: JSON with every dataset column, a normalized `tags` list, and a `wrapped` field containing the ticket inside <ticket> tags.",
    include_g4=True,
)


def get_ticket_impl(store: TicketStore, ticket_id: str) -> dict:
    rec = store.get(ticket_id)
    if rec is None:
        raise McpCstError(
            ErrorCode.TICKET_NOT_FOUND, f"no ticket with id {ticket_id!r}"
        )
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
