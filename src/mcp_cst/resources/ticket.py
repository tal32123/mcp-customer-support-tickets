"""ticket://{id} resource — citation handle returning wrapped ticket text."""

from __future__ import annotations

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import wrap_ticket


DESCRIPTION = make_description(
    summary="Verbatim content of one ticket, addressed by id (12-char hex for HF dataset rows or `usr_<uuidv7>` for user-created tickets).",
    use_for="Use this for: attaching a specific ticket to the chat as a citation, referencing a ticket whose id you already know.",
    not_for="Do NOT use this for: searching or aggregation — those have dedicated tools.",
    output="Output: the ticket wrapped in <ticket> tags with subject, body, and key metadata as child elements.",
    include_g4=True,
)


def ticket_resource_body(store: TicketStore, ticket_id: str) -> str:
    rec = store.get(ticket_id)
    if rec is None:
        raise McpCstError(
            ErrorCode.TICKET_NOT_FOUND, f"no ticket with id {ticket_id!r}"
        )
    return wrap_ticket(
        ticket_id=rec.id,
        subject=rec.subject,
        body=rec.body,
        extra={"language": rec.language, "queue": rec.queue, "priority": rec.priority},
    )
