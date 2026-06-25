"""delete_ticket tool — remove one ticket from the running store."""

from __future__ import annotations

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError


DESCRIPTION = make_description(
    summary="Delete one ticket by id. Returns confirmation; raises TICKET_NOT_FOUND if the id is unknown.",
    use_for=(
        "Use this for: 'delete ticket abc123', removing a mistakenly-created ticket. "
        "Confirm with the user before deleting — deletion is irreversible within the running store."
    ),
    not_for=(
        "Do NOT use this for: bulk deletion (call once per ticket), archiving (there is no archive — "
        "delete is destructive), or filtering out tickets at query time (use filters on "
        "search_tickets/aggregate_tickets instead)."
    ),
    output='Output: JSON {"id": "<12-char hex>", "deleted": true} on success.',
    include_g4=False,
)


def delete_ticket_impl(store: TicketStore, ticket_id: str) -> dict:
    if not store.delete_ticket(ticket_id):
        raise McpCstError(
            ErrorCode.TICKET_NOT_FOUND, f"no ticket with id {ticket_id!r}"
        )
    return {"id": ticket_id, "deleted": True}
