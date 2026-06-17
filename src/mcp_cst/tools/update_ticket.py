"""update_ticket tool — patch one ticket by id, re-embedding on text changes."""

from __future__ import annotations
from typing import Callable

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import looks_like_injection


DESCRIPTION = make_description(
    summary="Patch one ticket by id. Unspecified fields keep their current value. Returns the updated ticket.",
    use_for=(
        "Use this for: 'change ticket abc123's priority to high', 'update the body of ticket xyz789', "
        "correcting metadata on an existing ticket. Pass only the fields you want to change."
    ),
    not_for=(
        "Do NOT use this for: creating a new ticket (use create_ticket), generating new ticket text "
        "from a prompt (no LLM is wired up on the server), or replacing a ticket's id (ids are immutable)."
    ),
    output='Output: JSON {"id": "<12-char hex>", "updated": true} on success. If the id is not found, an error with code TICKET_NOT_FOUND is raised.',
    include_g4=False,
)


def update_ticket_impl(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    *,
    ticket_id: str,
    subject: str | None = None,
    body: str | None = None,
    answer: str | None = None,
    type: str | None = None,
    queue: str | None = None,
    priority: str | None = None,
    language: str | None = None,
    version: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    if subject is not None and not subject.strip():
        raise McpCstError(ErrorCode.INVALID_INPUT, "subject cannot be blanked")
    if body is not None and not body.strip():
        raise McpCstError(ErrorCode.INVALID_INPUT, "body cannot be blanked")
    if subject is not None and looks_like_injection(subject):
        raise McpCstError(
            ErrorCode.INJECTION_DETECTED,
            "input contains injection-shaped patterns; refusing",
        )
    if body is not None and looks_like_injection(body):
        raise McpCstError(
            ErrorCode.INJECTION_DETECTED,
            "input contains injection-shaped patterns; refusing",
        )
    ok = store.update_ticket(
        ticket_id=ticket_id,
        embedder=embedder,
        subject=subject,
        body=body,
        answer=answer,
        type=type,
        queue=queue,
        priority=priority,
        language=language,
        version=version,
        tags=tags,
    )
    if not ok:
        raise McpCstError(ErrorCode.TICKET_NOT_FOUND, f"no ticket with id {ticket_id!r}")
    return {"id": ticket_id, "updated": True}
