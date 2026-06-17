"""create_ticket tool — insert a new ticket into the running store."""

from __future__ import annotations
from typing import Callable

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import looks_like_injection


DESCRIPTION = make_description(
    summary="Insert a new ticket into the running store and return its 12-char id.",
    use_for=(
        "Use this for: 'add a ticket about X', 'log this as a new ticket', "
        "registering one-off tickets the client wants to make searchable. "
        "Subject and body must be already-composed text — this server does not generate ticket content."
    ),
    not_for=(
        "Do NOT use this for: editing an existing ticket (unsupported), bulk import "
        "(call once per ticket), generating ticket text from a prompt (no LLM is wired "
        "up on the server), or backfilling timestamps (the dataset has no time column)."
    ),
    output='Output: JSON {"id": "<12-char hex>"}. Use get_ticket or ticket://{id} to read it back.',
    include_g4=False,
)


def create_ticket_impl(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    *,
    subject: str,
    body: str,
    answer: str = "",
    type: str = "",
    queue: str = "",
    priority: str = "",
    language: str = "",
    version: str = "",
    tags: list[str] | None = None,
) -> dict:
    if not subject.strip() or not body.strip():
        raise McpCstError(ErrorCode.INVALID_INPUT, "subject and body are required")
    if looks_like_injection(subject) or looks_like_injection(body):
        raise McpCstError(
            ErrorCode.INJECTION_DETECTED,
            "input contains injection-shaped patterns; refusing",
        )
    new_id = store.add_ticket(
        subject=subject,
        body=body,
        embedder=embedder,
        answer=answer,
        type=type,
        queue=queue,
        priority=priority,
        language=language,
        version=version,
        tags=tags,
    )
    return {"id": new_id}
