"""create_ticket tool — insert a new ticket into the running store."""

from __future__ import annotations
import re
from typing import Callable

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import looks_like_injection


_ALLOWED_TYPES = {"question", "incident", "request", "problem"}
_ALLOWED_PRIORITIES = {"low", "medium", "high", "critical", "info"}
_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$|^$")


def _validate_enums(*, type: str, priority: str, version: str) -> None:
    if type and type not in _ALLOWED_TYPES:
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            f"type must be one of {sorted(_ALLOWED_TYPES)} or empty, got {type!r}",
        )
    if priority and priority not in _ALLOWED_PRIORITIES:
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            f"priority must be one of {sorted(_ALLOWED_PRIORITIES)} or empty, got {priority!r}",
        )
    if not _VERSION_RE.match(version):
        raise McpCstError(
            ErrorCode.INVALID_INPUT,
            f"version must match N.N or N.N.N, got {version!r}",
        )


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
    if (
        looks_like_injection(subject)
        or looks_like_injection(body)
        or looks_like_injection(answer)
    ):
        raise McpCstError(
            ErrorCode.INJECTION_DETECTED,
            "input contains injection-shaped patterns; refusing",
        )
    _validate_enums(type=type, priority=priority, version=version)
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
