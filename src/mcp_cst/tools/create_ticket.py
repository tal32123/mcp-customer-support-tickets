"""create_ticket tool — insert a new ticket into the running store."""

from __future__ import annotations
import hashlib
import re
import time
from collections import OrderedDict
from typing import Callable

import numpy as np

from ..data.store import TicketStore
from ..docs import make_description
from ..errors import ErrorCode, McpCstError
from ..safety import looks_like_injection


_ALLOWED_TYPES = {"question", "incident", "request", "problem"}
_ALLOWED_PRIORITIES = {"low", "medium", "high", "critical", "info"}
_VERSION_RE = re.compile(r"^\d+\.\d+(\.\d+)?$|^$")

# #162: in-memory idempotency cache. Same (subject|body|tags|answer) within
# the window returns the previously-minted id instead of inserting a dup row.
# ponytail: process-local dict, no schema column, no clock skew handling —
# upgrade to a persistent idempotency_key column if multi-process or restart
# survival ever matters.
_IDEMPOTENCY_WINDOW_S = 5 * 60
_IDEMPOTENCY_MAX = 256
_idempotency_cache: "OrderedDict[str, tuple[str, float]]" = OrderedDict()


def _idempotency_key(
    *, subject: str, body: str, answer: str, tags: list[str] | None
) -> str:
    # strip() so "Foo " and "Foo" dedupe; we already require non-empty
    # subject/body, so a pure-whitespace value never reaches this path.
    parts = [
        subject.strip(),
        body.strip(),
        answer.strip(),
        "|".join(sorted(tags or [])),
    ]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _cache_lookup(key: str, now: float) -> str | None:
    hit = _idempotency_cache.get(key)
    if hit is None:
        return None
    existing_id, ts = hit
    if now - ts >= _IDEMPOTENCY_WINDOW_S:
        # Expired entry; drop it so a follow-up insert is allowed and we don't
        # leak a slot until the cache fills.
        _idempotency_cache.pop(key, None)
        return None
    _idempotency_cache.move_to_end(key)
    return existing_id


def _cache_record(key: str, new_id: str, now: float) -> None:
    _idempotency_cache[key] = (new_id, now)
    _idempotency_cache.move_to_end(key)
    while len(_idempotency_cache) > _IDEMPOTENCY_MAX:
        _idempotency_cache.popitem(last=False)


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
    cross_tool_replay_warning=True,
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
    key = _idempotency_key(subject=subject, body=body, answer=answer, tags=tags)
    now = time.monotonic()
    existing = _cache_lookup(key, now)
    if existing is not None:
        return {"id": existing, "duplicate_of": existing, "created": False}
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
    _cache_record(key, new_id, now)
    return {"id": new_id}
