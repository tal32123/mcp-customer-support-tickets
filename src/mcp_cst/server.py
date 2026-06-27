"""FastMCP entry point. Wires up tools, resources, and prompts."""

from __future__ import annotations
import functools
import logging
import os
import sys
import threading
from typing import Callable, Literal, TypeVar

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field
from typing_extensions import Annotated, ParamSpec

from .config import Config
from .data.ingest import build_store_from_huggingface
from .data.store import TicketStore
from .errors import ErrorCode, McpCstError
from .prompts import draft_reply as draft_reply_module
from .resources import schema as schema_module
from .resources import ticket as ticket_module
from .tools import aggregate_tickets as aggregate_tickets_module
from .tools import create_ticket as create_ticket_module
from .tools import delete_ticket as delete_ticket_module
from .tools import get_ticket as get_ticket_module
from .tools import get_tickets as get_tickets_module
from .tools import search_and_fetch as search_and_fetch_module
from .tools import search_tickets as search_tickets_module
from .tools import server_info as server_info_module
from .tools import update_ticket as update_ticket_module


log = logging.getLogger(__name__)
mcp = FastMCP("customer-support-tickets")


EmbedFn = Callable[[list[str]], np.ndarray]

_PriorityLit = Literal["low", "medium", "high", "critical", "info"]
_LanguageLit = Literal["en", "de", "he"]
_TypeLit = Literal["question", "incident", "request", "problem"]


# Module-level singletons. Populated by _init() at startup so concurrent
# tool dispatch never races on first-call initialization.
_CFG: Config | None = None
_STORE: TicketStore | None = None
_EMBED_PASSAGES: EmbedFn | None = None
_EMBED_QUERIES: EmbedFn | None = None
# Background thread that warms the embedder so mcp.run() can start the
# stdio handshake without blocking on torch + model download.
_EMBED_THREAD: threading.Thread | None = None
# Captured warm-up exception; surfaced from the embedder accessors so a
# failed model load returns a structured error instead of None-deref.
_EMBED_ERROR: BaseException | None = None


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _wrap(impl: Callable[_P, _R]) -> Callable[_P, _R | dict]:
    """Catch structured + unexpected errors and convert to MCP error payloads.

    McpCstError -> {"error": {"code": <code>, "message": <message>}}
    Anything else -> log.exception(...) + {"error": {"code": "INTERNAL_ERROR", ...}}
    """

    @functools.wraps(impl)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | dict:
        try:
            return impl(*args, **kwargs)
        except McpCstError as e:
            return {"error": {"code": e.code.value, "message": e.message}}
        except Exception as e:
            log.exception("unhandled error in %s", impl.__name__)
            return {
                "error": {
                    "code": ErrorCode.INTERNAL_ERROR.value,
                    "message": f"{type(e).__name__}: {e}",
                }
            }

    return wrapper


def _make_embedders(model_name: str) -> tuple[EmbedFn, EmbedFn]:
    """Load the embedding model once and return (embed_passages, embed_queries).

    `intfloat/multilingual-e5-small` is trained with task-specific prefixes:
    `"passage: "` for documents at index time, `"query: "` for queries at
    search time. Using the same prefix for both halves silently degrades
    retrieval quality, so we expose them separately.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def _encode(prefixed: list[str]) -> np.ndarray:
        return model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def embed_passages(texts: list[str]) -> np.ndarray:
        return _encode([f"passage: {t}" for t in texts])

    def embed_queries(texts: list[str]) -> np.ndarray:
        return _encode([f"query: {t}" for t in texts])

    return embed_passages, embed_queries


def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config.from_env()
    return _CFG


def get_store() -> TicketStore:
    if _STORE is None:
        _init()
    return _STORE  # type: ignore[return-value]


def _await_embedder() -> None:
    """Block until the background warm-up thread finishes, if one is running."""
    t = _EMBED_THREAD
    if t is not None and t.is_alive():
        t.join()


def _check_embed_error() -> None:
    """Raise a structured error if the background warm-up failed.

    Without this the accessors return None and the next call to the embedder
    blows up with an obscure AttributeError; worse, _init() spawns a fresh
    thread on every retry, turning a single load failure into a thread storm.
    """
    if _EMBED_ERROR is not None:
        raise McpCstError(
            ErrorCode.DATASET_UNAVAILABLE,
            f"embedder failed to load: {type(_EMBED_ERROR).__name__}: {_EMBED_ERROR}",
        )


def get_query_embedder() -> EmbedFn:
    if _EMBED_QUERIES is None:
        _await_embedder()
    _check_embed_error()
    if _EMBED_QUERIES is None:
        _init()
        _await_embedder()
        _check_embed_error()
    return _EMBED_QUERIES  # type: ignore[return-value]


def get_passage_embedder() -> EmbedFn:
    if _EMBED_PASSAGES is None:
        _await_embedder()
    _check_embed_error()
    if _EMBED_PASSAGES is None:
        _init()
        _await_embedder()
        _check_embed_error()
    return _EMBED_PASSAGES  # type: ignore[return-value]


def _warm_embedders(cfg: Config) -> None:
    """Background warm-up: load the embedding model and publish the callables."""
    global _EMBED_PASSAGES, _EMBED_QUERIES, _EMBED_ERROR
    log.info("embedder warming up...")
    try:
        passages, queries = _make_embedders(cfg.embedding_model)
    except BaseException as e:
        _EMBED_ERROR = e
        log.exception("embedder warm-up failed")
        return
    _EMBED_PASSAGES, _EMBED_QUERIES = passages, queries
    log.info("embedder ready")


def _init() -> None:
    """Open/build the store eagerly and warm the embedder in the background.

    Called from main() before mcp.run(), and as a fallback from the
    accessors above so tests that import the module directly still work.
    Store load is fast and the store needs the embedder only on first build
    (cold cache) or insert; in that case we synchronously join the warm-up
    thread before passing the embedder into build_store_from_huggingface.
    """
    global _STORE, _EMBED_PASSAGES, _EMBED_QUERIES, _EMBED_THREAD
    cfg = get_config()

    if _EMBED_PASSAGES is None or _EMBED_QUERIES is None:
        # Don't respawn after a captured failure — that turns one load error
        # into a thread storm under retry.
        if _EMBED_ERROR is None and (
            _EMBED_THREAD is None or not _EMBED_THREAD.is_alive()
        ):
            t = threading.Thread(target=_warm_embedders, args=(cfg,), daemon=True)
            _EMBED_THREAD = t
            t.start()

    if _STORE is None:
        if TicketStore.is_valid(
            dsn=cfg.database_url,
            schema=cfg.db_schema,
            revision=cfg.dataset_revision,
        ):
            _STORE = TicketStore.connect(
                dsn=cfg.database_url,
                schema=cfg.db_schema,
                revision=cfg.dataset_revision,
                embedding_dim=cfg.embedding_dim,
            )
        else:
            # Cold DB: we need the passage embedder to build the index.
            _await_embedder()
            log.info(
                "Building store in %s schema=%s — first-run, this takes a few minutes.",
                cfg.database_url.split("@")[-1],
                cfg.db_schema,
            )
            _STORE = build_store_from_huggingface(
                dsn=cfg.database_url,
                schema=cfg.db_schema,
                dataset_id=cfg.dataset_id,
                revision=cfg.dataset_revision,
                embedder=_EMBED_PASSAGES,
                embedding_dim=cfg.embedding_dim,
            )


# --- server_info ---------------------------------------------------------


@mcp.tool(
    description=server_info_module.DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
)
@_wrap
def server_info() -> dict:
    return server_info_module.server_info_payload(cfg=get_config(), store=get_store())


# --- schema:// resource --------------------------------------------------


@mcp.resource("schema://tickets", description=schema_module.DESCRIPTION)
def schema_tickets() -> str:
    return schema_module.schema_resource_body()


# --- get_ticket + ticket:// resource -------------------------------------


@mcp.tool(
    description=get_ticket_module.DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
)
@_wrap
def get_ticket(id: str) -> dict:
    return get_ticket_module.get_ticket_impl(get_store(), id)


@mcp.resource("ticket://{id}", description=ticket_module.DESCRIPTION)
def ticket(id: str) -> str:
    return ticket_module.ticket_resource_body(get_store(), id)


# --- search_tickets ------------------------------------------------------


@mcp.tool(
    description=search_tickets_module.DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
)
@_wrap
def search_tickets(
    q: Annotated[
        str,
        Field(
            description="Free-text query; matched against subject, body, and tags with hybrid BM25 + vector.",
            max_length=1024,
        ),
    ],
    queue: Annotated[
        str | None,
        Field(
            description="Restrict to one queue value. Use schema://tickets to see valid values."
        ),
    ] = None,
    priority: Annotated[
        _PriorityLit | None,
        Field(description="Restrict to one priority."),
    ] = None,
    language: Annotated[
        _LanguageLit | None,
        Field(
            description=(
                "Restrict to one language (en, de, he). Pass this when the user's "
                "query is clearly in one language and they want same-language "
                "results; omit for cross-lingual recall."
            )
        ),
    ] = None,
    type: Annotated[
        _TypeLit | None,
        Field(description="Restrict to one ticket type."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(
            description="Filter to tickets whose normalized `tags` list contains these values. Combine with `tags_mode`.",
            max_length=16,
        ),
    ] = None,
    tags_mode: Annotated[
        Literal["and", "or"],
        Field(
            description="'and' = ticket must contain ALL listed tags; 'or' = ANY of them."
        ),
    ] = "and",
    limit: Annotated[
        int, Field(description="Max hits per page. Default 10, hard cap 50.")
    ] = 10,
    cursor: Annotated[
        str | None,
        Field(
            description="Opaque cursor for the next page (returned in next_cursor); pass None for the first page."
        ),
    ] = None,
) -> dict:
    return search_tickets_module.search_tickets_impl(
        get_store(),
        get_query_embedder(),
        q=q,
        queue=queue,
        priority=priority,
        language=language,
        type=type,
        tags=tags,
        tags_mode=tags_mode,
        limit=limit,
        cursor=cursor,
    )


# --- aggregate_tickets ---------------------------------------------------


@mcp.tool(
    description=aggregate_tickets_module.DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
)
@_wrap
def aggregate_tickets(
    group_by: Annotated[
        Literal["queue", "priority", "language", "type", "tags"],
        Field(description="Field to group rows by."),
    ],
    queue: Annotated[
        str | None, Field(description="Restrict to one queue value.")
    ] = None,
    priority: Annotated[
        _PriorityLit | None,
        Field(description="Restrict to one priority."),
    ] = None,
    language: Annotated[
        _LanguageLit | None,
        Field(
            description=(
                "Restrict to one language (en, de, he). Pass this when the user's "
                "query is clearly in one language; omit for cross-lingual recall."
            )
        ),
    ] = None,
    type: Annotated[
        _TypeLit | None,
        Field(description="Restrict to one ticket type."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(
            description="Filter to tickets whose normalized `tags` list contains these values."
        ),
    ] = None,
    tags_mode: Annotated[
        Literal["and", "or"], Field(description="'and' = all listed tags; 'or' = any.")
    ] = "and",
) -> list[dict]:
    return aggregate_tickets_module.aggregate_tickets_impl(
        get_store(),
        group_by=group_by,
        queue=queue,
        priority=priority,
        language=language,
        type=type,
        tags=tags,
        tags_mode=tags_mode,
    )


# --- create_ticket --------------------------------------------------------


@mcp.tool(
    description=create_ticket_module.DESCRIPTION,
    annotations=ToolAnnotations(idempotentHint=False),
)
@_wrap
def create_ticket(
    subject: Annotated[
        str,
        Field(
            description="Ticket subject line. Required, non-empty.",
            max_length=256,
        ),
    ],
    body: Annotated[
        str,
        Field(
            description="Ticket body text. Required, non-empty.",
            max_length=16384,
        ),
    ],
    answer: Annotated[
        str,
        Field(
            description="Optional resolved answer if the ticket already has one. Empty by default.",
            max_length=16384,
        ),
    ] = "",
    type: Annotated[
        _TypeLit | None,
        Field(description="Optional ticket type."),
    ] = None,
    queue: Annotated[
        str | None,
        Field(
            description="Optional queue. Any string — schema://tickets lists the 52 dataset values."
        ),
    ] = None,
    priority: Annotated[
        _PriorityLit | None,
        Field(description="Optional priority."),
    ] = None,
    language: Annotated[
        _LanguageLit | None, Field(description="Optional language tag.")
    ] = None,
    version: Annotated[
        str, Field(description="Optional product/version label. Empty by default.")
    ] = "",
    tags: Annotated[
        list[str] | None,
        Field(
            description="Optional list of tags (already normalized — non-empty strings only).",
            max_length=16,
        ),
    ] = None,
) -> dict:
    return create_ticket_module.create_ticket_impl(
        get_store(),
        get_passage_embedder(),
        subject=subject,
        body=body,
        answer=answer,
        type=type or "",
        queue=queue or "",
        priority=priority or "",
        language=language or "",
        version=version,
        tags=tags,
    )


# --- update_ticket --------------------------------------------------------


@mcp.tool(
    description=update_ticket_module.DESCRIPTION,
    annotations=ToolAnnotations(idempotentHint=False),
)
@_wrap
def update_ticket(
    ticket_id: Annotated[
        str,
        Field(
            description="32-char UUIDv7 hex ticket id.",
            max_length=32,
        ),
    ],
    subject: Annotated[
        str | None,
        Field(description="New subject. Omit to keep current.", max_length=256),
    ] = None,
    body: Annotated[
        str | None,
        Field(description="New body text. Omit to keep current.", max_length=16384),
    ] = None,
    answer: Annotated[
        str | None,
        Field(description="New answer text. Omit to keep current.", max_length=16384),
    ] = None,
    type: Annotated[
        _TypeLit | None,
        Field(description="New ticket type. Omit to keep current."),
    ] = None,
    queue: Annotated[
        str | None, Field(description="New queue. Omit to keep current.")
    ] = None,
    priority: Annotated[
        _PriorityLit | None,
        Field(description="New priority. Omit to keep current."),
    ] = None,
    language: Annotated[
        _LanguageLit | None,
        Field(description="New language. Omit to keep current."),
    ] = None,
    version: Annotated[
        str | None, Field(description="New version label. Omit to keep current.")
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(
            description="New tag list (replaces all). Omit to keep current.",
            max_length=16,
        ),
    ] = None,
) -> dict:
    return update_ticket_module.update_ticket_impl(
        get_store(),
        get_passage_embedder(),
        ticket_id=ticket_id,
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


# --- delete_ticket --------------------------------------------------------


@mcp.tool(
    description=delete_ticket_module.DESCRIPTION,
    annotations=ToolAnnotations(destructiveHint=True, idempotentHint=False),
)
@_wrap
def delete_ticket(
    ticket_id: Annotated[
        str,
        Field(
            description="32-char UUIDv7 hex ticket id to delete. Confirm with the user first — deletion is irreversible.",
            max_length=32,
        ),
    ],
) -> dict:
    return delete_ticket_module.delete_ticket_impl(get_store(), ticket_id)


# --- draft_reply prompt --------------------------------------------------


@mcp.prompt(description=draft_reply_module.DESCRIPTION)
def draft_reply(
    ticket_id: Annotated[
        str,
        Field(
            description="32-char UUIDv7 hex ticket id to reply to. Find via search_tickets or get_ticket first; confirm with the user before approving the draft.",
            max_length=32,
        ),
    ],
    target_language: Annotated[
        str | None,
        Field(
            description="Language to write the draft in. Defaults to the ticket's own language field."
        ),
    ] = None,
) -> str:
    # FastMCP's prompt protocol needs a string (or PromptMessage list); the impl
    # returns a dict so unit tests can assert on metadata, so unwrap here.
    return draft_reply_module.draft_reply_impl(
        get_store(),
        get_query_embedder(),
        ticket_id=ticket_id,
        target_language=target_language,
    )["prompt"]


# --- get_tickets (batch) --------------------------------------------------


@mcp.tool(
    description=get_tickets_module.DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
)
@_wrap
def get_tickets(
    ids: Annotated[
        list[str],
        Field(
            description=(
                "List of ticket ids to fetch in one round-trip. Order preserved; "
                "unknown ids become null. Hard cap 50."
            ),
            max_length=50,
        ),
    ],
) -> list[dict | None]:
    return get_tickets_module.get_tickets_impl(get_store(), ids)


# --- search_and_fetch -----------------------------------------------------


@mcp.tool(
    description=search_and_fetch_module.DESCRIPTION,
    annotations=ToolAnnotations(readOnlyHint=True),
)
@_wrap
def search_and_fetch(
    q: Annotated[
        str,
        Field(
            description="Free-text query; hybrid BM25 + vector retrieval.",
            max_length=1024,
        ),
    ],
    queue: Annotated[
        str | None,
        Field(description="Restrict to one queue value."),
    ] = None,
    priority: Annotated[
        _PriorityLit | None,
        Field(description="Restrict to one priority."),
    ] = None,
    language: Annotated[
        _LanguageLit | None,
        Field(
            description=(
                "Restrict to one language (en, de, he). Pass this when the user's "
                "query is clearly in one language; omit for cross-lingual recall."
            )
        ),
    ] = None,
    type: Annotated[
        _TypeLit | None,
        Field(description="Restrict to one ticket type."),
    ] = None,
    tags: Annotated[
        list[str] | None,
        Field(
            description="Filter to tickets whose normalized `tags` list contains these values.",
            max_length=16,
        ),
    ] = None,
    tags_mode: Annotated[
        Literal["and", "or"],
        Field(description="'and' = ALL listed tags; 'or' = ANY."),
    ] = "and",
    k: Annotated[
        int,
        Field(description="Max full rows to return. Default 10, hard cap 50."),
    ] = 10,
    include: Annotated[
        Literal["body", "answer", "all"],
        Field(
            description="'body' drops answer; 'answer' drops body; 'all' returns both."
        ),
    ] = "all",
) -> list[dict]:
    return search_and_fetch_module.search_and_fetch_impl(
        get_store(),
        get_query_embedder(),
        q=q,
        queue=queue,
        priority=priority,
        language=language,
        type=type,
        tags=tags,
        tags_mode=tags_mode,
        k=k,
        include=include,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _init()
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if transport in ("streamable-http", "http"):
        mcp.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
        mcp.settings.port = int(os.environ.get("PORT", os.environ.get("MCP_PORT", "8000")))
        log.info("starting streamable-http on %s:%s", mcp.settings.host, mcp.settings.port)
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
