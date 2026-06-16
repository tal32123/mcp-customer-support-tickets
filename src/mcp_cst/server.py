"""FastMCP entry point. Wires up tools, resources, and prompts."""

from __future__ import annotations
import logging
import sys
from typing import Callable, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing_extensions import Annotated

from .config import Config, LlmProvider
from .data.ingest import build_store_from_huggingface
from .data.store import TicketStore
from .errors import ErrorCode, McpCstError
from .llm.anthropic_client import AnthropicClient
from .llm.openai_client import OpenAIClient
from .prompts import draft_reply as draft_reply_module
from .resources import schema as schema_module
from .resources import ticket as ticket_module
from .tools import aggregate_tickets as aggregate_tickets_module
from .tools import get_ticket as get_ticket_module
from .tools import search_tickets as search_tickets_module
from .tools import server_info as server_info_module


log = logging.getLogger(__name__)
mcp = FastMCP("customer-support-tickets")


EmbedFn = Callable[[list[str]], np.ndarray]


# Module-level singletons. Populated by _init() at startup so concurrent
# tool dispatch never races on first-call initialization.
_CFG: Config | None = None
_STORE: TicketStore | None = None
_EMBED_PASSAGES: EmbedFn | None = None
_EMBED_QUERIES: EmbedFn | None = None


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
            prefixed, convert_to_numpy=True, normalize_embeddings=True,
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


def get_query_embedder() -> EmbedFn:
    if _EMBED_QUERIES is None:
        _init()
    return _EMBED_QUERIES  # type: ignore[return-value]


def _init() -> None:
    """Eagerly load the embedding model and open/build the store.

    Called from main() before mcp.run(), and as a fallback from the
    accessors above so tests that import the module directly still work.
    """
    global _STORE, _EMBED_PASSAGES, _EMBED_QUERIES
    cfg = get_config()
    if _EMBED_PASSAGES is None or _EMBED_QUERIES is None:
        _EMBED_PASSAGES, _EMBED_QUERIES = _make_embedders(cfg.embedding_model)
    if _STORE is None:
        if cfg.store_path.exists() and (cfg.store_path / "tickets.lance").exists():
            _STORE = TicketStore.open(path=cfg.store_path, revision=cfg.dataset_revision)
        else:
            log.info("Building store at %s — first-run, this takes a few minutes.", cfg.store_path)
            _STORE = build_store_from_huggingface(
                path=cfg.store_path,
                dataset_id=cfg.dataset_id,
                revision=cfg.dataset_revision,
                embedder=_EMBED_PASSAGES,
            )


# --- server_info ---------------------------------------------------------

@mcp.tool(description=server_info_module.DESCRIPTION)
def server_info() -> dict:
    return server_info_module.server_info_payload(cfg=get_config(), store=get_store())


# --- schema:// resource --------------------------------------------------

@mcp.resource("schema://tickets", description=schema_module.DESCRIPTION)
def schema_tickets() -> str:
    return schema_module.schema_resource_body()


# --- get_ticket + ticket:// resource -------------------------------------

@mcp.tool(description=get_ticket_module.DESCRIPTION)
def get_ticket(id: str) -> dict:
    return get_ticket_module.get_ticket_impl(get_store(), id)


@mcp.resource("ticket://{id}", description=ticket_module.DESCRIPTION)
def ticket(id: str) -> str:
    return ticket_module.ticket_resource_body(get_store(), id)


# --- search_tickets ------------------------------------------------------

@mcp.tool(description=search_tickets_module.DESCRIPTION)
def search_tickets(
    q: Annotated[str, Field(description="Free-text query; matched against subject, body, and tags with hybrid BM25 + vector.")],
    queue: Annotated[str | None, Field(description="Restrict to one queue value. Use schema://tickets to see valid values.")] = None,
    priority: Annotated[Literal["low", "medium", "high", "critical", "info"] | None, Field(description="Restrict to one priority.")] = None,
    language: Annotated[Literal["en", "de"] | None, Field(description="Restrict to English or German tickets.")] = None,
    type: Annotated[Literal["question", "incident", "request", "problem"] | None, Field(description="Restrict to one ticket type.")] = None,
    tags: Annotated[list[str] | None, Field(description="Filter to tickets whose normalized `tags` list contains these values. Combine with `tags_mode`.")] = None,
    tags_mode: Annotated[Literal["and", "or"], Field(description="'and' = ticket must contain ALL listed tags; 'or' = ANY of them.")] = "and",
    limit: Annotated[int, Field(description="Max hits to return. Default 10, hard cap 50.")] = 10,
) -> list[dict]:
    cfg = get_config()
    return search_tickets_module.search_tickets_impl(
        get_store(), get_query_embedder(),
        q=q, queue=queue, priority=priority, language=language, type=type,
        tags=tags, tags_mode=tags_mode, limit=limit,
        rerank_enabled=cfg.rerank_enabled,
    )


# --- aggregate_tickets ---------------------------------------------------

@mcp.tool(description=aggregate_tickets_module.DESCRIPTION)
def aggregate_tickets(
    group_by: Annotated[Literal["queue", "priority", "language", "type", "tags"], Field(description="Field to group rows by.")],
    queue: Annotated[str | None, Field(description="Restrict to one queue value.")] = None,
    priority: Annotated[Literal["low", "medium", "high", "critical", "info"] | None, Field(description="Restrict to one priority.")] = None,
    language: Annotated[Literal["en", "de"] | None, Field(description="Restrict to English or German.")] = None,
    type: Annotated[Literal["question", "incident", "request", "problem"] | None, Field(description="Restrict to one ticket type.")] = None,
    tags: Annotated[list[str] | None, Field(description="Filter to tickets whose normalized `tags` list contains these values.")] = None,
    tags_mode: Annotated[Literal["and", "or"], Field(description="'and' = all listed tags; 'or' = any.")] = "and",
) -> list[dict]:
    return aggregate_tickets_module.aggregate_tickets_impl(
        get_store(),
        group_by=group_by, queue=queue, priority=priority, language=language,
        type=type, tags=tags, tags_mode=tags_mode,
    )


# --- draft_reply prompt --------------------------------------------------

def _llm_client():
    cfg = get_config()
    if cfg.llm_provider is LlmProvider.ANTHROPIC:
        return AnthropicClient(model=cfg.anthropic_model)
    if cfg.llm_provider is LlmProvider.OPENAI:
        return OpenAIClient(model=cfg.openai_model)
    raise McpCstError(
        ErrorCode.NO_LLM_CONFIGURED,
        "draft_reply needs ANTHROPIC_API_KEY or OPENAI_API_KEY in the environment",
    )


@mcp.prompt(description=draft_reply_module.DESCRIPTION)
def draft_reply(
    ticket_id: Annotated[str, Field(description="12-char id of the ticket to reply to. Find via search_tickets or get_ticket first; confirm with the user before approving the draft.")],
    target_language: Annotated[str | None, Field(description="Language to write the draft in. Defaults to the ticket's own language field.")] = None,
) -> dict:
    return draft_reply_module.draft_reply_impl(
        get_store(), get_query_embedder(), _llm_client(),
        ticket_id=ticket_id, target_language=target_language,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _init()
    mcp.run()


if __name__ == "__main__":
    main()
