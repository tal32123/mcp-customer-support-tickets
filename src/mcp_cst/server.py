"""FastMCP entry point. Wires up tools, resources, and prompts."""

from __future__ import annotations
import logging
import sys
from typing import Literal

import numpy as np
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing_extensions import Annotated

from .config import Config
from .data.ingest import build_store_from_huggingface
from .data.store import TicketStore
from .resources import schema as schema_module
from .resources import ticket as ticket_module
from .tools import get_ticket as get_ticket_module
from .tools import search_tickets as search_tickets_module
from .tools import server_info as server_info_module


log = logging.getLogger(__name__)
mcp = FastMCP("customer-support-tickets")


# Lazy globals — initialized on first use so test code can override.
_CFG: Config | None = None
_STORE: TicketStore | None = None


def _embedder():
    """Return a real embedding function. Lazily loads sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model_name = get_config().embedding_model
    model = SentenceTransformer(model_name)
    def embed(texts: list[str]) -> np.ndarray:
        prefixed = [f"query: {t}" for t in texts]
        return model.encode(prefixed, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    return embed


def get_config() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = Config.from_env()
    return _CFG


def get_store() -> TicketStore:
    global _STORE
    if _STORE is not None:
        return _STORE
    cfg = get_config()
    if cfg.store_path.exists() and (cfg.store_path / "tickets.lance").exists():
        _STORE = TicketStore.open(path=cfg.store_path, revision=cfg.dataset_revision)
        return _STORE
    log.info("Building store at %s — first-run, this takes a few minutes.", cfg.store_path)
    _STORE = build_store_from_huggingface(
        path=cfg.store_path,
        dataset_id=cfg.dataset_id,
        revision=cfg.dataset_revision,
        embedder=_embedder(),
    )
    return _STORE


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
        get_store(), _embedder(),
        q=q, queue=queue, priority=priority, language=language, type=type,
        tags=tags, tags_mode=tags_mode, limit=limit,
        rerank_enabled=cfg.rerank_enabled,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
