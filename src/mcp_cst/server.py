"""FastMCP entry point. Wires up tools, resources, and prompts."""

from __future__ import annotations
import logging
import sys

import numpy as np
from mcp.server.fastmcp import FastMCP

from .config import Config
from .data.ingest import build_store_from_huggingface
from .data.store import TicketStore
from .resources import schema as schema_module
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
