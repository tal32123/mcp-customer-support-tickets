"""Runtime configuration parsed from environment variables."""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs


DATASET_ID = "Tobi-Bueck/customer-support-tickets"
DEFAULT_REVISION = "main"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
CACHE_APPNAME = "mcp-customer-support-tickets"


@dataclass(frozen=True)
class Config:
    dataset_id: str
    dataset_revision: str
    embedding_model: str
    embedding_dim: int
    cache_root: Path
    rerank_enabled: bool

    @property
    def store_path(self) -> Path:
        """Per-revision, per-model store directory."""
        model_slug = self.embedding_model.rsplit("/", 1)[-1]
        return self.cache_root / self.dataset_revision / model_slug

    @classmethod
    def from_env(cls) -> "Config":
        cache_override = os.environ.get("MCP_CST_CACHE_DIR")
        cache_root = Path(cache_override) if cache_override else Path(platformdirs.user_cache_dir(CACHE_APPNAME))

        rerank = os.environ.get("RERANK", "").lower() == "true"

        return cls(
            dataset_id=DATASET_ID,
            dataset_revision=os.environ.get("MCP_CST_DATASET_REVISION", DEFAULT_REVISION),
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=EMBEDDING_DIM,
            cache_root=cache_root,
            rerank_enabled=rerank,
        )
