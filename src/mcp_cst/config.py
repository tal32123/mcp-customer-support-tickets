"""Runtime configuration parsed from environment variables."""

from __future__ import annotations
import os
from dataclasses import dataclass


DATASET_ID = "Tobi-Bueck/customer-support-tickets"
DEFAULT_REVISION = "main"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
DEFAULT_SCHEMA = "public"
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/postgres"


@dataclass(frozen=True)
class Config:
    dataset_id: str
    dataset_revision: str
    embedding_model: str
    embedding_dim: int
    database_url: str
    db_schema: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            dataset_id=DATASET_ID,
            dataset_revision=os.environ.get(
                "MCP_CST_DATASET_REVISION", DEFAULT_REVISION
            ),
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=EMBEDDING_DIM,
            database_url=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
            db_schema=os.environ.get("MCP_CST_DB_SCHEMA", DEFAULT_SCHEMA),
        )
