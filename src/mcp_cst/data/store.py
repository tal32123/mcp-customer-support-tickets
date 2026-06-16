"""LanceDB-backed ticket store: rows + BM25 FTS + vectors."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lancedb
import numpy as np
import pyarrow as pa


TABLE_NAME = "tickets"

_TAG_COLS = [f"tag_{i}" for i in range(1, 7)]


@dataclass(frozen=True)
class TicketRecord:
    id: str
    subject: str
    body: str
    answer: str
    type: str
    queue: str
    priority: str
    language: str
    version: str
    tag_1: str
    tag_2: str
    tag_3: str
    tag_4: str
    tag_5: str
    tag_6: str
    tags: list[str]


def derive_id(revision: str, row_index: int) -> str:
    return hashlib.sha1(f"{revision}|{row_index}".encode()).hexdigest()[:12]


def _normalize_tags(row: dict) -> list[str]:
    return [v for v in (row.get(c, "") for c in _TAG_COLS) if v]


def _text_search(row: dict, tags: list[str]) -> str:
    return f"{row['subject']}\n{row['body']}\n{' '.join(tags)}"


def _schema(embedding_dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("row_index", pa.int32()),
        pa.field("subject", pa.string()),
        pa.field("body", pa.string()),
        pa.field("answer", pa.string()),
        pa.field("type", pa.string()),
        pa.field("queue", pa.string()),
        pa.field("priority", pa.string()),
        pa.field("language", pa.string()),
        pa.field("version", pa.string()),
        *(pa.field(c, pa.string()) for c in _TAG_COLS),
        pa.field("tags", pa.list_(pa.string())),
        pa.field("text_search", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), embedding_dim)),
    ])


class TicketStore:
    """Wraps a LanceDB table with our domain accessors."""

    def __init__(self, db, table, path: Path, revision: str) -> None:
        self._db = db
        self._table = table
        self.path = path
        self.revision = revision

    @classmethod
    def create(
        cls,
        *,
        path: Path,
        revision: str,
        rows: list[dict],
        embedder: Callable[[list[str]], np.ndarray],
        embedding_dim: int = 384,
    ) -> "TicketStore":
        path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(path))

        records: list[dict] = []
        texts_to_embed: list[str] = []
        for i, row in enumerate(rows):
            tags = _normalize_tags(row)
            text = _text_search(row, tags)
            texts_to_embed.append(text)
            records.append({
                "id": derive_id(revision, i),
                "row_index": i,
                **{k: row.get(k, "") for k in (
                    "subject", "body", "answer", "type", "queue",
                    "priority", "language", "version", *_TAG_COLS,
                )},
                "tags": tags,
                "text_search": text,
                "vector": None,  # filled below
            })

        vectors = embedder(texts_to_embed)
        for rec, vec in zip(records, vectors):
            rec["vector"] = vec.tolist()

        table = db.create_table(
            TABLE_NAME,
            data=records,
            schema=_schema(embedding_dim),
            mode="overwrite",
        )
        # BM25 full-text index over text_search
        table.create_fts_index("text_search", replace=True)
        return cls(db, table, path, revision)

    @classmethod
    def open(cls, *, path: Path, revision: str) -> "TicketStore":
        db = lancedb.connect(str(path))
        table = db.open_table(TABLE_NAME)
        return cls(db, table, path, revision)

    def row_count(self) -> int:
        return self._table.count_rows()

    def all_ids(self) -> list[str]:
        arr = self._table.to_arrow().sort_by("row_index").column("id").to_pylist()
        return list(arr)

    def get(self, ticket_id: str) -> TicketRecord | None:
        # Escape single quotes in the id to prevent WHERE-clause injection.
        # ids are 12-char hex by construction, but the parameter is callable
        # from outside and must be treated as untrusted.
        safe = ticket_id.replace("'", "''")
        rows = self._table.search().where(f"id = '{safe}'").limit(1).to_list()
        if not rows:
            return None
        r = rows[0]
        return TicketRecord(
            id=r["id"],
            subject=r["subject"],
            body=r["body"],
            answer=r["answer"],
            type=r["type"],
            queue=r["queue"],
            priority=r["priority"],
            language=r["language"],
            version=r["version"],
            tag_1=r["tag_1"], tag_2=r["tag_2"], tag_3=r["tag_3"],
            tag_4=r["tag_4"], tag_5=r["tag_5"], tag_6=r["tag_6"],
            tags=list(r["tags"]),
        )

    @property
    def table(self):
        """Low-level access for retrieval module."""
        return self._table
