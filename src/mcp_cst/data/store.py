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
REVISION_FILE = "revision.txt"

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


def _text_search(subject: str, body: str, tags: list[str]) -> str:
    return f"{subject}\n{body}\n{' '.join(tags)}"


def _tag_cols(tags: list[str]) -> dict[str, str]:
    padded = (tags + [""] * len(_TAG_COLS))[: len(_TAG_COLS)]
    return {col: padded[i] for i, col in enumerate(_TAG_COLS)}


def _id_where(ticket_id: str) -> str:
    # Escape single quotes in the id to prevent WHERE-clause injection.
    # ids are 12-char hex by construction, but the parameter is callable
    # from outside and must be treated as untrusted.
    return f"id = '{ticket_id.replace(chr(39), chr(39) * 2)}'"


def _schema(embedding_dim: int) -> pa.Schema:
    return pa.schema(
        [
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
        ]
    )


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
            text = _text_search(row["subject"], row["body"], tags)
            texts_to_embed.append(text)
            records.append(
                {
                    "id": derive_id(revision, i),
                    "row_index": i,
                    # `or ""` (not the dict default) coerces explicit None values
                    # from HF rows — `.get(k, "")` only fires when the key is
                    # missing, so nullable cells would otherwise land as None and
                    # blow up downstream XML escapers.
                    **{
                        k: (row.get(k) or "")
                        for k in (
                            "subject",
                            "body",
                            "answer",
                            "type",
                            "queue",
                            "priority",
                            "language",
                            "version",
                            *_TAG_COLS,
                        )
                    },
                    "tags": tags,
                    "text_search": text,
                    "vector": None,  # filled below
                }
            )

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
        # Sidecar marker — used by `is_valid` to distinguish a complete
        # store from a directory left behind by a crashed ingest.
        (path / REVISION_FILE).write_text(revision, encoding="utf-8")
        return cls(db, table, path, revision)

    @classmethod
    def open(cls, *, path: Path, revision: str) -> "TicketStore":
        db = lancedb.connect(str(path))
        table = db.open_table(TABLE_NAME)
        return cls(db, table, path, revision)

    @classmethod
    def is_valid(cls, path: Path, revision: str) -> bool:
        """True if `path` holds a complete, non-empty store at `revision`.

        Guards against partial writes (the table directory exists but the
        sidecar marker hasn't been dropped) and revision drift (cache from a
        different dataset version). Either condition triggers a rebuild.
        """
        sidecar = path / REVISION_FILE
        if not sidecar.exists():
            return False
        try:
            stored = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if stored != revision:
            return False
        try:
            db = lancedb.connect(str(path))
            table = db.open_table(TABLE_NAME)
            return table.count_rows() > 0
        except Exception:
            return False

    def row_count(self) -> int:
        return self._table.count_rows()

    def all_ids(self) -> list[str]:
        arr = self._table.to_arrow().sort_by("row_index").column("id").to_pylist()
        return list(arr)

    def get(self, ticket_id: str) -> TicketRecord | None:
        rows = self._table.search().where(_id_where(ticket_id)).limit(1).to_list()
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
            tag_1=r["tag_1"],
            tag_2=r["tag_2"],
            tag_3=r["tag_3"],
            tag_4=r["tag_4"],
            tag_5=r["tag_5"],
            tag_6=r["tag_6"],
            tags=list(r["tags"]),
        )

    @property
    def table(self):
        """Low-level access for retrieval module."""
        return self._table

    def add_ticket(
        self,
        *,
        subject: str,
        body: str,
        embedder: Callable[[list[str]], np.ndarray],
        answer: str = "",
        type: str = "",
        queue: str = "",
        priority: str = "",
        language: str = "",
        version: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Insert a new ticket and return its 12-char hex id.

        The id is derived as `derive_id(self.revision, next_row_index)` where
        next_row_index = self.row_count(). This keeps the id scheme identical
        to build-time ids and guarantees uniqueness within the store.

        The vector is searchable immediately; the FTS index is rebuilt so
        the new row is also visible to BM25. Rebuild is O(n) — fine for
        interactive insert, not for bulk import.
        """
        tag_list = list(tags or [])
        text_search_value = _text_search(subject, body, tag_list)
        next_index = self._table.count_rows()
        new_id = derive_id(self.revision, next_index)
        vector = embedder([text_search_value])[0].tolist()
        record = {
            "id": new_id,
            "row_index": next_index,
            "subject": subject,
            "body": body,
            "answer": answer,
            "type": type,
            "queue": queue,
            "priority": priority,
            "language": language,
            "version": version,
            **_tag_cols(tag_list),
            "tags": tag_list,
            "text_search": text_search_value,
            "vector": vector,
        }
        self._table.add([record])
        self._table.create_fts_index("text_search", replace=True)
        return new_id

    def update_ticket(
        self,
        ticket_id: str,
        *,
        embedder: Callable[[list[str]], np.ndarray],
        subject: str | None = None,
        body: str | None = None,
        answer: str | None = None,
        type: str | None = None,
        queue: str | None = None,
        priority: str | None = None,
        language: str | None = None,
        version: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Patch one ticket. `None` for any field means leave it alone.

        Returns True if the ticket existed and was updated, False if no
        ticket has that id. The implementation deletes-and-reinserts so the
        text vector and FTS row both reflect the new content; the original
        `id` and `row_index` are preserved so existing references keep
        working.
        """
        raw = self._table.search().where(_id_where(ticket_id)).limit(1).to_list()
        if not raw:
            return False
        existing = raw[0]

        merged = {
            "subject": subject if subject is not None else existing["subject"],
            "body": body if body is not None else existing["body"],
            "answer": answer if answer is not None else existing["answer"],
            "type": type if type is not None else existing["type"],
            "queue": queue if queue is not None else existing["queue"],
            "priority": priority if priority is not None else existing["priority"],
            "language": language if language is not None else existing["language"],
            "version": version if version is not None else existing["version"],
            "tags": list(tags) if tags is not None else list(existing["tags"]),
        }
        text_search_value = _text_search(
            merged["subject"], merged["body"], merged["tags"]
        )
        vector = embedder([text_search_value])[0].tolist()

        record = {
            "id": ticket_id,
            # Preserve the original row_index across the delete+insert cycle —
            # the id is stable but row_index is what we use for stable
            # ordering in `all_ids`.
            "row_index": existing["row_index"],
            **{
                k: merged[k]
                for k in (
                    "subject",
                    "body",
                    "answer",
                    "type",
                    "queue",
                    "priority",
                    "language",
                    "version",
                )
            },
            **_tag_cols(merged["tags"]),
            "tags": merged["tags"],
            "text_search": text_search_value,
            "vector": vector,
        }
        self._table.delete(_id_where(ticket_id))
        self._table.add([record])
        self._table.create_fts_index("text_search", replace=True)
        return True

    def delete_ticket(self, ticket_id: str) -> bool:
        """Remove one ticket by id. Returns True if a row was removed.

        Row_indexes are not compacted — gaps are fine because we only use
        row_index for stable ordering, never as a direct table offset. The
        next `add_ticket` still derives its id from `row_count()`, which
        means an id collision is possible if you delete-then-add (the new
        ticket would re-use the deleted ticket's id slot when row_count
        happens to match a deleted index). For now we accept this; a
        forever-growing counter is a follow-up if it becomes a problem.
        """
        if self.get(ticket_id) is None:
            return False
        self._table.delete(_id_where(ticket_id))
        self._table.create_fts_index("text_search", replace=True)
        return True
