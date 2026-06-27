"""Postgres + pgvector backed ticket store: rows + tsvector FTS + vectors."""

from __future__ import annotations
import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector


TABLE_NAME = "tickets"
META_TABLE_NAME = "store_meta"
# Bump when the on-disk schema changes so `is_valid` forces a clean rebuild.
SCHEMA_VERSION = "v3-postgres"

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
    original_system_id: str = ""


def derive_id(revision: str, row_index: int) -> str:
    """Deterministic 12-hex id for HF dataset rows. Stored in
    ``original_system_id`` at ingest — never used as the primary ``id``
    (always a UUIDv7). Preserved so tests, fixtures, and cross-references
    that knew the old id scheme still resolve.
    """
    return hashlib.sha1(f"{revision}|{row_index}".encode()).hexdigest()[:12]


def _uuid7_hex() -> str:
    """Hand-rolled UUIDv7 → 32 lowercase hex chars (no hyphens).

    ponytail: no monotonic counter; same-ms collisions astronomically
    unlikely at single-process MCP server write rates.
    """
    ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = os.urandom(10)
    out = bytearray(16)
    out[0:6] = ms.to_bytes(6, "big")
    out[6] = 0x70 | (rand[0] & 0x0F)
    out[7] = rand[1]
    out[8] = 0x80 | (rand[2] & 0x3F)
    out[9:16] = rand[3:10]
    return out.hex()


def _normalize_tags(row: dict) -> list[str]:
    return [v for v in (row.get(c, "") for c in _TAG_COLS) if v]


def _text_search(subject: str, body: str, tags: list[str]) -> str:
    return f"{subject}\n{body}\n{' '.join(tags)}"


def _tag_cols(tags: list[str]) -> dict[str, str]:
    padded = (tags + [""] * len(_TAG_COLS))[: len(_TAG_COLS)]
    return {col: padded[i] for i, col in enumerate(_TAG_COLS)}


def _qual(schema: str, table: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))


def _configure_conn(conn: psycopg.Connection) -> None:
    register_vector(conn)


_ALL_COLS = (
    "id",
    "original_system_id",
    "row_index",
    "subject",
    "body",
    "answer",
    "type",
    "queue",
    "priority",
    "language",
    "version",
    *_TAG_COLS,
    "tags",
)


def _record_from_row(r: dict) -> TicketRecord:
    return TicketRecord(
        id=r["id"],
        subject=r["subject"] or "",
        body=r["body"] or "",
        answer=r["answer"] or "",
        type=r["type"] or "",
        queue=r["queue"] or "",
        priority=r["priority"] or "",
        language=r["language"] or "",
        version=r["version"] or "",
        tag_1=r["tag_1"] or "",
        tag_2=r["tag_2"] or "",
        tag_3=r["tag_3"] or "",
        tag_4=r["tag_4"] or "",
        tag_5=r["tag_5"] or "",
        tag_6=r["tag_6"] or "",
        tags=list(r["tags"] or []),
        original_system_id=r.get("original_system_id") or "",
    )


class TicketStore:
    """Wraps a Postgres schema holding the tickets table + indexes.

    Public surface mirrors the previous LanceDB-backed store so tools and
    retrieval modules stay unchanged.
    """

    def __init__(
        self, pool: ConnectionPool, schema: str, revision: str, embedding_dim: int
    ) -> None:
        self._pool = pool
        self._schema = schema
        self.revision = revision
        self.embedding_dim = embedding_dim
        self._write_lock = threading.Lock()
        # Monotonically incremented on every mutation. Read-side caches
        # (aggregates) key on this so update_ticket still busts cached results.
        self._write_seq = 0

    # --- lifecycle ----------------------------------------------------------

    @classmethod
    def connect(
        cls,
        *,
        dsn: str,
        schema: str = "public",
        revision: str,
        embedding_dim: int = 384,
        min_size: int = 1,
        max_size: int = 4,
    ) -> "TicketStore":
        """Open a pool and ensure the schema + tables + indexes exist.

        Idempotent — safe to call on an already-initialised database. Does
        not insert rows; pair with ``create_with_rows`` for first-boot
        ingest, or check ``is_valid`` and ingest via ``ingest_rows``.
        """
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
            )

        pool = ConnectionPool(
            conninfo=dsn,
            min_size=min_size,
            max_size=max_size,
            configure=_configure_conn,
            kwargs={"autocommit": False},
        )
        pool.wait()
        store = cls(pool, schema, revision, embedding_dim)
        store._ensure_tables()
        return store

    @classmethod
    def create_with_rows(
        cls,
        *,
        dsn: str,
        schema: str = "public",
        revision: str,
        rows: list[dict],
        embedder: Callable[[list[str]], np.ndarray],
        embedding_dim: int = 384,
    ) -> "TicketStore":
        """DROP + recreate the schema, then bulk-ingest ``rows``.

        Destructive — only the named schema is wiped, but every existing
        row in it is gone. Used by tests and by the first-boot ingest path.
        """
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
            conn.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )
        store = cls.connect(
            dsn=dsn, schema=schema, revision=revision, embedding_dim=embedding_dim
        )
        store.ingest_rows(rows, embedder)
        return store

    @classmethod
    def is_valid(
        cls,
        *,
        dsn: str,
        schema: str = "public",
        revision: str,
    ) -> bool:
        """True if the schema holds a complete, non-empty store at ``revision``.

        Guards against partial ingest (table exists but no rows), revision
        drift, schema-version drift, and connection failure.
        """
        try:
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT EXISTS (
                          SELECT 1 FROM information_schema.tables
                          WHERE table_schema = %s AND table_name = %s
                        )
                        """,
                        (schema, TABLE_NAME),
                    )
                    if not cur.fetchone()[0]:
                        return False
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {}").format(
                            _qual(schema, TABLE_NAME)
                        )
                    )
                    if cur.fetchone()[0] == 0:
                        return False
                    meta = _qual(schema, META_TABLE_NAME)
                    cur.execute(
                        sql.SQL("SELECT key, value FROM {}").format(meta)
                    )
                    kv = dict(cur.fetchall())
                    if kv.get("revision") != revision:
                        return False
                    if kv.get("schema_version") != SCHEMA_VERSION:
                        return False
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._pool.close()

    # --- schema -------------------------------------------------------------

    def _ensure_tables(self) -> None:
        # Single transaction: CREATE TABLE + indexes + meta seeding. Safe to
        # re-run — every statement is IF NOT EXISTS / ON CONFLICT.
        tickets = _qual(self._schema, TABLE_NAME)
        meta = _qual(self._schema, META_TABLE_NAME)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {tickets} (
                            id                 TEXT PRIMARY KEY,
                            original_system_id TEXT NOT NULL DEFAULT '',
                            row_index          BIGSERIAL NOT NULL,
                            subject            TEXT NOT NULL DEFAULT '',
                            body               TEXT NOT NULL DEFAULT '',
                            answer             TEXT NOT NULL DEFAULT '',
                            type               TEXT NOT NULL DEFAULT '',
                            queue              TEXT NOT NULL DEFAULT '',
                            priority           TEXT NOT NULL DEFAULT '',
                            language           TEXT NOT NULL DEFAULT '',
                            version            TEXT NOT NULL DEFAULT '',
                            tag_1              TEXT NOT NULL DEFAULT '',
                            tag_2              TEXT NOT NULL DEFAULT '',
                            tag_3              TEXT NOT NULL DEFAULT '',
                            tag_4              TEXT NOT NULL DEFAULT '',
                            tag_5              TEXT NOT NULL DEFAULT '',
                            tag_6              TEXT NOT NULL DEFAULT '',
                            tags               TEXT[] NOT NULL DEFAULT '{{}}',
                            text_search        TEXT NOT NULL DEFAULT '',
                            tsv                tsvector
                                               GENERATED ALWAYS AS
                                               (to_tsvector('simple', text_search))
                                               STORED,
                            embedding          vector({dim}) NOT NULL
                        )
                        """
                    ).format(tickets=tickets, dim=sql.Literal(self.embedding_dim))
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {meta} (
                            key   TEXT PRIMARY KEY,
                            value TEXT NOT NULL
                        )
                        """
                    ).format(meta=meta)
                )
                # GIN over tsvector for BM25-ish ranked search.
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} USING gin(tsv)"
                    ).format(
                        idx=sql.Identifier(f"{TABLE_NAME}_tsv_gin"), tbl=tickets
                    )
                )
                # GIN over tags for @> / && membership.
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} USING gin(tags)"
                    ).format(
                        idx=sql.Identifier(f"{TABLE_NAME}_tags_gin"), tbl=tickets
                    )
                )
                # Scalar btrees — cheap, help filter+aggregate.
                for col in ("queue", "priority", "language", "type", "row_index"):
                    cur.execute(
                        sql.SQL(
                            "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} ({col})"
                        ).format(
                            idx=sql.Identifier(f"{TABLE_NAME}_{col}_idx"),
                            tbl=tickets,
                            col=sql.Identifier(col),
                        )
                    )
                # HNSW ANN over the embedding column. Cheap to maintain at
                # this scale; the index is built lazily on first SELECT.
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} "
                        "USING hnsw (embedding vector_cosine_ops)"
                    ).format(
                        idx=sql.Identifier(f"{TABLE_NAME}_embedding_hnsw"),
                        tbl=tickets,
                    )
                )
                # Seed schema_version once. Revision is updated by ingest.
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {meta} (key, value) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                    ).format(meta=meta),
                    ("schema_version", SCHEMA_VERSION),
                )
            conn.commit()

    # --- ingest -------------------------------------------------------------

    def ingest_rows(
        self,
        rows: list[dict],
        embedder: Callable[[list[str]], np.ndarray],
        batch_size: int = 256,
    ) -> None:
        """Bulk-insert ``rows`` into a freshly-created store.

        Caller is responsible for embedding batching if it needs progress
        reporting — this method calls ``embedder`` once per ``batch_size``
        chunk so memory stays bounded on the full corpus.
        """
        tickets = _qual(self._schema, TABLE_NAME)
        meta = _qual(self._schema, META_TABLE_NAME)
        insert_sql = sql.SQL(
            """
            INSERT INTO {tickets} (
                id, original_system_id, row_index,
                subject, body, answer, type, queue, priority, language, version,
                tag_1, tag_2, tag_3, tag_4, tag_5, tag_6,
                tags, text_search, embedding
            ) VALUES (
                %(id)s, %(original_system_id)s, %(row_index)s,
                %(subject)s, %(body)s, %(answer)s, %(type)s, %(queue)s,
                %(priority)s, %(language)s, %(version)s,
                %(tag_1)s, %(tag_2)s, %(tag_3)s, %(tag_4)s, %(tag_5)s, %(tag_6)s,
                %(tags)s, %(text_search)s, %(embedding)s
            )
            """
        ).format(tickets=tickets)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for batch_start in range(0, len(rows), batch_size):
                    batch = rows[batch_start : batch_start + batch_size]
                    payload: list[dict[str, Any]] = []
                    texts: list[str] = []
                    for i, row in enumerate(batch):
                        # `or ""` coerces explicit None values from HF rows;
                        # `str(...)` defends against numeric cells leaking through.
                        coerced = {
                            k: str(row.get(k) or "")
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
                        }
                        tags = _normalize_tags(row)
                        text = _text_search(coerced["subject"], coerced["body"], tags)
                        texts.append(text)
                        payload.append(
                            {
                                "id": _uuid7_hex(),
                                "original_system_id": derive_id(
                                    self.revision, batch_start + i
                                ),
                                "row_index": batch_start + i,
                                **coerced,
                                "tags": tags,
                                "text_search": text,
                            }
                        )
                    vectors = embedder(texts)
                    for rec, vec in zip(payload, vectors):
                        rec["embedding"] = np.asarray(vec, dtype=np.float32)
                    cur.executemany(insert_sql, payload)
                # Bump the row_index sequence past the explicit values so
                # subsequent add_ticket() inserts get fresh row_indexes.
                if rows:
                    cur.execute(
                        "SELECT setval(pg_get_serial_sequence(%s, %s), %s, false)",
                        (
                            f'"{self._schema}"."{TABLE_NAME}"',
                            "row_index",
                            len(rows),
                        ),
                    )
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {meta} (key, value) VALUES (%s, %s) "
                        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                    ).format(meta=meta),
                    ("revision", self.revision),
                )
            conn.commit()

    # --- read paths ---------------------------------------------------------

    def row_count(self) -> int:
        tickets = _qual(self._schema, TABLE_NAME)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(tickets))
            return int(cur.fetchone()[0])

    def all_ids(self) -> list[str]:
        tickets = _qual(self._schema, TABLE_NAME)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id FROM {} ORDER BY row_index").format(tickets)
            )
            return [r[0] for r in cur.fetchall()]

    def _raw_get(self, ticket_id: str) -> dict | None:
        tickets = _qual(self._schema, TABLE_NAME)
        cols = sql.SQL(", ").join(sql.Identifier(c) for c in _ALL_COLS)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql.SQL("SELECT {cols} FROM {tbl} WHERE id = %s").format(
                        cols=cols, tbl=tickets
                    ),
                    (ticket_id,),
                )
                return cur.fetchone()

    def get(self, ticket_id: str) -> TicketRecord | None:
        r = self._raw_get(ticket_id)
        return _record_from_row(r) if r else None

    def text_search_of(self, ticket_id: str) -> str | None:
        """Return the persisted ``text_search`` for ``ticket_id``.

        Debug helper used by tests that need to assert what BM25 actually
        sees — the regular ``get()`` path strips this column.
        """
        tickets = _qual(self._schema, TABLE_NAME)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT text_search FROM {} WHERE id = %s").format(tickets),
                (ticket_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    @property
    def schema_name(self) -> str:
        return self._schema

    @property
    def pool(self) -> ConnectionPool:
        """Low-level access for retrieval / aggregates modules."""
        return self._pool

    @property
    def write_seq(self) -> int:
        """Monotonic write counter — use as cache key in read-side caches."""
        return self._write_seq

    def _bump_seq(self) -> None:
        self._write_seq += 1

    # --- write paths --------------------------------------------------------

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
        """Insert a new ticket and return its UUIDv7 hex id."""
        tag_list = list(tags or [])
        text = _text_search(subject, body, tag_list)
        vector = np.asarray(embedder([text])[0], dtype=np.float32)
        new_id = _uuid7_hex()
        tickets = _qual(self._schema, TABLE_NAME)
        cols_payload = {
            "id": new_id,
            "original_system_id": "",
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
            "text_search": text,
            "embedding": vector,
        }
        insert_sql = sql.SQL(
            """
            INSERT INTO {tickets} (
                id, original_system_id,
                subject, body, answer, type, queue, priority, language, version,
                tag_1, tag_2, tag_3, tag_4, tag_5, tag_6,
                tags, text_search, embedding
            ) VALUES (
                %(id)s, %(original_system_id)s,
                %(subject)s, %(body)s, %(answer)s, %(type)s, %(queue)s,
                %(priority)s, %(language)s, %(version)s,
                %(tag_1)s, %(tag_2)s, %(tag_3)s, %(tag_4)s, %(tag_5)s, %(tag_6)s,
                %(tags)s, %(text_search)s, %(embedding)s
            )
            """
        ).format(tickets=tickets)
        with self._write_lock:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(insert_sql, cols_payload)
                conn.commit()
            self._bump_seq()
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
        """Patch one ticket. ``None`` for any field means leave it alone.

        Returns True if the ticket existed and was updated, False if no
        ticket has that id. The text_search column is regenerated whenever
        subject/body/tags change, so the tsvector (a STORED generated
        column) and the embedding stay in sync.
        """
        with self._write_lock:
            existing = self._raw_get(ticket_id)
            if existing is None:
                return False
            merged = {
                "subject": subject if subject is not None else existing["subject"],
                "body": body if body is not None else existing["body"],
                "answer": answer if answer is not None else existing["answer"],
                "type": type if type is not None else existing["type"],
                "queue": queue if queue is not None else existing["queue"],
                "priority": priority
                if priority is not None
                else existing["priority"],
                "language": language
                if language is not None
                else existing["language"],
                "version": version if version is not None else existing["version"],
                "tags": list(tags) if tags is not None else list(existing["tags"]),
            }
            text = _text_search(merged["subject"], merged["body"], merged["tags"])
            vector = np.asarray(embedder([text])[0], dtype=np.float32)
            tickets = _qual(self._schema, TABLE_NAME)
            update_sql = sql.SQL(
                """
                UPDATE {tickets} SET
                    subject = %(subject)s,
                    body = %(body)s,
                    answer = %(answer)s,
                    type = %(type)s,
                    queue = %(queue)s,
                    priority = %(priority)s,
                    language = %(language)s,
                    version = %(version)s,
                    tag_1 = %(tag_1)s, tag_2 = %(tag_2)s, tag_3 = %(tag_3)s,
                    tag_4 = %(tag_4)s, tag_5 = %(tag_5)s, tag_6 = %(tag_6)s,
                    tags = %(tags)s,
                    text_search = %(text_search)s,
                    embedding = %(embedding)s
                WHERE id = %(id)s
                """
            ).format(tickets=tickets)
            payload = {
                "id": ticket_id,
                **merged,
                **_tag_cols(merged["tags"]),
                "text_search": text,
                "embedding": vector,
            }
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(update_sql, payload)
                conn.commit()
            self._bump_seq()
        return True

    def delete_ticket(self, ticket_id: str) -> bool:
        """Remove one ticket by id. Returns True if a row was removed."""
        tickets = _qual(self._schema, TABLE_NAME)
        with self._write_lock:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {} WHERE id = %s").format(tickets),
                        (ticket_id,),
                    )
                    removed = cur.rowcount > 0
                conn.commit()
            if removed:
                self._bump_seq()
        return removed

    # --- retrieval helpers (used by retrieval/hybrid + prompts/draft_reply) -

    def search_bm25(
        self,
        *,
        query: str,
        where_sql: sql.Composable | None,
        where_params: tuple,
        limit: int,
    ) -> list[str]:
        """Top-``limit`` ids ranked by ts_rank_cd against ``query``."""
        tickets = _qual(self._schema, TABLE_NAME)
        base = sql.SQL(
            "SELECT id FROM {tbl}, websearch_to_tsquery('simple', %s) q "
            "WHERE tsv @@ q"
        ).format(tbl=tickets)
        params: tuple = (query,)
        if where_sql is not None:
            base = base + sql.SQL(" AND ") + where_sql
            params = params + where_params
        base = base + sql.SQL(" ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %s")
        params = params + (limit,)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(base, params)
            return [r[0] for r in cur.fetchall()]

    def search_vector(
        self,
        *,
        qvec: np.ndarray,
        where_sql: sql.Composable | None,
        where_params: tuple,
        limit: int,
    ) -> list[str]:
        """Top-``limit`` ids by cosine distance to ``qvec``."""
        tickets = _qual(self._schema, TABLE_NAME)
        base = sql.SQL("SELECT id FROM {tbl}").format(tbl=tickets)
        params: tuple = ()
        if where_sql is not None:
            base = base + sql.SQL(" WHERE ") + where_sql
            params = where_params
        base = base + sql.SQL(" ORDER BY embedding <=> %s LIMIT %s")
        params = params + (np.asarray(qvec, dtype=np.float32), limit)
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(base, params)
            return [r[0] for r in cur.fetchall()]

    def fetch_for_hydration(self, ids: list[str]) -> list[dict]:
        """Return rows needed for hydrate_ids (subject, body snippet, etc.)."""
        if not ids:
            return []
        tickets = _qual(self._schema, TABLE_NAME)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT id, subject, body, language, queue, priority "
                        "FROM {tbl} WHERE id = ANY(%s)"
                    ).format(tbl=tickets),
                    (list(ids),),
                )
                return cur.fetchall()

    def grounding_candidates(
        self,
        *,
        qvec: np.ndarray,
        limit: int = 50,
    ) -> list[dict]:
        """Top-``limit`` rows by vector similarity for draft_reply grounding.

        Returns subject/body/answer/language plus a pre-computed
        ``similarity`` (cosine) so callers don't need the raw embedding.
        """
        tickets = _qual(self._schema, TABLE_NAME)
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT id, subject, body, answer, language, "
                        "1 - (embedding <=> %s) AS similarity "
                        "FROM {tbl} ORDER BY embedding <=> %s LIMIT %s"
                    ).format(tbl=tickets),
                    (
                        np.asarray(qvec, dtype=np.float32),
                        np.asarray(qvec, dtype=np.float32),
                        limit,
                    ),
                )
                return cur.fetchall()

    def group_count_query(
        self,
        *,
        group_by: str,
        where_sql: sql.Composable | None,
        where_params: tuple,
    ) -> list[dict]:
        """GROUP BY counts for ``group_by`` (scalar field or 'tags')."""
        tickets = _qual(self._schema, TABLE_NAME)
        if group_by == "tags":
            base = sql.SQL(
                "SELECT t AS group_value, count(*) AS cnt "
                "FROM {tbl}, unnest(tags) AS t"
            ).format(tbl=tickets)
            if where_sql is not None:
                base = base + sql.SQL(" WHERE ") + where_sql
            base = (
                base
                + sql.SQL(" AND t <> '' " if where_sql is not None else " WHERE t <> '' ")
                + sql.SQL("GROUP BY t ORDER BY cnt DESC")
            )
        else:
            base = sql.SQL(
                "SELECT {col} AS group_value, count(*) AS cnt FROM {tbl}"
            ).format(col=sql.Identifier(group_by), tbl=tickets)
            if where_sql is not None:
                base = base + sql.SQL(" WHERE ") + where_sql
            base = base + sql.SQL(" GROUP BY {col} ORDER BY cnt DESC").format(
                col=sql.Identifier(group_by)
            )
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(base, where_params)
                return cur.fetchall()
