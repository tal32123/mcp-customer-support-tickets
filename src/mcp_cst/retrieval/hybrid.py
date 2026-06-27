"""Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion.

BM25-ish branch uses Postgres ts_rank_cd over a STORED tsvector column.
Vector branch uses pgvector's cosine distance operator (``<=>``) over an
HNSW index. Filters compile to a single SQL WHERE clause shared by both
branches so candidate_k applies AFTER filtering, preserving recall.
"""

from __future__ import annotations
from typing import Callable, Literal, TypedDict

import numpy as np
from psycopg import sql

from ..data.store import TicketStore
from ..errors import ErrorCode, McpCstError
from ..safety import escape_text, looks_like_injection


FILTER_FIELDS = {"queue", "priority", "language", "type"}


class TicketFilters(TypedDict, total=False):
    queue: str
    priority: str
    language: str
    type: str
    tags: list[str]
    tags_mode: Literal["and", "or"]


def build_filters(
    *,
    queue: str | None = None,
    priority: str | None = None,
    language: str | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    tags_mode: Literal["and", "or"] = "and",
) -> TicketFilters:
    filters: TicketFilters = {}
    if queue is not None:
        filters["queue"] = queue
    if priority is not None:
        filters["priority"] = priority
    if language is not None:
        filters["language"] = language
    if type is not None:
        filters["type"] = type
    if tags:
        filters["tags"] = tags
        filters["tags_mode"] = tags_mode
    return filters


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Merge multiple ranked lists into one. Higher rank → higher score."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _build_where(
    filters: TicketFilters,
) -> tuple[sql.Composable | None, tuple]:
    """Compile a filters dict into (WHERE clause, params).

    Scalar filters → ``col = %s``. Tags: ``tags @> %s`` for tags_mode='and',
    ``tags && %s`` for tags_mode='or'. Pushed down so candidate_k applies
    AFTER the tag restriction, preserving recall.
    """
    clauses: list[sql.Composable] = []
    params: list = []
    tags: list[str] | None = None
    tags_mode: str = "and"
    for key, value in filters.items():
        if key == "tags":
            tags = value  # type: ignore[assignment]
            continue
        if key == "tags_mode":
            tags_mode = value  # type: ignore[assignment]
            continue
        if key not in FILTER_FIELDS:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER, f"unsupported filter field: {key}"
            )
        clauses.append(
            sql.SQL("{col} = %s").format(col=sql.Identifier(key))
        )
        params.append(value)

    if tags:
        if tags_mode not in {"and", "or"}:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER,
                f"tags_mode must be 'and' or 'or', got {tags_mode!r}",
            )
        non_empty = [t for t in tags if t]
        if non_empty:
            op = sql.SQL("@>") if tags_mode == "and" else sql.SQL("&&")
            clauses.append(sql.SQL("tags {op} %s").format(op=op))
            params.append(non_empty)

    if not clauses:
        return None, ()
    return sql.SQL(" AND ").join(clauses), tuple(params)


def hybrid_search_full(
    store: TicketStore,
    *,
    query: str,
    filters: TicketFilters,
    embedder: Callable[[list[str]], np.ndarray],
    candidate_k: int = 50,
) -> list[str]:
    """Run BM25 + vector, return the full fused id list (no slicing).

    Used by the cursor-paginated search path so we can cache one fusion
    result and slice it across many pages without re-running RRF.
    """
    where_sql, where_params = _build_where(filters)
    bm25_ids = store.search_bm25(
        query=query,
        where_sql=where_sql,
        where_params=where_params,
        limit=candidate_k,
    )
    qvec = embedder([query])[0]
    vec_ids = store.search_vector(
        qvec=qvec,
        where_sql=where_sql,
        where_params=where_params,
        limit=candidate_k,
    )
    return reciprocal_rank_fusion([bm25_ids, vec_ids])


def hydrate_ids(
    store: TicketStore, ids: list[str], *, rank_offset: int = 0
) -> list[dict]:
    """Look up each id in the store and shape it into the search-result dict.

    ``rank_offset`` lets paginated callers continue numbering across pages.
    """
    if not ids:
        return []
    rows = store.fetch_for_hydration(ids)
    by_id = {r["id"]: r for r in rows}
    out: list[dict] = []
    for ix, rid in enumerate(ids):
        r = by_id.get(rid)
        if r is None:
            continue
        subject_raw = r["subject"] or ""
        snippet_raw = (r["body"] or "")[:240]
        if looks_like_injection(subject_raw + "\n" + snippet_raw):
            snippet = "[redacted: possible injection]"
        else:
            snippet = escape_text(snippet_raw)
        out.append(
            {
                "id": rid,
                "subject": escape_text(subject_raw),
                "snippet": snippet,
                "language": r["language"],
                "queue": r["queue"],
                "priority": r["priority"],
                "score_rank": rank_offset + ix + 1,
            }
        )
    return out


def hybrid_search(
    store: TicketStore,
    *,
    query: str,
    filters: TicketFilters,
    embedder: Callable[[list[str]], np.ndarray],
    limit: int = 10,
    candidate_k: int = 50,
) -> list[dict]:
    """Thin wrapper: fuse, slice to ``limit``, hydrate."""
    fused = hybrid_search_full(
        store,
        query=query,
        filters=filters,
        embedder=embedder,
        candidate_k=candidate_k,
    )
    return hydrate_ids(store, fused[:limit])
