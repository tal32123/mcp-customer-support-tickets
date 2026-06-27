"""Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations
from typing import Callable, Literal, TypedDict

import numpy as np

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
    """Collect a filters dict, omitting None scalars and dropping tags_mode
    when tags is empty (it would be meaningless and trip strict validation)."""
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


def _quote_tag(t: str) -> str:
    # LanceDB SQL: wrap in single quotes, double up embedded single quotes.
    return "'" + t.replace("'", "''") + "'"


def _build_where(filters: TicketFilters) -> str | None:
    """Translate filters dict into a LanceDB WHERE clause.

    Scalar filters compile to `col = 'value'`. Tag filters compile to
    `array_has_all(tags, [...])` for tags_mode='and' (default) and
    `array_has_any(tags, [...])` for tags_mode='or' — pushed down so
    candidate_k applies AFTER the tag restriction, preserving recall.
    Empty tag values are skipped.
    """
    clauses: list[str] = []
    tags: list[str] | None = None
    tags_mode: str = "and"
    for key, value in filters.items():
        if key == "tags":
            tags = value
            continue
        if key == "tags_mode":
            tags_mode = value
            continue
        if key not in FILTER_FIELDS:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER, f"unsupported filter field: {key}"
            )
        safe = str(value).replace("'", "''")
        clauses.append(f"{key} = '{safe}'")

    if tags:
        if tags_mode not in {"and", "or"}:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER,
                f"tags_mode must be 'and' or 'or', got {tags_mode!r}",
            )
        quoted = [_quote_tag(t) for t in tags if t != ""]
        if quoted:
            fn = "array_has_all" if tags_mode == "and" else "array_has_any"
            clauses.append(f"{fn}(tags, [{', '.join(quoted)}])")

    return " AND ".join(clauses) if clauses else None


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
    where = _build_where(filters)

    # BM25 branch
    bm25_q = store.table.search(query, query_type="fts").limit(candidate_k)
    if where:
        bm25_q = bm25_q.where(where)
    bm25_ids = [r["id"] for r in bm25_q.to_list()]

    # Vector branch
    qvec = embedder([query])[0].tolist()
    vec_q = store.table.search(qvec, query_type="vector").limit(candidate_k)
    if where:
        vec_q = vec_q.where(where)
    vec_ids = [r["id"] for r in vec_q.to_list()]

    return reciprocal_rank_fusion([bm25_ids, vec_ids])


def hydrate_ids(
    store: TicketStore, ids: list[str], *, rank_offset: int = 0
) -> list[dict]:
    """Look up each id in the store and shape it into the search-result dict.

    `rank_offset` lets paginated callers continue numbering across pages
    (page 2 starts at rank `page_size + 1`).
    """
    if not ids:
        return []
    quoted = ", ".join(_quote_tag(i) for i in ids)
    rows = store.table.search().where(f"id IN ({quoted})").limit(len(ids)).to_list()
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
    """Thin wrapper: fuse, slice to `limit`, hydrate. Kept for callers that
    don't need pagination (draft_reply, direct lookups)."""
    fused = hybrid_search_full(
        store,
        query=query,
        filters=filters,
        embedder=embedder,
        candidate_k=candidate_k,
    )
    return hydrate_ids(store, fused[:limit])
