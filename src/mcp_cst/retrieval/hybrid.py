"""Hybrid BM25 + vector retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations
from typing import Callable

import numpy as np

from ..data.store import TicketStore
from ..errors import ErrorCode, McpCstError


FILTER_FIELDS = {"queue", "priority", "language", "type"}
_TAG_FILTER_KEYS = frozenset({"tags", "tags_mode"})


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Merge multiple ranked lists into one. Higher rank → higher score."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _build_where(filters: dict) -> str | None:
    """Translate filters dict into a LanceDB WHERE clause.

    Tag filters are NOT included here — they are applied as a post-filter
    in Python because LanceDB's list-contains support varies by version.
    """
    clauses: list[str] = []
    for key, value in filters.items():
        if key in _TAG_FILTER_KEYS:
            continue
        if key not in FILTER_FIELDS:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER, f"unsupported filter field: {key}"
            )
        # Escape single quotes
        safe = str(value).replace("'", "''")
        clauses.append(f"{key} = '{safe}'")
    return " AND ".join(clauses) if clauses else None


def _post_filter_tags(rows: list[dict], filters: dict) -> list[dict]:
    tags = filters.get("tags")
    mode = filters.get("tags_mode", "and")
    if not tags:
        return rows
    if mode not in {"and", "or"}:
        raise McpCstError(
            ErrorCode.UNSUPPORTED_FILTER, "tags_mode must be 'and' or 'or'"
        )
    pred = all if mode == "and" else any
    return [r for r in rows if pred(t in (r.get("tags") or []) for t in tags)]


def hybrid_search(
    store: TicketStore,
    *,
    query: str,
    filters: dict,
    embedder: Callable[[list[str]], np.ndarray],
    limit: int = 10,
    candidate_k: int = 50,
) -> list[dict]:
    where = _build_where(filters)

    # BM25 branch
    bm25_q = store.table.search(query, query_type="fts").limit(candidate_k)
    if where:
        bm25_q = bm25_q.where(where)
    bm25_rows = bm25_q.to_list()
    bm25_rows = _post_filter_tags(bm25_rows, filters)
    bm25_ids = [r["id"] for r in bm25_rows]

    # Vector branch
    qvec = embedder([query])[0].tolist()
    vec_q = store.table.search(qvec, query_type="vector").limit(candidate_k)
    if where:
        vec_q = vec_q.where(where)
    vec_rows = vec_q.to_list()
    vec_rows = _post_filter_tags(vec_rows, filters)
    vec_ids = [r["id"] for r in vec_rows]

    fused_ids = reciprocal_rank_fusion([bm25_ids, vec_ids])[:limit]

    by_id = {r["id"]: r for r in (*bm25_rows, *vec_rows)}
    out: list[dict] = []
    for ix, rid in enumerate(fused_ids):
        r = by_id.get(rid)
        if r is None:
            continue
        snippet = (r["body"] or "")[:240]
        out.append(
            {
                "id": rid,
                "subject": r["subject"],
                "snippet": snippet,
                "language": r["language"],
                "queue": r["queue"],
                "priority": r["priority"],
                "score_rank": ix + 1,
            }
        )
    return out
