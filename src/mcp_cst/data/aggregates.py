"""SQL group-by counts over a TicketStore."""

from __future__ import annotations

from psycopg import sql

from .store import TicketStore
from ..errors import ErrorCode, McpCstError
from ..retrieval.hybrid import TicketFilters


GROUP_BY_FIELDS = {"queue", "priority", "language", "type", "tags"}
FILTER_SCALAR_FIELDS = {"queue", "priority", "language", "type"}


def _build_where(filters: TicketFilters) -> tuple[sql.Composable | None, tuple]:
    """Mirrors search_tickets' filter contract so counts match what search returns."""
    tags = filters.get("tags")
    tags_mode = filters.get("tags_mode", "and")
    if tags_mode not in {"and", "or"}:
        raise McpCstError(
            ErrorCode.UNSUPPORTED_FILTER,
            f"tags_mode must be 'and' or 'or', got {tags_mode!r}",
        )
    clauses: list[sql.Composable] = []
    params: list = []
    for key, value in filters.items():
        if key in {"tags", "tags_mode"}:
            continue
        if key not in FILTER_SCALAR_FIELDS:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER, f"unsupported filter field: {key}"
            )
        clauses.append(sql.SQL("{col} = %s").format(col=sql.Identifier(key)))
        params.append(value)
    if tags:
        non_empty = [t for t in tags if t]
        if non_empty:
            op = sql.SQL("@>") if tags_mode == "and" else sql.SQL("&&")
            clauses.append(sql.SQL("tags {op} %s").format(op=op))
            params.append(non_empty)
    if not clauses:
        return None, ()
    return sql.SQL(" AND ").join(clauses), tuple(params)


def group_count(
    store: TicketStore, *, group_by: str, filters: TicketFilters
) -> list[dict]:
    """Validate ``group_by``, push the count down to Postgres."""
    if group_by not in GROUP_BY_FIELDS:
        raise McpCstError(
            ErrorCode.UNSUPPORTED_GROUP_BY,
            f"group_by must be one of {sorted(GROUP_BY_FIELDS)}, got {group_by!r}",
        )
    where_sql, where_params = _build_where(filters)
    rows = store.group_count_query(
        group_by=group_by, where_sql=where_sql, where_params=where_params
    )
    return [{"group": r["group_value"], "count": int(r["cnt"])} for r in rows]
