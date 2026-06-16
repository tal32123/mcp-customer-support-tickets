"""Polars-based group-by counts over a TicketStore."""

from __future__ import annotations

import polars as pl

from .store import TicketStore
from ..errors import ErrorCode, McpCstError


GROUP_BY_FIELDS = {"queue", "priority", "language", "type", "tags"}
FILTER_SCALAR_FIELDS = {"queue", "priority", "language", "type"}


def _apply_filters(df: pl.DataFrame, filters: dict) -> pl.DataFrame:
    tags = filters.get("tags")
    tags_mode = filters.get("tags_mode", "and")

    if tags_mode not in {"and", "or"}:
        raise McpCstError(ErrorCode.UNSUPPORTED_FILTER, f"tags_mode must be 'and' or 'or', got {tags_mode!r}")

    for key, value in filters.items():
        if key in {"tags", "tags_mode"}:
            continue
        if key not in FILTER_SCALAR_FIELDS:
            raise McpCstError(ErrorCode.UNSUPPORTED_FILTER, f"unsupported filter field: {key}")
        df = df.filter(pl.col(key) == value)

    if tags:
        if tags_mode == "and":
            for t in tags:
                df = df.filter(pl.col("tags").list.contains(t))
        else:  # or
            df = df.filter(
                pl.col("tags").list.eval(pl.element().is_in(tags)).list.any()
            )
    return df


def group_count(store: TicketStore, *, group_by: str, filters: dict) -> list[dict]:
    if group_by not in GROUP_BY_FIELDS:
        raise McpCstError(
            ErrorCode.UNSUPPORTED_GROUP_BY,
            f"group_by must be one of {sorted(GROUP_BY_FIELDS)}, got {group_by!r}",
        )

    arr = store.table.to_arrow()
    df = pl.from_arrow(arr)
    df = _apply_filters(df, filters)

    if group_by == "tags":
        df = df.explode("tags").filter(pl.col("tags").is_not_null() & (pl.col("tags") != ""))

    counts = (
        df.group_by(group_by)
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
    )
    return [{"group": row[group_by], "count": int(row["count"])} for row in counts.to_dicts()]
