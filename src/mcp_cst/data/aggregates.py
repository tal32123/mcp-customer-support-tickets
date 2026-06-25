"""Polars-based group-by counts over a TicketStore."""

from __future__ import annotations

import polars as pl

from .store import TicketStore
from ..errors import ErrorCode, McpCstError
from ..retrieval.hybrid import TicketFilters


GROUP_BY_FIELDS = {"queue", "priority", "language", "type", "tags"}
FILTER_SCALAR_FIELDS = {"queue", "priority", "language", "type"}

# Columns aggregation actually needs. Skipping `vector` (62k * 384 floats =
# ~95MB) and other unused columns keeps memory + Arrow->Polars conversion
# bounded on the full 62k corpus.
_AGG_COLUMNS = ["id", "queue", "priority", "language", "type", "tags"]


def _apply_filters(df: pl.DataFrame, filters: TicketFilters) -> pl.DataFrame:
    tags = filters.get("tags")
    tags_mode = filters.get("tags_mode", "and")

    if tags_mode not in {"and", "or"}:
        raise McpCstError(
            ErrorCode.UNSUPPORTED_FILTER,
            f"tags_mode must be 'and' or 'or', got {tags_mode!r}",
        )

    for key, value in filters.items():
        if key in {"tags", "tags_mode"}:
            continue
        if key not in FILTER_SCALAR_FIELDS:
            raise McpCstError(
                ErrorCode.UNSUPPORTED_FILTER, f"unsupported filter field: {key}"
            )
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


# `(id(table), write_seq)` → Polars DF. write_seq increments on every
# mutation, so update_ticket (delete+insert, row_count unchanged) still
# busts the cache. row_count alone would miss that case.
_MATERIALIZE_CACHE: dict[tuple[int, int], pl.DataFrame] = {}
_MATERIALIZE_CACHE_MAX = 32


def _materialize(store: TicketStore) -> pl.DataFrame:
    key = (id(store.table), store.write_seq)
    cached = _MATERIALIZE_CACHE.get(key)
    if cached is not None:
        return cached
    arr = store.table.search().select(_AGG_COLUMNS).to_arrow()
    df = pl.from_arrow(arr)
    if len(_MATERIALIZE_CACHE) >= _MATERIALIZE_CACHE_MAX:
        _MATERIALIZE_CACHE.pop(next(iter(_MATERIALIZE_CACHE)))
    _MATERIALIZE_CACHE[key] = df
    return df


def group_count(
    store: TicketStore, *, group_by: str, filters: TicketFilters
) -> list[dict]:
    if group_by not in GROUP_BY_FIELDS:
        raise McpCstError(
            ErrorCode.UNSUPPORTED_GROUP_BY,
            f"group_by must be one of {sorted(GROUP_BY_FIELDS)}, got {group_by!r}",
        )

    # `search().select(cols).to_arrow()` is LanceDB's column-projection
    # path -- skips reading the 384-dim `vector` column from disk.
    df = _materialize(store)
    df = _apply_filters(df, filters)

    if group_by == "tags":
        df = df.explode("tags").filter(
            pl.col("tags").is_not_null() & (pl.col("tags") != "")
        )

    counts = (
        df.group_by(group_by)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    return [
        {"group": row[group_by], "count": int(row["count"])}
        for row in counts.to_dicts()
    ]
