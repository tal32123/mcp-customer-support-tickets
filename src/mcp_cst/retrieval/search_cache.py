"""In-memory cache of fused id lists for cursor-paginated search.

Long agent sweeps ("find every refund-flow ticket in EN") need stable
pagination across many MCP calls. Re-running RRF for every page would be
both slow and non-deterministic if the underlying table shifted under us.
We cache the full fused id list once per (q + filters) and slice from it.

Cache policy: dict keyed by `search_id`, value is `(timestamp, ids)`.
LRU-by-age with a hard size ceiling — on insert we drop expired entries,
then if still over `_SEARCH_CACHE_MAX` we evict the oldest. Good enough
for a single-process MCP server; if multi-worker becomes a thing, swap
this for an external store.
"""

from __future__ import annotations
import hashlib
import json
import threading
import time
from typing import Any

from ..errors import ErrorCode, McpCstError


_SEARCH_CACHE: dict[str, tuple[float, list[str]]] = {}
_SEARCH_CACHE_MAX = 64
_SEARCH_CACHE_TTL_S = 15 * 60
# Guards _SEARCH_CACHE under concurrent dispatch (stdio is serial today,
# but HTTP / streamable-http transports run handlers in parallel).
_CACHE_LOCK = threading.Lock()


def compute_search_id(q: str, filters: dict[str, Any]) -> str:
    """Stable 16-hex-char id for a (query, filters) pair.

    `sort_keys=True` makes filter dict order irrelevant; the tags list is
    additionally sorted because `tags=['a','b']` and `tags=['b','a']` are
    logically identical filters (both compile to LanceDB array_has_all/any,
    which is set-semantic) but JSON preserves list order — without this
    normalization the same query would miss the cache.
    """
    normalized: dict[str, Any] = dict(filters)
    if isinstance(normalized.get("tags"), list):
        normalized["tags"] = sorted(normalized["tags"])
    payload = json.dumps({"q": q, "filters": normalized}, sort_keys=True)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def encode_cursor(search_id: str, offset: int) -> str:
    return f"{search_id}:{offset}"


def decode_cursor(cursor: str) -> tuple[str, int]:
    """Parse `cursor` into (search_id, offset). Raise INVALID_INPUT on garbage."""
    try:
        sid, offset_str = cursor.split(":", 1)
        offset = int(offset_str)
    except (ValueError, AttributeError):
        raise McpCstError(ErrorCode.INVALID_INPUT, f"malformed cursor: {cursor!r}")
    if offset < 0:
        raise McpCstError(ErrorCode.INVALID_INPUT, f"malformed cursor: {cursor!r}")
    return sid, offset


def _evict_expired(now: float) -> None:
    expired = [k for k, (ts, _) in _SEARCH_CACHE.items() if now - ts > _SEARCH_CACHE_TTL_S]
    for k in expired:
        del _SEARCH_CACHE[k]


def cache_put(search_id: str, ids: list[str]) -> None:
    """Store a fused id list. Evicts expired entries; if still over capacity,
    drops the oldest by timestamp."""
    with _CACHE_LOCK:
        now = time.time()
        _evict_expired(now)
        _SEARCH_CACHE[search_id] = (now, ids)
        if len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX:
            oldest_key = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
            del _SEARCH_CACHE[oldest_key]


def cache_get(search_id: str) -> list[str] | None:
    """Return cached ids if present and not expired; otherwise None."""
    with _CACHE_LOCK:
        entry = _SEARCH_CACHE.get(search_id)
        if entry is None:
            return None
        ts, ids = entry
        if time.time() - ts > _SEARCH_CACHE_TTL_S:
            del _SEARCH_CACHE[search_id]
            return None
        return ids


def cache_clear() -> None:
    """Test helper — wipe the cache so per-test state never leaks."""
    with _CACHE_LOCK:
        _SEARCH_CACHE.clear()
