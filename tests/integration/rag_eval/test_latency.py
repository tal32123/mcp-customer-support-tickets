"""Per-query latency distribution for search_tickets — BEIR-style p50/p95.

Times the embedder + retrieval together (the latency the LLM client sees).
First call is warm-up to amortise lazy load; the assertion is on the remaining
runs. Bars are generous CPU-machine bars; tighten if a GPU runner appears.
"""

from __future__ import annotations

import statistics
import time

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder
from mcp_cst.retrieval import search_cache
from mcp_cst.tools.search_tickets import search_tickets_impl

from tests.integration.rag_eval._scenarios import (
    PURITY_QUERIES_DE,
    PURITY_QUERIES_EN,
)


_P50_BAR_SEC = 0.40
_P95_BAR_SEC = 1.00


def test_search_latency_p50_p95(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    queries = PURITY_QUERIES_EN + PURITY_QUERIES_DE
    # Warm-up; e5-small's first inference triggers torch graph compile.
    search_cache.cache_clear()
    search_tickets_impl(
        eval_store, real_embedder.embed_queries, q=queries[0], limit=10
    )

    timings: list[float] = []
    for q in queries:
        search_cache.cache_clear()
        t0 = time.perf_counter()
        search_tickets_impl(
            eval_store, real_embedder.embed_queries, q=q, limit=10
        )
        timings.append(time.perf_counter() - t0)

    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[int(0.95 * len(timings)) - 1]
    record_summary("latency_p50_sec", p50)
    record_summary("latency_p95_sec", p95)

    assert p50 <= _P50_BAR_SEC, (
        f"p50 latency {p50:.3f}s exceeds bar {_P50_BAR_SEC:.2f}s "
        f"(n={len(timings)})"
    )
    assert p95 <= _P95_BAR_SEC, (
        f"p95 latency {p95:.3f}s exceeds bar {_P95_BAR_SEC:.2f}s "
        f"(n={len(timings)})"
    )
