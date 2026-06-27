"""Language purity, filter pushdown, and cursor pagination — behavioural
checks against the real embedder + HF dataset."""

from __future__ import annotations

import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder
from mcp_cst.retrieval import search_cache
from mcp_cst.tools.search_tickets import search_tickets_impl

from tests.integration.rag_eval._scenarios import (
    PURITY_QUERIES_DE,
    PURITY_QUERIES_EN,
)
from tests.integration.rag_eval.conftest import hit_languages


def _mean_purity(store, embedder, queries: list[str], target_lang: str) -> float:
    """Mean fraction of top-10 hits whose `language` equals `target_lang`,
    averaged across `queries`. Queries that return no hits are skipped."""
    fractions: list[float] = []
    for q in queries:
        langs = hit_languages(store, embedder, q, limit=10)
        if not langs:
            continue
        fractions.append(langs.count(target_lang) / len(langs))
    assert fractions, "no queries returned any hits"
    return sum(fractions) / len(fractions)


def test_language_purity_german(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """Mean German-purity across 30 free-text DE queries: >= 90%."""
    avg = _mean_purity(
        eval_store, real_embedder.embed_queries, PURITY_QUERIES_DE, "de"
    )
    record_summary("lang_purity_de", avg)
    assert avg >= 0.90, (
        f"DE purity: expected mean >= 90% across {len(PURITY_QUERIES_DE)} queries, "
        f"got {avg:.2%}"
    )


def test_language_purity_english(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """Mean English-purity across 30 free-text EN queries: >= 70%.

    Lower bar than DE because EN IT loanwords ("login", "password") appear in
    DE tickets, so naked EN queries leak. Filter pushdown (test_filter_pushdown)
    is the precision tool when callers need 100%.
    """
    avg = _mean_purity(
        eval_store, real_embedder.embed_queries, PURITY_QUERIES_EN, "en"
    )
    record_summary("lang_purity_en", avg)
    assert avg >= 0.70, (
        f"EN purity: expected mean >= 70% across {len(PURITY_QUERIES_EN)} queries, "
        f"got {avg:.2%}"
    )


@pytest.mark.parametrize("language", ["de", "he"])
def test_filter_pushdown_language(
    language: str,
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
) -> None:
    """With language= filter, every returned hit has that language."""
    lang_seeds = [s for s in eval_seeds if s["language"] == language]
    if not lang_seeds:
        pytest.skip(f"no seeds for language {language!r}")
    q = lang_seeds[0]["body_query"]

    search_cache.cache_clear()
    result = search_tickets_impl(
        eval_store,
        real_embedder.embed_queries,
        q=q,
        language=language,
        limit=10,
    )
    hits = result["hits"]
    assert hits, (
        f"filter_pushdown[{language}]: search returned no hits (query={q!r})"
    )
    wrong = [h for h in hits if h["language"] != language]
    assert not wrong, (
        f"filter_pushdown[{language}]: {len(wrong)}/{len(hits)} hits have "
        f"wrong language: {[h['language'] for h in wrong]}"
    )


def test_pagination_disjoint_pages(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
) -> None:
    """Walking all pages yields no duplicate ids and total equals total_estimate."""
    seen: set[str] = set()
    cursor = None
    total_estimate = None
    pages = 0

    # ponytail: cache_clear once before the walk; clearing per page invalidates the cursor.
    search_cache.cache_clear()
    while True:
        result = search_tickets_impl(
            eval_store,
            real_embedder.embed_queries,
            q="login account error",
            limit=5,
            cursor=cursor,
        )
        if total_estimate is None:
            total_estimate = result["total_estimate"]
        for h in result["hits"]:
            tid = h["id"]
            assert tid not in seen, f"duplicate id {tid!r} on page {pages + 1}"
            seen.add(tid)
        pages += 1
        cursor = result["next_cursor"]
        if cursor is None:
            break
        assert pages < 500, "pagination did not terminate within 500 pages"

    assert len(seen) == total_estimate, (
        f"collected {len(seen)} ids across {pages} pages but "
        f"total_estimate={total_estimate}"
    )
