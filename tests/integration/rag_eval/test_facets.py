"""Language purity, filter pushdown, and cursor pagination — behavioural
checks against the real embedder + HF dataset."""

from __future__ import annotations

import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder
from mcp_cst.retrieval import search_cache
from mcp_cst.tools.search_tickets import search_tickets_impl

from tests.integration.rag_eval.conftest import hit_languages


def test_language_purity_german_query(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """German free-text query: >= 90% of top-10 hits must be German."""
    q = "Anmeldung Passwort"
    langs = hit_languages(eval_store, real_embedder.embed_queries, q, limit=10)
    assert langs, f"no hits for German query {q!r}"
    de_fraction = langs.count("de") / len(langs)
    record_summary("lang_purity_de", de_fraction)
    assert de_fraction >= 0.90, (
        f"German query {q!r}: expected >= 90% German results in top-10, "
        f"got {de_fraction:.2%} — languages: {langs}"
    )


def test_language_purity_english_query(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """English free-text query: >= 70% of top-10 hits must be English.

    Threshold is 0.70 rather than 0.90 because the HF dataset is ~54% DE /
    ~46% EN and short generic queries ("login", "problem", "password") include
    English loanwords common in German IT-support text. A 30% cross-language
    leak for these loanword queries is expected; the assertion is that the
    majority of hits are English, not that it equals German purity.
    """
    q = "login problem reset password"
    langs = hit_languages(eval_store, real_embedder.embed_queries, q, limit=10)
    assert langs, f"no hits for English query {q!r}"
    en_fraction = langs.count("en") / len(langs)
    record_summary("lang_purity_en", en_fraction)
    assert en_fraction >= 0.70, (
        f"English query {q!r}: expected >= 70% English results in top-10, "
        f"got {en_fraction:.2%} — languages: {langs}"
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

    while True:
        search_cache.cache_clear()
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
