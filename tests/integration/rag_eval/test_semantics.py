"""Topical intent, cross-lingual recall, and hard-negative robustness."""

from __future__ import annotations

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder

from tests.integration.rag_eval._scenarios import (
    CROSS_LINGUAL_SCENARIOS,
    HARD_NEGATIVE_SCENARIOS,
    TOPICAL_SCENARIOS,
)
from tests.integration.rag_eval.conftest import hit_text, search_hits


def test_topical_intent_queries_in_top_3(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """>=16/20 topical intent queries surface at least one keyword in top-3."""
    matches: list[bool] = []
    misses: list[str] = []
    for query, keywords, _k in TOPICAL_SCENARIOS:
        hits = search_hits(eval_store, real_embedder.embed_queries, query, limit=3)
        matched = any(any(kw in hit_text(h) for kw in keywords) for h in hits)
        matches.append(matched)
        if not matched:
            misses.append(
                f"  MISS top-3 | query={query!r} | keywords={keywords} | "
                f"texts={[hit_text(h)[:80] for h in hits]}"
            )
    n_pass = sum(matches)
    record_summary("topical_intent_top3_pass_rate", n_pass / len(TOPICAL_SCENARIOS))
    assert n_pass >= 16, (
        f"topical intent top-3: {n_pass}/20 scenarios passed (need >=16).\n"
        + "\n".join(misses)
    )


def test_topical_intent_queries_in_top_10(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """>=17/20 topical intent queries surface at least one keyword in top-10."""
    matches: list[bool] = []
    misses: list[str] = []
    for query, keywords, _k in TOPICAL_SCENARIOS:
        hits = search_hits(eval_store, real_embedder.embed_queries, query, limit=10)
        matched = any(any(kw in hit_text(h) for kw in keywords) for h in hits)
        matches.append(matched)
        if not matched:
            misses.append(
                f"  MISS top-10 | query={query!r} | keywords={keywords} | "
                f"texts={[hit_text(h)[:80] for h in hits[:3]]}"
            )
    n_pass = sum(matches)
    record_summary("topical_intent_top10_pass_rate", n_pass / len(TOPICAL_SCENARIOS))
    assert n_pass >= 17, (
        f"topical intent top-10: {n_pass}/20 scenarios passed (need >=17).\n"
        + "\n".join(misses)
    )


def test_cross_lingual_recall_diagnostic(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """Cross-lingual recall — diagnostic only, no pass/fail bar.

    n=6 is too small to be a metric. Records the observed pass-rate to the
    summary so regressions or improvements are visible, but does not assert.
    Promote to a real test when the scenario list grows to ~50 pairs.
    """
    pass_count = 0
    for query, target_lang, keywords in CROSS_LINGUAL_SCENARIOS:
        hits = search_hits(eval_store, real_embedder.embed_queries, query, limit=10)
        target_hits = [h for h in hits if h.get("language") == target_lang]
        if any(any(kw in hit_text(h) for kw in keywords) for h in target_hits):
            pass_count += 1
    record_summary("cross_lingual_pass_rate", pass_count / len(CROSS_LINGUAL_SCENARIOS))


def test_hard_negatives_not_in_top_3(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    record_summary,
) -> None:
    """Top-3 must not be dominated by pure off-topic hits (<=2 total)."""
    total = 0
    detail: list[str] = []
    for query, topic_a, topic_b in HARD_NEGATIVE_SCENARIOS:
        hits = search_hits(eval_store, real_embedder.embed_queries, query, limit=3)
        for h in hits:
            text = hit_text(h)
            has_b = any(kw in text for kw in topic_b)
            has_a = any(kw in text for kw in topic_a)
            if has_b and not has_a:
                total += 1
                detail.append(
                    f"  CONTAMINATION | query={query!r} hit={text[:100]!r}"
                )
    record_summary("hard_negative_contamination_count", total)
    assert total <= 2, (
        f"hard negatives: {total} pure-off-topic hits in top-3 (limit=2).\n"
        + "\n".join(detail)
    )
