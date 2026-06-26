"""Known-item retrieval eval for search_tickets.

Each seed is queried twice: once with its subject verbatim (wide recall check)
and once with its full body text (precision check). Produces hit-rate@k, MRR@10,
NDCG@10 on a known-relevance set without any LLM in the loop.

Fixture note: the conftest known_item_seeds fixture requires 12 four-char tokens
per body, but the synthetic fixture bodies are too short (~6 such tokens). Seeds
are built here directly from raw_ticket_rows using the full body as the query.

Threshold calibration note: the "(case #N)" suffix is unique per row, but
LanceDB's FTS tokenizer strips "#", so "case 0" matches all rows equally and
BM25 cannot discriminate at @10. Hit-rate@50 is 1.000 (body and subject both);
hit-rate@10 is ~0.33 because RRF tie-breaks among 20 equally-scoring siblings.
Thresholds are set against @50 where the lexical path is demonstrably correct,
and at calibrated-low values for @10.  The semantic bar lives in
tests/integration/test_rag_full_eval.py (real multilingual-e5-small embedder).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pytest

from mcp_cst.data.store import derive_id
from mcp_cst.retrieval import search_cache
from mcp_cst.tools.search_tickets import search_tickets_impl

from tests.eval.metrics import average, hit_rate_at_k, mrr, ndcg_at_k

# Revision hardcoded to match eval_store_session in conftest.py
_REVISION = "eval"
_PER_LANG = 15  # 15 seeds per language → 30 total on EN/DE synthetic fixture


@pytest.fixture(autouse=True)
def _clear_cache():
    """Per-test isolation — search_cache is module-level state."""
    search_cache.cache_clear()
    yield
    search_cache.cache_clear()


@pytest.fixture(scope="session")
def seeds(raw_ticket_rows: list[dict]) -> list[dict[str, Any]]:
    """Stratified seed list built from the 200-row synthetic fixture.

    Uses full body text as body_query. Each body is unique by the "(case #N)"
    suffix — though FTS cannot discriminate on the "#N" part, the full text
    phrase still gets the gold ticket into the top-50 BM25 result set.
    """
    by_lang: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for ix, row in enumerate(raw_ticket_rows):
        lang = row.get("language", "")
        if lang:
            by_lang[lang].append((ix, row))

    result: list[dict[str, Any]] = []
    for lang, items in by_lang.items():
        for ix, row in items[:_PER_LANG]:
            body = row.get("body", "").strip()
            if not body:
                continue
            result.append(
                {
                    "id_index": ix,
                    "subject_query": row["subject"],
                    "body_query": body,
                    "language": lang,
                }
            )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ranked_ids(store, embedder, q: str, limit: int = 50) -> list[str]:
    result = search_tickets_impl(store, embedder, q=q, limit=limit)
    return [h["id"] for h in result["hits"]]


def _gold_id(store, seed: dict) -> str:
    # ponytail: revision is hardcoded to match eval_store_session
    return derive_id(store.revision, seed["id_index"])


# ---------------------------------------------------------------------------
# Body-query hit-rate@50
#
# @50 is the primary body-query signal for this CI tier. The FTS tokenizer
# strips "#" so "(case #N)" does not discriminate at @10; but every gold ticket
# still lands in the top-50 because it shares all body tokens with the query.
# Observed: 1.000. Threshold 0.90 gives 10pp regression margin.
# ---------------------------------------------------------------------------

def test_body_hit_rate_at_50(eval_store_session, eval_embedder, seeds):
    """Gold ticket must appear in the 50-result cap when queried by full body."""
    assert seeds, "seed list must not be empty"
    scores = [
        hit_rate_at_k(
            _ranked_ids(eval_store_session, eval_embedder, seed["body_query"], limit=50),
            {_gold_id(eval_store_session, seed)},
            k=50,
        )
        for seed in seeds
    ]
    avg = average(scores)
    # calibrated: observed 1.000; slow tier hits 1.000 too but also asserts @10
    assert avg >= 0.90, f"body hit-rate@50: expected >= 0.90, got {avg:.3f}"


# ---------------------------------------------------------------------------
# Body-query @10 metrics (weak — FTS cannot discriminate among siblings)
#
# The hash embedder produces noise; BM25 tie-breaks among ~20 identical-scoring
# sibling tickets sharing the same template body. Observed: hit@10=0.33,
# MRR@10=0.19, NDCG@10=0.22. Thresholds are set well below observed values so
# a total BM25 failure (0.0) is caught, but day-to-day tie-break variance is not.
# ---------------------------------------------------------------------------

def test_body_hit_rate_at_10(eval_store_session, eval_embedder, seeds):
    """Body hit-rate@10 — catches total BM25 failure, not precision ranking."""
    assert seeds, "seed list must not be empty"
    scores = [
        hit_rate_at_k(
            _ranked_ids(eval_store_session, eval_embedder, seed["body_query"]),
            {_gold_id(eval_store_session, seed)},
            k=10,
        )
        for seed in seeds
    ]
    avg = average(scores)
    # calibrated for synthetic fixture: observed ~0.33; threshold catches BM25 breakage
    assert avg >= 0.20, f"body hit-rate@10: expected >= 0.20, got {avg:.3f}"


def test_body_mrr_at_10(eval_store_session, eval_embedder, seeds):
    """Body MRR@10 — calibrated for hash-embedder + synthetic fixture."""
    assert seeds, "seed list must not be empty"
    scores = [
        mrr(
            _ranked_ids(eval_store_session, eval_embedder, seed["body_query"])[:10],
            {_gold_id(eval_store_session, seed)},
        )
        for seed in seeds
    ]
    avg = average(scores)
    # calibrated: observed ~0.19; threshold catches complete rank degradation
    assert avg >= 0.10, f"body MRR@10: expected >= 0.10, got {avg:.3f}"


def test_body_ndcg_at_10(eval_store_session, eval_embedder, seeds):
    """Body NDCG@10 — calibrated for hash-embedder + synthetic fixture."""
    assert seeds, "seed list must not be empty"
    scores = [
        ndcg_at_k(
            _ranked_ids(eval_store_session, eval_embedder, seed["body_query"]),
            {_gold_id(eval_store_session, seed)},
            k=10,
        )
        for seed in seeds
    ]
    avg = average(scores)
    # calibrated: observed ~0.22; threshold catches complete rank degradation
    assert avg >= 0.12, f"body NDCG@10: expected >= 0.12, got {avg:.3f}"


# ---------------------------------------------------------------------------
# Subject hit-rate@50 — confirms lexical match path works
# ---------------------------------------------------------------------------

def test_subject_hit_rate_at_50(eval_store_session, eval_embedder, seeds):
    """Subject query -> gold id anywhere in the full 50-hit cap.

    Subjects repeat across ~20 rows each so precision @10 is meaningless.
    @50 confirms the lexical match path works. Observed: 1.000.
    """
    assert seeds, "seed list must not be empty"
    scores = [
        hit_rate_at_k(
            _ranked_ids(eval_store_session, eval_embedder, seed["subject_query"], limit=50),
            {_gold_id(eval_store_session, seed)},
            k=50,
        )
        for seed in seeds
    ]
    avg = average(scores)
    assert avg >= 0.90, f"subject hit-rate@50: expected >= 0.90, got {avg:.3f}"


# ---------------------------------------------------------------------------
# Per-language body hit-rate@50 — guards multilingual regressions
# ---------------------------------------------------------------------------

def test_per_language_body_hit_rate_at_50(eval_store_session, eval_embedder, seeds):
    """Each language must independently reach body hit-rate@50 >= 0.85.

    Observed: de=1.000, en=1.000. Threshold catches a language-specific
    BM25 index corruption or ingest regression.
    """
    assert seeds, "seed list must not be empty"
    by_lang: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        score = hit_rate_at_k(
            _ranked_ids(eval_store_session, eval_embedder, seed["body_query"], limit=50),
            {_gold_id(eval_store_session, seed)},
            k=50,
        )
        by_lang[seed["language"]].append(score)

    failures = []
    for lang, scores in sorted(by_lang.items()):
        avg = average(scores)
        if avg < 0.85:
            failures.append(f"{lang}: expected >= 0.85, got {avg:.3f}")

    assert not failures, (
        "per-language body hit-rate@50 failures:\n" + "\n".join(failures)
    )
