"""Retriever robustness to surface-form noise.

For a 30-seed sample, perturb the body-slice query three ways (lowercase,
adjacent-char swap, drop-a-word) and assert the per-perturbation hit-rate@10
stays within 0.15 of the clean baseline. Guards against tokenizer regressions
and over-fit to exact-string matching.
"""

from __future__ import annotations

import random

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder

from tests.eval.metrics import average, hit_rate_at_k
from tests.integration.rag_eval.conftest import ranked_ids


_N_SEEDS = 30
_MAX_DROP = 0.15


def _lowercase(q: str, rng: random.Random) -> str:
    return q.lower()


def _swap_adjacent(q: str, rng: random.Random) -> str:
    if len(q) < 4:
        return q
    i = rng.randrange(len(q) - 1)
    return q[:i] + q[i + 1] + q[i] + q[i + 2 :]


def _drop_word(q: str, rng: random.Random) -> str:
    words = q.split()
    if len(words) < 3:
        return q
    j = rng.randrange(len(words))
    return " ".join(words[:j] + words[j + 1 :])


_PERTURBATIONS = [
    ("lowercase", _lowercase),
    ("swap_chars", _swap_adjacent),
    ("drop_word", _drop_word),
]


def test_robustness_perturbations(
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    eval_seeds: list[dict],
    record_summary,
) -> None:
    rng = random.Random(42)
    sample = eval_seeds[:_N_SEEDS]
    assert sample, "no seeds available"

    embedder = real_embedder.embed_queries
    clean_scores = [
        hit_rate_at_k(
            ranked_ids(eval_store, embedder, s["body_query"]),
            {s["id"]},
            k=10,
        )
        for s in sample
    ]
    clean_hr = average(clean_scores)
    record_summary("robustness_clean_hit_rate_10", clean_hr)

    failures: list[str] = []
    for name, fn in _PERTURBATIONS:
        scores = [
            hit_rate_at_k(
                ranked_ids(eval_store, embedder, fn(s["body_query"], rng)),
                {s["id"]},
                k=10,
            )
            for s in sample
        ]
        hr = average(scores)
        record_summary(f"robustness_{name}_hit_rate_10", hr)
        drop = clean_hr - hr
        if drop > _MAX_DROP:
            failures.append(
                f"{name}: clean={clean_hr:.3f} perturbed={hr:.3f} drop={drop:.3f}"
            )

    assert not failures, (
        "robustness regressions (drop > {:.2f}):\n  ".format(_MAX_DROP)
        + "\n  ".join(failures)
    )
