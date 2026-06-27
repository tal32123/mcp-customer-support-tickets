"""draft_reply grounding coherence: selected grounding shares the target's
language and type at or above the documented floor."""

from __future__ import annotations

import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder

from tests.eval.metrics import average
from tests.integration.rag_eval.conftest import sample_grounding_targets


# field, threshold, summary label
_DIMENSIONS = [
    ("type", 0.60, "grounding_type_coherence"),
    ("language", 0.95, "grounding_lang_coherence"),
]


@pytest.mark.parametrize("field,threshold,summary_key", _DIMENSIONS)
def test_draft_reply_grounding_coherence(
    field: str,
    threshold: float,
    summary_key: str,
    eval_store: TicketStore,
    real_embedder: SentenceTransformerEmbedder,
    sampled_rows: list[dict],
    store_ids_by_row_index: list[str],
    record_summary,
) -> None:
    """For a 20-ticket sample, grounding docs match the target's `field` at
    >= threshold. Type=0.60 (4 coarse types, hybrid retrieval crosses them);
    language=0.95 (server explicitly prefers same-language candidates).
    """
    pairs = sample_grounding_targets(
        eval_store,
        real_embedder.embed_queries,
        sampled_rows,
        store_ids_by_row_index,
        n=20,
    )
    assert pairs, "no usable grounding pairs — check sampled_rows content"

    per_target_match: list[float] = []
    for target_row, grounding_ids in pairs:
        target_value = (target_row.get(field) or "").strip()
        if not target_value:
            continue
        matching = sum(
            1
            for gid in grounding_ids
            if (rec := eval_store.get(gid)) is not None
            and (getattr(rec, field) or "").strip() == target_value
        )
        per_target_match.append(matching / len(grounding_ids))

    assert per_target_match, (
        f"all targets skipped (missing {field}) — sample_size={len(pairs)}"
    )
    avg = average(per_target_match)
    record_summary(summary_key, avg)
    assert avg >= threshold, (
        f"draft_reply {field} coherence: expected >= {threshold:.0%}, "
        f"got {avg:.2%} (evaluated={len(per_target_match)})"
    )
