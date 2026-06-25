"""End-to-end test with the real `multilingual-e5-small` embedder.

Skipped by default because it downloads ~470MB of model weights from
Hugging Face. Run locally with `MCP_CST_INTEGRATION=1 uv run pytest -q
tests/integration` (or just `-m integration`). CI does NOT run this — the
`integration` marker is declared in `pyproject.toml` so `--strict-markers`
won't reject the decorator on collection.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder
from mcp_cst.prompts.draft_reply import draft_reply_impl
from mcp_cst.tools.search_tickets import search_tickets_impl

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        "MCP_CST_INTEGRATION" not in os.environ,
        reason="set MCP_CST_INTEGRATION=1 to run integration tests (downloads ~470MB model)",
    ),
]


def _pick_seed_rows(raw: list[dict], k: int = 30) -> list[dict]:
    """Pick `k` rows that contain at least a few Hebrew, German, and English
    samples so the language filter has real signal to find."""
    by_lang: dict[str, list[dict]] = {}
    for r in raw:
        by_lang.setdefault(r["language"], []).append(r)
    take_per = max(k // len(by_lang), 1)
    seeded: list[dict] = []
    for lang in ("de", "he", "en"):
        if lang in by_lang:
            seeded.extend(by_lang[lang][:take_per])
    # Pad if rounding left us short. O(n) dedup via id set.
    seen = {r["id"] for r in seeded}
    for r in raw:
        if len(seeded) >= k:
            break
        if r["id"] not in seen:
            seeded.append(r)
            seen.add(r["id"])
    return seeded[:k]


def test_real_embedder_finds_german_login_ticket(
    tmp_path: Path, raw_ticket_rows: list[dict]
) -> None:
    """`search_tickets` with the German query "Anmeldung" must surface a
    German ticket above any English/Hebrew ticket. Then `draft_reply` against
    that ticket must produce a prompt that quotes the German subject verbatim.
    """
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    seed = _pick_seed_rows(raw_ticket_rows, k=30)

    store = TicketStore.create(
        path=tmp_path / "real-store",
        revision="integration",
        rows=seed,
        embedder=embedder.embed_passages,
        embedding_dim=embedder.dim,
    )

    result = search_tickets_impl(
        store,
        embedder.embed_queries,
        q="Anmeldung",
        limit=5,
    )
    hits = result["hits"]
    assert hits, "expected at least one search hit for 'Anmeldung'"
    assert any(h["language"] == "de" for h in hits[:3]), (
        f"expected at least one German ticket in top-3, got "
        f"languages={[h['language'] for h in hits[:3]]!r}"
    )

    de_hits = [h for h in hits if h["language"] == "de"]
    target_id = de_hits[0]["id"] if de_hits else hits[0]["id"]
    target_rec = store.get(target_id)
    assert target_rec is not None

    out = draft_reply_impl(
        store,
        embedder.embed_queries,
        ticket_id=target_id,
        target_language="de",
    )
    # The target ticket's German subject must appear verbatim in the assembled
    # prompt — confirming that ingest preserved the Unicode payload and that
    # wrap_ticket inlines it without mangling.
    assert target_rec.subject in out["prompt"]
