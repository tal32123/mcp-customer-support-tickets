"""Shared fixtures, helpers, and summary hook for the real-embedder RAG eval.

Gated by MCP_CST_EVAL_FULL=1 (set on the package so every test under
tests/integration/rag_eval/ skips together when the env var is absent).
Distinct from MCP_CST_INTEGRATION so each tier opts in independently.

Run:
    $env:MCP_CST_EVAL_FULL="1"; uv run pytest tests/integration/rag_eval -q -s

Expected wall time: ~5-10 minutes on CPU (model download ~470MB on first run).

Dataset notes (as of revision "main"):
  - The HF dataset (Tobi-Bueck/customer-support-tickets) contains only "de"
    and "en" rows. There are no Hebrew rows despite the spec targeting he.
    Per-language and purity tests skip when a language bucket is empty.
  - The dataset "version" column is int, not str. Rows are coerced to str so
    the PyArrow schema does not reject them.
  - The dataset has tag_1..tag_8; the store schema uses tag_1..tag_6. Extra
    columns are silently ignored by TicketStore.create_with_rows.
"""

from __future__ import annotations

import os
import random
import re
from collections import defaultdict
from typing import Callable

import numpy as np
import pytest

from mcp_cst.data.store import TicketStore
from mcp_cst.embedder import SentenceTransformerEmbedder
from mcp_cst.errors import ErrorCode, McpCstError
from mcp_cst.prompts.draft_reply import draft_reply_impl
from mcp_cst.retrieval import search_cache
from mcp_cst.tools.search_tickets import search_tickets_impl


# ponytail: pytestmark in conftest is a no-op; apply the skip+integration marks
# to every collected item in this package via the hook below.
_SKIP = pytest.mark.skipif(
    "MCP_CST_EVAL_FULL" not in os.environ,
    reason="set MCP_CST_EVAL_FULL=1 to run full RAG eval (downloads HF data + ~470MB model, ~5-10 min)",
)


def pytest_collection_modifyitems(config, items) -> None:
    for item in items:
        if "rag_eval" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.integration)
            item.add_marker(_SKIP)

_N_SEEDS = 500
_LANG_SEED_TARGETS: dict[str, int] = {"en": 250, "de": 250}
_REVISION = "main"
_MODEL_NAME = "intfloat/multilingual-e5-small"

_WORD_RE = re.compile(r"\w{4,}", re.UNICODE)


def _body_slice(body: str, words: int = 12) -> str | None:
    """First contiguous run of `words` 4+-char tokens; None if too sparse."""
    found = _WORD_RE.findall(body or "")
    if len(found) < words:
        return None
    return " ".join(found[:words])


def _coerce_row(row: dict) -> dict:
    """Stringify all values; the HF dataset ships int `version` and None tags."""
    return {k: (str(v) if v is not None else "") for k, v in row.items()}


def _stratified_sample(
    rows: list[dict],
    lang_targets: dict[str, int],
    rng: random.Random,
) -> list[dict]:
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        lang = row.get("language", "")
        if lang in lang_targets:
            by_lang[lang].append(row)
    sampled: list[dict] = []
    for lang, target in lang_targets.items():
        bucket = list(by_lang[lang])
        rng.shuffle(bucket)
        sampled.extend(bucket[:target])
    return sampled


@pytest.fixture(scope="package")
def real_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(_MODEL_NAME)


@pytest.fixture(scope="package")
def eval_store(
    pg_dsn: str,
    real_embedder: SentenceTransformerEmbedder,
) -> TicketStore:
    """Open the existing pre-seeded ticket store (no DROP / no re-ingest).

    The eval runs against whatever is already in ``pg_dsn`` — typically
    a Railway-hosted pgvector with the full 62k-row HF dataset ingested.
    Set TEST_DATABASE_URL to point the suite at any pg you want.
    """
    schema = os.environ.get("MCP_CST_DB_SCHEMA", "public")
    return TicketStore.connect(
        dsn=pg_dsn,
        schema=schema,
        revision=_REVISION,
        embedding_dim=real_embedder.dim,
    )


@pytest.fixture(scope="package")
def store_ids_by_row_index(eval_store: TicketStore) -> list[str]:
    """Primary ids in row-index order; gold-id lookup for known-item."""
    return eval_store.all_ids()


@pytest.fixture(scope="package")
def sampled_rows(eval_store: TicketStore) -> list[dict]:
    """All rows from the live store in row_index order."""
    import psycopg
    from psycopg.rows import dict_row
    from psycopg import sql as psql

    schema = eval_store.schema_name
    with psycopg.connect(os.environ["TEST_DATABASE_URL"]) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                psql.SQL(
                    "SELECT subject, body, answer, type, queue, priority, "
                    "language, version, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6 "
                    "FROM {}.tickets ORDER BY row_index"
                ).format(psql.Identifier(schema))
            )
            return [dict(r) for r in cur.fetchall()]


@pytest.fixture(scope="package")
def eval_seeds(
    sampled_rows: list[dict],
    store_ids_by_row_index: list[str],
) -> list[dict]:
    """Stratified known-item seeds (~500, split en/de). Each carries the
    real primary id read back from the store, not a recomputed derive_id."""
    rng = random.Random(42)
    by_lang_idx: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for ix, row in enumerate(sampled_rows):
        lang = row.get("language", "")
        if lang in _LANG_SEED_TARGETS:
            by_lang_idx[lang].append((ix, row))

    seeds: list[dict] = []
    for lang, target in _LANG_SEED_TARGETS.items():
        bucket = list(by_lang_idx[lang])
        rng.shuffle(bucket)
        lang_count = 0
        for ix, row in bucket:
            if lang_count >= target:
                break
            bq = _body_slice(row.get("body", ""))
            if bq is None:
                continue
            seeds.append(
                {
                    "id_index": ix,
                    "id": store_ids_by_row_index[ix],
                    "subject_query": row.get("subject", ""),
                    "body_query": bq,
                    "language": lang,
                    "queue": row.get("queue", ""),
                    "type": row.get("type", ""),
                    "answer": row.get("answer", "") or "",
                }
            )
            lang_count += 1
    return seeds[:_N_SEEDS]


# ---------------------------------------------------------------------------
# Helpers shared across the eval files
# ---------------------------------------------------------------------------


def ranked_ids(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    q: str,
    limit: int = 10,
    **kwargs,
) -> list[str]:
    search_cache.cache_clear()
    result = search_tickets_impl(store, embedder, q=q, limit=limit, **kwargs)
    return [h["id"] for h in result["hits"]]


def hit_languages(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    q: str,
    limit: int = 10,
) -> list[str]:
    search_cache.cache_clear()
    result = search_tickets_impl(store, embedder, q=q, limit=limit)
    return [h["language"] for h in result["hits"]]


def search_hits(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    query: str,
    limit: int,
    **kwargs,
) -> list[dict]:
    search_cache.cache_clear()
    return search_tickets_impl(store, embedder, q=query, limit=limit, **kwargs)["hits"]


def hit_text(hit: dict) -> str:
    return (hit.get("subject", "") + " " + hit.get("snippet", "")).lower()


def sample_grounding_targets(
    store: TicketStore,
    embedder: Callable[[list[str]], np.ndarray],
    sampled_rows: list[dict],
    store_ids: list[str],
    n: int = 20,
) -> list[tuple[dict, list[str]]]:
    """Pick `n` candidate (target_row, grounding_ids) pairs.

    Skips rows with empty answers, language buckets <5, and draft_reply
    NO_GROUNDING_AVAILABLE failures. Deterministic via Random(42).
    """
    rng = random.Random(42)
    by_lang: dict[str, list[int]] = defaultdict(list)
    for ix, row in enumerate(sampled_rows):
        by_lang[row.get("language", "")].append(ix)

    candidates = [
        ix
        for ix, row in enumerate(sampled_rows)
        if (row.get("answer") or "").strip()
        and len(by_lang.get(row.get("language", ""), [])) >= 5
    ]
    rng.shuffle(candidates)

    out: list[tuple[dict, list[str]]] = []
    for ix in candidates:
        if len(out) >= n:
            break
        target_row = sampled_rows[ix]
        try:
            result = draft_reply_impl(
                store,
                embedder,
                ticket_id=store_ids[ix],
                target_language=target_row.get("language") or None,
            )
        except McpCstError as exc:
            if exc.code == ErrorCode.NO_GROUNDING_AVAILABLE:
                continue
            raise
        gids = result["grounding_ids"]
        if gids:
            out.append((target_row, gids))
    return out


# ---------------------------------------------------------------------------
# Summary table — pytest_terminal_summary hook replaces the zzzz print test.
# Tests append (label, value) tuples via the `record_summary` fixture; the
# hook reads from session.config._rag_eval_summary at the end of the run.
# ---------------------------------------------------------------------------


@pytest.fixture
def record_summary(request) -> Callable[[str, float], None]:
    bag = request.config.__dict__.setdefault("_rag_eval_summary", {})

    def _record(label: str, value: float) -> None:
        bag[label] = value

    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    bag = getattr(config, "_rag_eval_summary", None)
    if not bag:
        return
    tr = terminalreporter
    tr.write_sep("=", "RAG eval summary (MCP_CST_EVAL_FULL, e5-small)")
    for label, value in bag.items():
        if isinstance(value, float):
            tr.write_line(f"  {label:40s} {value:.4f}")
        else:
            tr.write_line(f"  {label:40s} {value}")
