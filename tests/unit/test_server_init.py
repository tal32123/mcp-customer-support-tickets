"""Tests for the eager-init wiring and the passage/query embed split (B1+B2+M1)."""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mcp_cst.server as server


@pytest.fixture
def reset_globals():
    """Snap server module globals so each test starts clean."""
    saved = (
        server._CFG,
        server._STORE,
        server._EMBED_PASSAGES,
        server._EMBED_QUERIES,
        server._EMBED_THREAD,
    )
    server._CFG = None
    server._STORE = None
    server._EMBED_PASSAGES = None
    server._EMBED_QUERIES = None
    server._EMBED_THREAD = None
    yield
    # Make sure any background warm-up has finished before the next test
    # touches module globals.
    t = server._EMBED_THREAD
    if t is not None and t.is_alive():
        t.join(timeout=10)
    (
        server._CFG,
        server._STORE,
        server._EMBED_PASSAGES,
        server._EMBED_QUERIES,
        server._EMBED_THREAD,
    ) = saved


def test_make_embedders_loads_model_once(reset_globals):
    """B2: the SentenceTransformer is constructed exactly once, even when
    both passage and query callables are invoked many times."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    with patch(
        "sentence_transformers.SentenceTransformer", return_value=fake_model
    ) as ctor:
        embed_passages, embed_queries = server._make_embedders("test-model")
        embed_passages(["a", "b"])
        embed_queries(["c"])
        embed_passages(["d"])
        embed_queries(["e"])

    assert ctor.call_count == 1
    ctor.assert_called_with("test-model")


def test_passage_and_query_prefixes_differ(reset_globals):
    """B1: passages get 'passage: ' and queries get 'query: '."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((2, 384), dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model):
        embed_passages, embed_queries = server._make_embedders("test-model")
        embed_passages(["hello", "world"])
        embed_queries(["login broken"])

    encode_calls = fake_model.encode.call_args_list
    # First call -- passages.
    passage_inputs = encode_calls[0].args[0]
    assert passage_inputs == ["passage: hello", "passage: world"]
    # Second call -- queries.
    query_inputs = encode_calls[1].args[0]
    assert query_inputs == ["query: login broken"]


def test_init_is_idempotent(reset_globals, pg_dsn, monkeypatch):
    """M1: calling _init() repeatedly is safe and does no extra work
    beyond the first call."""
    import uuid

    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    monkeypatch.setenv("MCP_CST_DB_SCHEMA", f"test_{uuid.uuid4().hex[:12]}")

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    fake_store = MagicMock()

    with (
        patch(
            "sentence_transformers.SentenceTransformer", return_value=fake_model
        ) as ctor,
        patch(
            "mcp_cst.server.build_store_from_huggingface", return_value=fake_store
        ) as build_fn,
    ):
        server._init()
        server._init()
        server._init()

    # Model and store each built exactly once.
    assert ctor.call_count == 1
    assert build_fn.call_count == 1


def test_init_opens_existing_store_without_rebuild(
    reset_globals, pg_dsn, pg_schema, monkeypatch, raw_ticket_rows
):
    """H1: when a valid store already exists for the configured DSN+schema,
    _init() must open it rather than re-download and rebuild from
    HuggingFace. This is the production restart hot-path."""
    from mcp_cst.data.store import TicketStore

    monkeypatch.setenv("DATABASE_URL", pg_dsn)
    monkeypatch.setenv("MCP_CST_DB_SCHEMA", pg_schema)

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    def fixture_embedder(texts):
        return np.zeros((len(texts), 384), dtype=np.float32)

    # Pre-build a valid store so the meta marker passes is_valid().
    cfg = server.Config.from_env()
    TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision=cfg.dataset_revision,
        rows=raw_ticket_rows[:10],
        embedder=fixture_embedder,
    )

    with (
        patch("sentence_transformers.SentenceTransformer", return_value=fake_model),
        patch("mcp_cst.server.build_store_from_huggingface") as build_fn,
    ):
        server._init()

    build_fn.assert_not_called()
    assert server._STORE is not None
    assert server._STORE.row_count() == 10
