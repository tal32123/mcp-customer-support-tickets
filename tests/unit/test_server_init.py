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
        server._CFG, server._STORE, server._EMBED_PASSAGES, server._EMBED_QUERIES,
        server._LLM_CLIENT,
    )
    server._CFG = None
    server._STORE = None
    server._EMBED_PASSAGES = None
    server._EMBED_QUERIES = None
    server._LLM_CLIENT = None
    yield
    (
        server._CFG, server._STORE, server._EMBED_PASSAGES, server._EMBED_QUERIES,
        server._LLM_CLIENT,
    ) = saved


def test_make_embedders_loads_model_once(reset_globals):
    """B2: the SentenceTransformer is constructed exactly once, even when
    both passage and query callables are invoked many times."""
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as ctor:
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


def test_init_is_idempotent(reset_globals, tmp_path, monkeypatch):
    """M1: calling _init() repeatedly is safe and does no extra work
    beyond the first call."""
    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path / "nonexistent"))

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    fake_store = MagicMock()

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as ctor, \
         patch("mcp_cst.server.build_store_from_huggingface", return_value=fake_store) as build_fn:
        server._init()
        server._init()
        server._init()

    # Model and store each built exactly once.
    assert ctor.call_count == 1
    assert build_fn.call_count == 1


def test_init_opens_existing_store_without_rebuild(reset_globals, tmp_path, monkeypatch, raw_ticket_rows):
    """H1: when a valid store exists at cfg.store_path, _init() must open it
    rather than re-download and rebuild from HuggingFace. This is the
    production restart hot-path."""
    from mcp_cst.data.store import TicketStore

    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path))

    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)

    def fixture_embedder(texts):
        return np.zeros((len(texts), 384), dtype=np.float32)

    # Pre-build a valid store at the exact path Config.from_env will compute.
    cfg = server.Config.from_env()
    TicketStore.create(
        path=cfg.store_path, revision=cfg.dataset_revision,
        rows=raw_ticket_rows[:10], embedder=fixture_embedder,
    )

    with patch("sentence_transformers.SentenceTransformer", return_value=fake_model), \
         patch("mcp_cst.server.build_store_from_huggingface") as build_fn:
        server._init()

    build_fn.assert_not_called()
    assert server._STORE is not None
    assert server._STORE.row_count() == 10


def test_llm_client_raises_when_no_provider_configured(reset_globals, tmp_path, monkeypatch):
    """The NO_LLM_CONFIGURED error path on _llm_client()."""
    from mcp_cst.errors import ErrorCode, McpCstError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path / "nonexistent"))

    with pytest.raises(McpCstError) as exc:
        server._llm_client()
    assert exc.value.code == ErrorCode.NO_LLM_CONFIGURED


def test_llm_client_caches_singleton(reset_globals, tmp_path, monkeypatch):
    """H3: the SDK client must be constructed once and reused.

    Otherwise each draft_reply call would open a fresh HTTP connection pool.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path / "nonexistent"))

    with patch("mcp_cst.server.AnthropicClient") as ctor:
        ctor.return_value = MagicMock()
        first = server._llm_client()
        second = server._llm_client()
        third = server._llm_client()

    assert first is second is third
    assert ctor.call_count == 1
