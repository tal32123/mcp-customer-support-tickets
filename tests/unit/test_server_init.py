"""Tests for the eager-init wiring and the passage/query embed split (B1+B2+M1)."""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mcp_cst.server as server


@pytest.fixture
def reset_globals():
    """Snap server module globals so each test starts clean."""
    saved = (server._CFG, server._STORE, server._EMBED_PASSAGES, server._EMBED_QUERIES)
    server._CFG = None
    server._STORE = None
    server._EMBED_PASSAGES = None
    server._EMBED_QUERIES = None
    yield
    server._CFG, server._STORE, server._EMBED_PASSAGES, server._EMBED_QUERIES = saved


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
