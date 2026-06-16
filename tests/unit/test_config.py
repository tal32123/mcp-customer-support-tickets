import os
import pytest
from mcp_cst.config import Config, LlmProvider


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MCP_CST_DATASET_REVISION", raising=False)
    monkeypatch.delenv("MCP_CST_CACHE_DIR", raising=False)
    monkeypatch.delenv("RERANK", raising=False)
    cfg = Config.from_env()
    assert cfg.dataset_id == "Tobi-Bueck/customer-support-tickets"
    assert cfg.dataset_revision  # baked-in default
    assert cfg.embedding_model == "intfloat/multilingual-e5-small"
    assert cfg.rerank_enabled is False
    assert cfg.llm_provider is LlmProvider.NONE


def test_anthropic_preferred(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-x")
    cfg = Config.from_env()
    assert cfg.llm_provider is LlmProvider.ANTHROPIC


def test_openai_fallback(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-x")
    cfg = Config.from_env()
    assert cfg.llm_provider is LlmProvider.OPENAI


def test_cache_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path))
    cfg = Config.from_env()
    assert cfg.cache_root == tmp_path


def test_rerank_flag(monkeypatch):
    monkeypatch.setenv("RERANK", "true")
    assert Config.from_env().rerank_enabled is True
    monkeypatch.setenv("RERANK", "false")
    assert Config.from_env().rerank_enabled is False


def test_revision_override(monkeypatch):
    monkeypatch.setenv("MCP_CST_DATASET_REVISION", "main")
    assert Config.from_env().dataset_revision == "main"


def test_store_path_keyed_on_revision_and_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("MCP_CST_DATASET_REVISION", "abc123")
    cfg = Config.from_env()
    assert cfg.store_path == tmp_path / "abc123" / "multilingual-e5-small"
