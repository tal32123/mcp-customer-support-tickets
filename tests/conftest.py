"""Shared pytest configuration and fixtures."""

from __future__ import annotations
import json
from pathlib import Path

import pytest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tickets.json"


@pytest.fixture(scope="session")
def raw_ticket_rows() -> list[dict]:
    """200 synthetic ticket rows, schema-identical to the HF dataset."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Override MCP_CST_CACHE_DIR for the duration of a test."""
    monkeypatch.setenv("MCP_CST_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _clear_create_ticket_idempotency_cache():
    """create_ticket keeps a process-global idempotency cache (#162). Tests
    each get a fresh store but share that cache, so without per-test isolation
    a cached id from one test points into another test's discarded store."""
    from mcp_cst.tools import create_ticket as _ct

    _ct._idempotency_cache.clear()
    yield
    _ct._idempotency_cache.clear()
