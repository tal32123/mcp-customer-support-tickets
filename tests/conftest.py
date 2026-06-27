"""Shared pytest configuration and fixtures.

Postgres-backed: every test that needs a TicketStore gets a fresh,
isolated schema in a single per-session Postgres container (pgvector image).

How the DB is provided:
  - If ``TEST_DATABASE_URL`` is set, use it as-is (CI / local dev fast path).
  - Otherwise spin up ``pgvector/pgvector:pg17`` via testcontainers.
  - If neither works, every DB-touching test is skipped with a clear message.
"""

from __future__ import annotations
import json
import os
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tickets.json"


@pytest.fixture(scope="session")
def raw_ticket_rows() -> list[dict]:
    """200 synthetic ticket rows, schema-identical to the HF dataset."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pg_dsn():
    """Yield a DSN pointing at a pgvector-enabled Postgres instance."""
    env = os.environ.get("TEST_DATABASE_URL")
    if env:
        yield env
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip(
            "testcontainers not installed; set TEST_DATABASE_URL or "
            "`uv sync --group dev` to install it"
        )
    container = PostgresContainer("pgvector/pgvector:pg17", driver=None)
    try:
        container.start()
    except Exception as e:
        pytest.skip(f"could not start pgvector container: {e}")
    try:
        dsn = container.get_connection_url()
        # testcontainers may return a SQLAlchemy-style URL; psycopg wants plain.
        dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")
        dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        # Wait until the extension is creatable (image is ready earlier than
        # the DB accepts CREATE EXTENSION on some hosts).
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        yield dsn
    finally:
        container.stop()


@pytest.fixture
def pg_schema(pg_dsn):
    """Allocate a unique schema for one test, drop it on teardown."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield name
    try:
        with psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(name)
                )
            )
    except Exception:
        # ponytail: best-effort drop on teardown; if the session-level
        # container is already gone there's nothing to clean.
        pass


@pytest.fixture(autouse=True)
def _clear_create_ticket_idempotency_cache():
    """create_ticket keeps a process-global idempotency cache. Each test
    gets a fresh store but shares that cache, so without per-test isolation
    a cached id from one test points into another test's discarded store."""
    from mcp_cst.tools import create_ticket as _ct

    _ct._idempotency_cache.clear()
    yield
    _ct._idempotency_cache.clear()
