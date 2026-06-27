import numpy as np
import pytest
from mcp_cst.tools.server_info import server_info_payload
from mcp_cst.data.store import TicketStore
from mcp_cst.config import Config


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def store(pg_dsn, pg_schema, raw_ticket_rows):
    s = TicketStore.create_with_rows(
        dsn=pg_dsn,
        schema=pg_schema,
        revision="rev42",
        rows=raw_ticket_rows,
        embedder=fake_embed,
    )
    yield s
    s.close()


def test_payload_shape(store, monkeypatch):
    monkeypatch.setenv("MCP_CST_DATASET_REVISION", "rev42")
    cfg = Config.from_env()
    payload = server_info_payload(cfg=cfg, store=store)
    assert payload["dataset_id"] == cfg.dataset_id
    assert payload["dataset_revision"] == "rev42"
    assert payload["embedding_model"] == cfg.embedding_model
    assert payload["row_count"] == store.row_count()
    assert payload["license"] == "CC-BY-NC-4.0"
    assert "package_version" in payload
