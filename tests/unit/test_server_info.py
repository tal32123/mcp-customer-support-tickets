import numpy as np
import pytest
from mcp_cst.tools.server_info import server_info_payload
from mcp_cst.data.store import TicketStore
from mcp_cst.config import Config


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s", revision="rev42", rows=raw_ticket_rows, embedder=fake_embed,
    )


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
