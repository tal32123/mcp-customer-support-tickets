from mcp_cst.config import Config, DEFAULT_DATABASE_URL, DEFAULT_SCHEMA


def test_defaults(monkeypatch):
    monkeypatch.delenv("MCP_CST_DATASET_REVISION", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MCP_CST_DB_SCHEMA", raising=False)
    cfg = Config.from_env()
    assert cfg.dataset_id == "Tobi-Bueck/customer-support-tickets"
    assert cfg.dataset_revision  # baked-in default
    assert cfg.embedding_model == "intfloat/multilingual-e5-small"
    assert cfg.database_url == DEFAULT_DATABASE_URL
    assert cfg.db_schema == DEFAULT_SCHEMA


def test_database_url_override(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5433/db")
    cfg = Config.from_env()
    assert cfg.database_url == "postgresql://u:p@h:5433/db"


def test_db_schema_override(monkeypatch):
    monkeypatch.setenv("MCP_CST_DB_SCHEMA", "tickets_v2")
    cfg = Config.from_env()
    assert cfg.db_schema == "tickets_v2"


def test_revision_override(monkeypatch):
    monkeypatch.setenv("MCP_CST_DATASET_REVISION", "main")
    assert Config.from_env().dataset_revision == "main"
