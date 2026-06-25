import json

from mcp_cst.resources.schema import (
    schema_payload,
    schema_resource_body,
    schema_writes_payload,
)


def test_schema_describes_columns():
    payload = schema_payload()
    assert isinstance(payload, dict)
    cols = {c["name"] for c in payload["columns"]}
    assert {
        "subject",
        "body",
        "answer",
        "queue",
        "priority",
        "language",
        "tags",
    }.issubset(cols)


def test_schema_lists_filter_values():
    payload = schema_payload()
    assert "valid_filters" in payload
    assert payload["valid_filters"]["language"] == ["en", "de", "he"]
    assert isinstance(payload["valid_filters"]["priority"], list)
    assert isinstance(payload["valid_filters"]["queue"], (list, str))


def test_schema_calls_out_missing_fields():
    payload = schema_payload()
    missing = payload["not_available"]
    assert any("timestamp" in m.lower() for m in missing)
    assert any("customer" in m.lower() for m in missing)


def test_schema_payload_excludes_writes():
    """#109: don't openly enumerate destructive writes in the public schema."""
    payload = schema_payload()
    assert "writes" not in payload


def test_schema_writes_payload_kept_separate():
    """Writes documentation still exists as a helper but isn't exposed via
    schema://tickets."""
    writes = schema_writes_payload()
    assert "writes" in writes
    assert any("delete_ticket" in w for w in writes["writes"])


def test_schema_resource_body_serializes_payload():
    body = schema_resource_body()
    assert body == json.dumps(schema_payload(), indent=2)
    parsed = json.loads(body)
    assert parsed == schema_payload()
    assert "writes" not in parsed
