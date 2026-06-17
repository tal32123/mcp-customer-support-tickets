from mcp_cst.resources.schema import schema_payload


def test_schema_describes_columns():
    payload = schema_payload()
    assert isinstance(payload, dict)
    cols = {c["name"] for c in payload["columns"]}
    assert {"subject", "body", "answer", "queue", "priority", "language", "tags"}.issubset(cols)


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
