from mcp_cst.docs import G4_REMINDER, make_description


def test_g4_reminder_text():
    assert "data" in G4_REMINDER.lower()
    assert "instructions" in G4_REMINDER.lower()
    assert "<ticket>" in G4_REMINDER


def test_make_description_includes_required_sections():
    desc = make_description(
        summary="One-line summary.",
        use_for="Use this for: finding tickets about X.",
        not_for="Do NOT use this for: counting (use aggregate_tickets).",
        output="Output: list of {id, subject, snippet}.",
        include_g4=True,
    )
    assert "One-line summary." in desc
    assert "Use this for:" in desc
    assert "Do NOT use this for:" in desc
    assert "Output:" in desc
    assert G4_REMINDER in desc


def test_make_description_no_g4():
    desc = make_description(
        summary="x", use_for="x", not_for="x", output="x", include_g4=False,
    )
    assert G4_REMINDER not in desc
