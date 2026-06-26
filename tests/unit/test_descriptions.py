import pytest

from mcp_cst.tools import (
    aggregate_tickets,
    create_ticket,
    delete_ticket,
    get_ticket,
    search_tickets,
    server_info,
    update_ticket,
)
from mcp_cst.resources import ticket, schema
from mcp_cst.prompts import draft_reply
from mcp_cst.docs import CROSS_TOOL_REPLAY_WARNING, G4_REMINDER


REQUIRED_SECTIONS = ["Use this for:", "Do NOT use this for:", "Output:"]

TICKET_RETURNING_SURFACES = [
    ("get_ticket tool", get_ticket.DESCRIPTION),
    ("search_tickets tool", search_tickets.DESCRIPTION),
    ("ticket resource", ticket.DESCRIPTION),
    ("draft_reply prompt", draft_reply.DESCRIPTION),
]

ALL_SURFACES = [
    ("server_info tool", server_info.DESCRIPTION),
    ("schema resource", schema.DESCRIPTION),
    ("aggregate_tickets tool", aggregate_tickets.DESCRIPTION),
    *TICKET_RETURNING_SURFACES,
]


@pytest.mark.parametrize("name,desc", ALL_SURFACES)
def test_required_sections_present(name, desc):
    for section in REQUIRED_SECTIONS:
        assert section in desc, f"{name} missing section: {section!r}"


@pytest.mark.parametrize("name,desc", TICKET_RETURNING_SURFACES)
def test_g4_reminder_on_ticket_returning_surfaces(name, desc):
    assert G4_REMINDER in desc, f"{name} missing the G4 reminder"


@pytest.mark.parametrize("name,desc", ALL_SURFACES)
def test_descriptions_have_summary_first_line(name, desc):
    first = desc.splitlines()[0]
    assert first.strip(), f"{name} has empty first line"
    assert len(first) <= 200, f"{name} summary too long ({len(first)} chars)"


# ---------------------------------------------------------------------------
# #106: cross-tool replay warning on write tools only
# ---------------------------------------------------------------------------

WRITE_SURFACES = [
    ("create_ticket", create_ticket.DESCRIPTION),
    ("update_ticket", update_ticket.DESCRIPTION),
    ("delete_ticket", delete_ticket.DESCRIPTION),
]

READ_SURFACES = [
    ("server_info", server_info.DESCRIPTION),
    ("get_ticket", get_ticket.DESCRIPTION),
    ("search_tickets", search_tickets.DESCRIPTION),
    ("aggregate_tickets", aggregate_tickets.DESCRIPTION),
    ("schema resource", schema.DESCRIPTION),
    ("ticket resource", ticket.DESCRIPTION),
    ("draft_reply prompt", draft_reply.DESCRIPTION),
]


@pytest.mark.parametrize("name,desc", WRITE_SURFACES)
def test_write_tools_carry_replay_warning(name, desc):
    assert CROSS_TOOL_REPLAY_WARNING in desc, f"{name} missing cross-tool replay warning"


@pytest.mark.parametrize("name,desc", READ_SURFACES)
def test_read_surfaces_do_not_carry_replay_warning(name, desc):
    assert CROSS_TOOL_REPLAY_WARNING not in desc, f"{name} should not carry write-tool warning"
