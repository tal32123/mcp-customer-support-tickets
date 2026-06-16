import numpy as np
import pytest
from mcp_cst.resources.ticket import ticket_resource_body, DESCRIPTION
from mcp_cst.data.store import TicketStore


def fake_embed(texts):
    return np.ones((len(texts), 384), dtype=np.float32)


@pytest.fixture
def store(tmp_path, raw_ticket_rows):
    return TicketStore.create(
        path=tmp_path / "s", revision="r", rows=raw_ticket_rows, embedder=fake_embed,
    )


def test_body_returns_wrapped(store):
    first_id = store.all_ids()[0]
    body = ticket_resource_body(store, first_id)
    assert body.startswith(f'<ticket id="{first_id}">')


def test_description_contains_g4():
    from mcp_cst.docs import G4_REMINDER
    assert G4_REMINDER in DESCRIPTION
