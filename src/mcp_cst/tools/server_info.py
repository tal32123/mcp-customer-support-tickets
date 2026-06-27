"""server_info tool — read-only metadata."""

from __future__ import annotations

from .. import __version__
from ..config import Config
from ..data.store import TicketStore
from ..docs import make_description


DESCRIPTION = make_description(
    summary="Return read-only metadata about the running server: dataset id, revision, embedding model, row count, license, package version.",
    use_for="Use this for: 'which dataset version is this?', 'how many tickets are there in total?', 'what license is the data under?', diagnostics.",
    not_for="Do NOT use this for: searching tickets (use search_tickets), fetching a ticket by id (use get_ticket), counts grouped by a field (use aggregate_tickets).",
    output="Output: a JSON object with dataset_id, dataset_revision, embedding_model, row_count, license, package_version.",
    include_g4=False,
)


def server_info_payload(*, cfg: Config, store: TicketStore) -> dict:
    return {
        "dataset_id": cfg.dataset_id,
        "dataset_revision": cfg.dataset_revision,
        "embedding_model": cfg.embedding_model,
        "row_count": store.row_count(),
        "license": "CC-BY-NC-4.0",
        "package_version": __version__,
    }
