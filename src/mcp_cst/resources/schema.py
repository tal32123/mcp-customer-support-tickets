"""schema://tickets resource — describes the dataset shape."""

from __future__ import annotations
import json

from ..docs import make_description


DESCRIPTION = make_description(
    summary="Schema for the ticket corpus: columns, valid filter values, and notes on what is NOT available.",
    use_for="Use this for: discovering valid filter values before calling search_tickets or aggregate_tickets, understanding the data shape.",
    not_for="Do NOT use this for: fetching ticket content (use get_ticket or ticket:// resource).",
    output="Output: JSON with `columns`, `valid_filters`, and `not_available` sections.",
    include_g4=False,
)


def schema_payload() -> dict:
    return {
        "columns": [
            {
                "name": "id",
                "description": "Either a 12-char hex (sha1(revision||row_index) for HF dataset rows) or a `usr_<32-hex>` UUIDv7 prefix for user-created tickets.",
            },
            {"name": "subject", "description": "Ticket subject line, verbatim."},
            {"name": "body", "description": "Ticket body, verbatim."},
            {
                "name": "answer",
                "description": "Support team's reply, verbatim. May be empty.",
            },
            {
                "name": "type",
                "description": "Ticket type. One of: question, incident, request, problem.",
            },
            {
                "name": "queue",
                "description": "Queue assigned to the ticket. 52 possible values.",
            },
            {
                "name": "priority",
                "description": "Priority. One of: low, medium, high, critical, info.",
            },
            {"name": "language", "description": "Language. One of: en, de, he."},
            {
                "name": "version",
                "description": "Product version associated with the ticket.",
            },
            {
                "name": "tag_1..tag_6",
                "description": "Original six tag slots, preserved verbatim.",
            },
            {
                "name": "tags",
                "description": "Normalized List[str] of non-empty tags; use this for filtering and aggregation.",
            },
        ],
        "valid_filters": {
            "language": ["en", "de", "he"],
            "priority": ["low", "medium", "high", "critical", "info"],
            "type": ["question", "incident", "request", "problem"],
            "queue": "<52 string values; see server_info or sample via aggregate_tickets>",
        },
        "not_available": [
            "No timestamp column — date-range filters will be refused.",
            "No customer fields (name, email, id) — cannot filter by customer.",
            "No ticket-id column from source — server fabricates stable ids.",
        ],
    }


def schema_writes_payload() -> dict:
    """Write-tool documentation kept separate so the public schema resource
    doesn't openly enumerate destructive operations to ticket-borne attackers
    (#109). Each write tool already documents itself via its own description.
    """
    return {
        "writes": [
            "create_ticket: append one ticket (subject + body required; answer, type, queue, priority, language, version, tags optional). Returns {id}. User-created ticket ids are `usr_<uuidv7-hex>` (36 chars total) — collision-safe across delete-then-create cycles. The dataset-ingest path still produces stable 12-hex ids for the 62k bulk rows. New tickets live in the per-revision cache and survive restarts.",
            "update_ticket: patch one ticket by id; unspecified fields are left alone; `tags` replaces the full list. Re-embeds and re-indexes. Returns {id, updated}. TICKET_NOT_FOUND if id is unknown.",
            "delete_ticket: remove one ticket by id. Returns {id, deleted}. Destructive and irreversible within the running store. TICKET_NOT_FOUND if id is unknown.",
        ],
    }


def schema_resource_body() -> str:
    """Serialized form of `schema_payload()` for the schema:// resource."""
    return json.dumps(schema_payload(), indent=2)
