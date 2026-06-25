"""schema://tickets resource — describes the dataset shape."""

from __future__ import annotations

from ..docs import make_description


DESCRIPTION = make_description(
    summary="Schema for the ticket corpus: columns, valid filter values, and notes on what is NOT available.",
    use_for="Use this for: discovering valid filter values before calling search_tickets or aggregate_tickets, understanding the data shape.",
    not_for="Do NOT use this for: fetching ticket content (use get_ticket or ticket:// resource).",
    output="Output: JSON with `columns`, `valid_filters`, `not_available`, and `writes` sections.",
    include_g4=False,
)


def schema_payload() -> dict:
    return {
        "columns": [
            {
                "name": "id",
                "description": "12-char hex derived as sha1(revision || row_index).",
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
        "writes": [
            "create_ticket: append one ticket (subject + body required; answer, type, queue, priority, language, version, tags optional). Returns {id}. The id is derived from the next available row_index using the same scheme as ingest. New tickets live in the per-revision cache and survive restarts.",
            "update_ticket: patch one ticket by id; unspecified fields are left alone; `tags` replaces the full list. Re-embeds and re-indexes. Returns {id, updated}. TICKET_NOT_FOUND if id is unknown.",
            "delete_ticket: remove one ticket by id. Returns {id, deleted}. Destructive and irreversible within the running store. TICKET_NOT_FOUND if id is unknown.",
        ],
    }

