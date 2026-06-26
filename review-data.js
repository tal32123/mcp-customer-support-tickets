// Consolidated review findings.
//
// State after iteration 4 (current):
//   • Iteration 1 → 47 items implemented.
//   • Iteration 2 → 8 Q&A clarifications captured (qa field on QA_FINDINGS).
//   • Iteration 3 → 24 simplification findings landed (4 dead modules deleted,
//     ~321 LoC; dedupes/inlines across store/server/retrieval/prompts/safety
//     /docs/embedder/ingest/schema). Verified by the unit-test suite.
//   • Iteration 4 → 48 items landed across data layer (FTS throttle, ANN
//     index, write lock, NaN check, BM25 None coercion, _raw_get helper),
//     retrieval (tag predicate pushdown, TicketFilters TypedDict,
//     build_filters, aggregate write-seq cache, snippet safety wrap,
//     empty-query guard, recall-floor test), safety/prompts (quoteattr
//     unified, trust-boundary notice, DE/HE injection patterns, schema
//     writes split, language-match grounding, body cap, markdown
//     neutralization, prompt invariants, warn-once on None, schema_resource_body
//     wired up, Grounding NamedTuple), server/tooling (INTERNAL_ERROR
//     envelope, max_length validators, lazy embedder warm-up with error
//     capture, CPU-default torch, Python 3.11 floor, answer injection
//     screen, type/priority/version validators, ruff --fix off,
//     integration marker registered, brittle test relaxed, O(n²) dedup),
//     plus three AI-Agent UX features (#200 cursor pagination + search_id,
//     #201 batch get_tickets, #202 search_and_fetch composite).
//     Verified: full unit-test suite (167 tests) passes deterministically.
//
// RESOLVED_COUNT (= 124) drives the green banner at the top of review.html.
//
// What remains here:
//   • QA_FINDINGS — empty (all 8 prior Q&A items have landed).
//   • OPEN_FINDINGS — 3 items: the reverted CI/scripts/pre-commit triple
//     (#315/#316/#317) that needs proper rework before re-adding.
//   • GUARDRAIL_FINDINGS — 17 LLM-safety/Ops/Data-quality proposals that
//     are feature-shaped (rate limits, audit log, soft delete, taxonomy,
//     drift detector, language detection, idempotency keys, etc.).
//   • FEATURE_FINDINGS — 39 product ideas across AI-Agent/Retrieval/Infra/
//     Support workflow/Platform·DX.
//   • SIMPLIFY_FINDINGS — empty (refilled per iteration as new sweeps run).
//
// The HTML renders the sections in this order.
// OPEN_FINDINGS, GUARDRAIL_FINDINGS, FEATURE_FINDINGS, and SIMPLIFY_FINDINGS.

window.RESOLVED_COUNT = 129;

window.QA_FINDINGS = [];

// ============================================================================
// FEATURE_FINDINGS — feature proposals from 3 product-manager subagents.
// Internal-use deployment (no PII / external-customer features).
//   200-219: AI/LLM-tooling PM (agentic UX, retrieval, infra)
//   220-239: Support-ops PM (workflows reps + managers actually want)
//   240-259: Platform/DX PM (forking, plug-points, observability)
// ============================================================================

window.FEATURE_FINDINGS = [
  { id: 203, title: "find_resolved_similar — grounding-quality retrieval as its own tool", score: 9, band: "P0", effort: "S", crossValidated: 1, category: "Retrieval",
    location: "src/mcp_cst/prompts/draft_reply.py (select_grounding is private)",
    problem: "select_grounding's logic — cosine≥0.70, non-empty answer, excluding self — is the highest-signal retrieval mode the server has, but it's locked inside draft_reply and unreachable as a tool. Agents doing analysis fall back to search_tickets which returns unresolved noise.",
    implication: "Surfacing this as a tool turns a 4-turn loop into one call and makes draft_reply a thin wrapper.",
    fix: "Promote to find_resolved_similar(subject, body, k=3, min_similarity=0.70) -> [{id, subject, body, answer, similarity, ticket_uri}]. Refactor draft_reply to call it." },
  { id: 204, title: "Server-side session memory keyed by MCP session_id", score: 9, band: "P1", effort: "M", crossValidated: 0, category: "AI/Agent UX",
    location: "design-level — new module src/mcp_cst/session.py",
    problem: "Long agent sessions accumulate implicit context ('we're auditing the refund flow this week', 'always exclude tickets in queue=Sales'). Every tool call is stateless so the agent has to re-state filters and intent on each turn, burning tokens and risking drift.",
    implication: "With sticky scope, an agent sets 'sticky_filters={language:en, queue:Billing}' once and runs 50 follow-up searches without re-specifying. Removes a whole class of consistency bugs.",
    fix: "set_scope(filters, notes='') + get_scope(). Stored in-process keyed by FastMCP session id; cleared on disconnect. search/aggregate merge sticky with call-time (call-time wins on conflict, return effective_filters)." },
  { id: 205, title: "Saved queries persisted across sessions", score: 8, band: "P1", effort: "M", crossValidated: 1, category: "AI/Agent UX",
    location: "design-level — new LanceDB table saved_queries",
    problem: "The same compound query gets reissued every time the agent comes back to a workflow. There's no way to name and re-run it — the spec for what 'the refund-flow audit' means lives in CLAUDE.md, not the server.",
    implication: "Saved queries become a first-class durable artifact the agent can build a workflow library around. Weekly cron-style agent runs become trivial.",
    fix: "save_query(name, q, filters), list_saved_queries(), run_saved_query(name, limit?, cursor?). Persist in per-revision cache dir; invalidate on dataset revision bump." },
  { id: 206, title: "Prompts library: triage_ticket, summarize_thread, compare_tickets, quality_review", score: 8, band: "P1", effort: "M", crossValidated: 1, category: "AI/Agent UX",
    location: "src/mcp_cst/prompts/ — only draft_reply.py exists",
    problem: "MCP prompts are a first-class user-discoverable surface (Claude Desktop shows them in /, Claude Code in slash menu). Today there's exactly one, so the human-in-loop UX for the server is one button.",
    implication: "Prompts move the UX from 'agent improvises a workflow' to 'user picks a named workflow with pre-wired grounding'. Each prompt pins a consistent output shape.",
    fix: "Add triage_ticket(id), summarize_thread(ids[]), compare_tickets(ids[]), quality_review(id). All deterministic context assembly, no server-side LLM — same pattern as draft_reply." },
  { id: 207, title: "Retriever selection per query (BM25 / vector / hybrid) and a debug mode", score: 8, band: "P2", effort: "S", crossValidated: 1, category: "Retrieval",
    location: "tools/search_tickets.py, retrieval/hybrid.py:56",
    problem: "Hybrid is the right default but not the right answer for every query. Exact error codes ('ERR_5021') should run BM25-only — vector noise dilutes the rank. Conceptual queries should run vector-only. Agent has no way to express that.",
    implication: "Letting the agent pick mode gives ~free precision wins on the easy cases. A debug=True flag returns per-branch rank lists for retrieval failure debugging and eval-harness work.",
    fix: "mode: 'hybrid'|'bm25'|'vector' = 'hybrid' and debug: bool = False. When debug, return {bm25_ranks, vector_ranks, fused_ranks, candidate_k} alongside hits." },
  { id: 208, title: "suggest_tags(subject, body) — server-side knn over tag distribution", score: 7, band: "P2", effort: "S", crossValidated: 1, category: "AI/Agent UX",
    location: "design-level — new tool; reuses passage embedder",
    problem: "create_ticket accepts arbitrary tags. An agent triaging has no grounded way to pick tags matching the corpus vocabulary — invents new tags (drift) or skips.",
    implication: "Auto-tag suggestions keep agent writes consistent with the corpus. Compounds: better tags → better tag-filtered searches → better grounding.",
    fix: "suggest_tags(subject, body, k=3) -> [{tag, support, example_ticket_ids}]. Embed input, fetch top-50 vector neighbours, count tag frequency, return top-k with contributing ticket ids." },
  { id: 209, title: "Streamable HTTP transport — unlock remote/multi-tenant agentic use", score: 8, band: "P1", effort: "L", crossValidated: 2, category: "Infra",
    location: "src/mcp_cst/server.py:254 (mcp.run() = stdio only)",
    problem: "stdio-only means one process per client, no multi-client sharing of the embedding model + LanceDB handle, and no way for cloud agents to reach the server. (Also flagged by Platform PM as #243.)",
    implication: "Streamable HTTP turns this from 'a thing I run locally' into 'a thing my whole org's agents can use over the network'. Composes with shared session state (#204).",
    fix: "mcp-customer-support-tickets serve --http --port 8080 invokes mcp.run(transport='streamable-http'). Keep stdio as default. Bearer-token auth behind MCP_CST_BEARER first; OAuth follow-up." },
  { id: 210, title: "MCP elicitation: disambiguation mid-call instead of guessing", score: 7, band: "P2", effort: "M", crossValidated: 1, category: "AI/Agent UX",
    location: "design-level — draft_reply, delete_ticket, ambiguous-search paths",
    problem: "Several flows have a hidden user-decision step the agent has to invent: delete_ticket says 'confirm with the user' (soft contract); draft_reply picks a target language without asking; ambiguous searches silently pick the top hit.",
    implication: "Elicitation turns these from 'agent must remember to confirm' soft contracts into protocol-level safety. delete_ticket becomes physically impossible without explicit user yes.",
    fix: "Use FastMCP's ctx.elicit(...). For delete_ticket: confirmation showing subject/body. For draft_reply when target_language is None and ticket language differs from session. For search_tickets when top-2 hits have near-tied scores and disambiguate=True." },
  { id: 211, title: "MCP completion provider for filter values (queue, tags, language)", score: 7, band: "P2", effort: "S", crossValidated: 1, category: "AI/Agent UX",
    location: "src/mcp_cst/resources/schema.py + new completion handler",
    problem: "queue has 52 valid values; docs say 'Use schema://tickets to see valid values'. Agents either fetch the schema resource (extra round-trip) or guess a queue name and hit a silent zero-result search.",
    implication: "Completion turns 'agent guesses queue=Billing' into 'client autocompletes from server-provided values' for human-facing clients, and lets agents call a fast complete_filter(field, prefix).",
    fix: "Implement MCP completion for search_tickets.queue, .tags, aggregate_tickets.queue. Source: existing schema://tickets distinct-value cache." },
  { id: 212, title: "Negative search (q_exclude) and OR-of-terms", score: 7, band: "P2", effort: "S", crossValidated: 1, category: "Retrieval",
    location: "tools/search_tickets.py, retrieval/hybrid.py",
    problem: "Agents iteratively refine queries by adding negative constraints ('login error but not mobile'). The agent has to post-filter results in its own context, paying token cost for content it then discards.",
    implication: "First-class negative-term support cuts a common 3-call refinement loop to 1 call.",
    fix: "q_exclude: str | None (BM25 NOT clause + vector post-filter on candidate set). Could also add q_any: list[str] for OR-of-terms." },
  { id: 213, title: "recent_tools_used resource — agent self-observation to break loops", score: 7, band: "P2", effort: "S", crossValidated: 1, category: "AI/Agent UX",
    location: "design-level — new resource recent://tool_calls",
    problem: "Long-running agents hit two loops: (a) issuing the same search 3× with trivial rewording because they forgot they already tried it, (b) re-fetching the same ticket every turn. The server sees every call but exposes none of it back.",
    implication: "Agents that can introspect 'what have I tried this session' write better plans. Empirically halves redundant calls.",
    fix: "In-memory ring buffer (last 50 calls per session) of {tool, args_hash, result_summary, ts}. Expose as recent://tool_calls resource." },
  { id: 214, title: "sample_random(n, filters) for stratified spot-checks and eval seeding", score: 6, band: "P3", effort: "S", crossValidated: 1, category: "Retrieval",
    location: "design-level — new tool",
    problem: "Two common agentic workflows have no good primitive: (a) 'show me 5 random Billing/EN tickets so I can understand what this queue looks like', (b) seeding the eval harness.",
    implication: "Turns the corpus from 'a thing you query' into 'a thing you can explore'. Critical for eval-harness story (#241).",
    fix: "sample_random(n=5, queue?, language?, type?, has_answer?: bool, seed?: int) -> list[ticket_preview]. Deterministic when seed is provided." },
  { id: 215, title: "Time-bucketed aggregates over ingested_at (locally-created tickets)", score: 6, band: "P3", effort: "S", crossValidated: 1, category: "Retrieval",
    location: "tools/aggregate_tickets.py + store.py (ingested_at column)",
    problem: "HF dataset has no timestamps, but locally-created tickets via create_ticket should have an ingested_at. Once that lands, agents will want 'tickets created this week, grouped by queue'.",
    implication: "Time-bucketed agg is the first non-trivial analytical primitive and unlocks 'is this issue trending?' workflows. (Depends on the timestamp column landing.)",
    fix: "Add ingested_at on writes (NULL for HF rows). Extend aggregate_tickets with time_bucket: 'day'|'week'|'month'. Raise UNSUPPORTED_FILTER when filter selects HF rows + time_bucket set." },
  { id: 216, title: "Quality signal: mark_unhelpful(query, ticket_id) for LTR retraining", score: 6, band: "P3", effort: "M", crossValidated: 2, category: "Retrieval",
    location: "design-level — new tool + new feedback table",
    problem: "Every time an agent (or user) decides a result was off-topic, that judgement evaporates. No closed loop between retrieval mistakes and improvement. (Also flagged by Platform PM as #252.)",
    implication: "Even without a learned reranker, recording {query, returned_ids, marked_unhelpful_ids, ts} builds the dataset that lets a future cross-encoder rerank step learn from this deployment.",
    fix: "mark_unhelpful(query, ticket_id, reason?='') and mark_helpful. Append-only LanceDB table. Expose recent_feedback(limit=50) resource." },
  { id: 217, title: "Conditional reads via version — skip ticket bodies the agent has already seen", score: 6, band: "P3", effort: "M", crossValidated: 0, category: "Infra",
    location: "tools/get_ticket.py + ticket://{id} resource",
    problem: "In multi-turn workflows the same ticket gets fetched repeatedly because the agent re-checks before reasoning. Each re-fetch costs a full body in the context window.",
    implication: "A version field lets the agent ask get_ticket(id, if_version_not='v3') and get a one-line 'unchanged' response, preserving context budget.",
    fix: "Add monotonic version: int on every row, bumped by create/update. get_ticket accepts if_version_not; if current==if_version_not, return {id, version, unchanged: true}." },
  { id: 218, title: "explain_match(ticket_id, query) — make retrieval rationale inspectable", score: 6, band: "P3", effort: "M", crossValidated: 0, category: "Retrieval",
    location: "design-level — composes hybrid_search internals",
    problem: "When a result is surprising the agent has two choices: trust it or refetch. There's no way to ask why a ticket ranked where it did.",
    implication: "Surfacing per-result rank, BM25 score, cosine, matched tokens lets the agent self-assess confidence and decide when to broaden vs narrow.",
    fix: "explain_match(ticket_id, q) -> {bm25_rank, vector_rank, fused_rank, bm25_score, cosine, matched_terms: [str], snippet_with_highlights}. Cheap: rerun the two branches with limit=candidate_k." },
  { id: 220, title: "auto_triage tool — suggest queue, priority, type for an incoming ticket", score: 9, band: "P0", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "When a new ticket arrives, a rep or routing bot has to read it, decide which queue it belongs to (52 values), set priority, and tag the type. create_ticket accepts these as inputs but provides zero guidance, so triage quality varies by rep tenure.",
    implication: "Cuts mis-routes, shortens time-to-first-response, gives newer reps the same routing intuition as a 5-year veteran. Plausibly the highest-ROI single feature for a support manager.",
    fix: "auto_triage(subject, body) runs the existing embedder, pulls top-k nearest historical tickets, returns majority-vote queue/priority/type plus supporting neighbour ids and confidence score." },
  { id: 221, title: "find_related_tickets — cluster everything about the same customer/order/issue", score: 9, band: "P0", effort: "S", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "Reps need to see every ticket connected to the one they're holding — same customer, same order id, same SKU, same outage. Today they run search_tickets multiple times with hand-crafted queries and stitch results.",
    implication: "Stops 'we already answered this last Tuesday' embarrassment, surfaces repeat offenders for proactive outreach.",
    fix: "find_related_tickets(ticket_id, mode='auto'|'customer'|'entity'|'semantic') combines semantic neighbours with regex-extracted entities (order numbers, SKUs, email-shaped tokens), returns a deduped ranked list grouped by relation type." },
  { id: 222, title: "knowledge_gaps tool — surface queries with no good historical match", score: 9, band: "P0", effort: "S", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "draft_reply already refuses to scaffold when nothing clears 0.70, but those refusals vanish into the rep's session. The support manager never learns 30 reps this week hit walls on the same novel issue — the signal that should drive runbook investment.",
    implication: "Turns every refused draft into a measurable runbook gap, prioritising what content/docs/macros to write next.",
    fix: "Log refused-draft queries to a small local table; expose knowledge_gaps(window='7d', limit=20) returning clustered gap topics with example queries, hit counts, and the best (still weak) historical match." },
  { id: 223, title: "macro_library tool — mine the top resolution patterns per queue/type", score: 8, band: "P1", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool / resource)",
    problem: "Corpus contains ~62k answer strings — almost certainly dozens of high-frequency reply patterns (password resets, refund confirmations, shipping ETAs). Reps re-type variations of the same response daily.",
    implication: "Saves measurable seconds per reply across thousands of replies/day, and gives the macro library a defensible basis ('these are the patterns we actually use').",
    fix: "macro_library(queue=None, type=None, top_n=10) clusters answers by embedding, picks a representative per cluster, returns subject template + body skeleton + frequency + example ticket ids. Could ship as macros://{queue} resource for browsing." },
  { id: 224, title: "next_in_queue — defensible 'what should I work on next' ordering", score: 8, band: "P1", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "A rep opens their shift and faces an unsorted pile. They cherry-pick the easy ones, angry ones get bumped, and the manager has no policy lever. No way to ask Claude 'what should I pick up next?'",
    implication: "Eliminates queue-grazing, ensures hard/aged tickets don't rot, gives the manager a knob (weights) to enforce policy without micromanaging.",
    fix: "next_in_queue(assignee=None, queue=None, limit=10) returns ordered list scored by priority, age, similarity-to-known-easy-wins, and an 'angry' signal from the body. Each row carries a one-line rationale." },
  { id: 225, title: "escalation_radar — flag tickets that look like prior P0 incidents", score: 8, band: "P1", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "Most incidents start as ordinary tickets and look small until they aren't. Tier-1 reps don't have the pattern recognition to spot 'this smells like the March checkout outage' early.",
    implication: "Cuts mean-time-to-escalate by minutes-to-hours on real incidents — the metric escalation leads actually get measured on. Creates an audit trail for postmortems.",
    fix: "escalation_radar(window='1h'|'24h') compares recent open tickets against historical 'this became P0' fingerprints and returns matches with similarity, contributing neighbour ids, and a suggested escalation message." },
  { id: 226, title: "reply_quality_check prompt — grade a draft before it sends", score: 8, band: "P1", effort: "S", crossValidated: 1, category: "Support workflow",
    location: "design-level (new prompt)",
    problem: "draft_reply ships the rep a draft. There's no symmetric surface for 'I wrote this myself — does it match how we usually answer?' Reps send blind or paste into a separate tool.",
    implication: "Catches missing links, missing apologies, wrong tone, missed-fact mistakes before the customer sees them. Cheap insurance.",
    fix: "reply_quality_check(ticket_id, draft_text) retrieves the same top-k neighbours draft_reply would, then assembles a graded-rubric prompt (covers customer concern? matches historical tone? cites a fix? includes a next step?) for the client's LLM." },
  { id: 227, title: "themes tool — top N topics in recent tickets via embedding clustering", score: 7, band: "P1", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "aggregate_tickets gives counts by structured fields. Support managers and product partners want 'what are reps hearing about this week?' — the open-ended themes that don't fit existing tags.",
    implication: "Turns the weekly support-to-product sync from anecdote-trading into a data-backed handoff. Surfaces emerging issues before they show up in NPS or churn.",
    fix: "themes(scope_filter=None, k=5) clusters embeddings within the filtered slice (HDBSCAN or k-means on existing vectors), labels each cluster via centroid-nearest subjects, returns size, representative ticket ids, and closest existing queue/tag." },
  { id: 228, title: "bulk_action — preview-then-apply close/retag/reassign across a similar cohort", score: 7, band: "P2", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (new tool)",
    problem: "When a known issue resolves, reps have to find and update dozens of related tickets one by one. update_ticket and delete_ticket are single-id only, so post-incident cleanup is a tedious copy-paste exercise that gets skipped.",
    implication: "Turns post-incident cleanup from a half-day chore into a single grounded call; keeps queues honest.",
    fix: "bulk_action(seed_ticket_id, action='close'|'retag'|'reassign', params, dry_run=True) uses find_related_tickets to gather the cohort, returns the impacted list for human review, applies on a second confirmed call. Logs every change with the seed for auditability." },
  { id: 229, title: "sentiment_signal field — flag angry/frustrated tickets for prioritisation", score: 7, band: "P2", effort: "M", crossValidated: 1, category: "Support workflow",
    location: "design-level (ingest enrichment + new filter)",
    problem: "Priority is a structural tag, not a tone tag. A polite P3 from a furious customer ('still no resolution after 4 emails…') should jump the queue but currently looks identical to a calm one.",
    implication: "Catches churn risk earlier, lets managers proactively reach out before customers tweet, makes next_in_queue and escalation_radar materially better.",
    fix: "At ingest, compute a sentiment score per ticket (lexicon or distilled classifier — CPU). Add the field to schema, let search/aggregate filter/group on it, surface a sentiment_hot tool returning the top frustrated open tickets." },
  { id: 240, title: "Pluggable corpus adapter — config knob to point at any HF/CSV/Parquet/SQL/S3 source", score: 9, band: "P0", effort: "L", crossValidated: 1, category: "Platform/DX",
    location: "src/mcp_cst/data/ingest.py:49-71; config.py:11-43",
    problem: "Ingest is hardwired to load_dataset('Tobi-Bueck/customer-support-tickets'). The README's headline 'swap the corpus and the pipeline doesn't change' has no implementation behind it. A forker with Zendesk exports has to fork ingest.py, store.py, and every Literal-typed tool parameter.",
    implication: "Anyone trying to deploy against their own corpus (the entire pitched use case) hits a hard fork. Adoption stops at 'cool demo, can't use it.'",
    fix: "CorpusAdapter Protocol (yield dict rows + declare field schema + filter enum values) + MCP_CST_CORPUS=hf://... | csv://... | postgres://... URL scheme. Bundle HFDatasetAdapter, ParquetAdapter, CsvAdapter; document a 30-line custom-adapter template." },
  { id: 241, title: "Eval harness exposed as an MCP tool — golden queries + precision@k/MRR/nDCG", score: 9, band: "P0", effort: "M", crossValidated: 1, category: "Platform/DX",
    location: "design-one-page.html:196 (promised, not implemented)",
    problem: "Design doc promises '50-200 golden queries tracking precision@5 per deploy' but no eval scaffold ships. Forkers who swap the embedder, corpus, or fusion strategy have no way to know if retrieval got worse.",
    implication: "Embedder/index/rerank swaps become flying-blind decisions. Kills the 'pluggable' story — every plug-point needs evidence.",
    fix: "eval_retrieval tool taking a golden-set name (bundled cst-default, or a path) and returning {p_at_5, mrr, ndcg_10, queries_failed}. Bundle golden set as YAML fixture in evals/. Expose mcp-cst eval CLI for CI." },
  { id: 242, title: "Pluggable embedder — config-driven backend (E5/OpenAI/Cohere/BGE)", score: 8, band: "P0", effort: "S", crossValidated: 1, category: "Platform/DX",
    location: "config.py:13 (EMBEDDING_MODEL constant); embedder.py (Protocol exists, only one impl)",
    problem: "The Embedder Protocol is shaped for swap-in alternatives, but Config.embedding_model + _make_embedders lock you to SentenceTransformer. No way to point at OpenAI text-embedding-3-small, Cohere embed-multilingual-v3, or local Ollama endpoint without editing code.",
    implication: "Half the platform pitch is unrealized despite the Protocol being in place. Orgs already standardized on a managed embedding provider have to fork.",
    fix: "MCP_CST_EMBEDDER=sentence-transformers://... | openai://... | cohere://... | http://... URL scheme. Ship OpenAIEmbedder and HTTPEmbedder as bundled implementations." },
  { id: 243, title: "Streamable HTTP transport alongside stdio — required for org-wide deployment", score: 8, band: "P0", effort: "M", crossValidated: 2, category: "Platform/DX",
    location: "src/mcp_cst/server.py:254 (mcp.run() defaults to stdio)",
    problem: "Server only speaks stdio JSON-RPC. Every consumer launches their own subprocess and pays the 2-min first-run embed cost individually; no shared instance behind a URL. (Also flagged by AI PM as #209.)",
    implication: "Blocks the 'one MCP server reachable from every AI surface' design claim. Multi-tenant, rate-limited, audited operation is impossible.",
    fix: "MCP_CST_TRANSPORT=stdio|http and wire mcp.run(transport='streamable-http', host=..., port=...). Document Authorization: Bearer pass-through. Provide Dockerfile + docker-compose." },
  { id: 244, title: "Observability — structured JSON logs, Prometheus /metrics, OTel traces", score: 8, band: "P0", effort: "M", crossValidated: 1, category: "Platform/DX",
    location: "src/mcp_cst/server.py:252 (bare text logs)",
    problem: "No structured logging, no per-tool latency/error counters, no trace context propagation. Can't tell whether p95 search latency is dominated by BM25, embedding, or LanceDB; can't tell which queries returned zero results.",
    implication: "Forks deployed in any production-adjacent setting are operationally blind. SREs can't write SLOs.",
    fix: "structlog for JSON logs keyed by request_id/tool_name/duration_ms/result_count. prometheus_client /metrics in HTTP mode. Honor OTEL_EXPORTER_OTLP_ENDPOINT via opentelemetry-instrumentation." },
  { id: 245, title: "First-class CLI — mcp-cst search, mcp-cst eval, mcp-cst ingest", score: 7, band: "P1", effort: "S", crossValidated: 1, category: "Platform/DX",
    location: "pyproject entry point is server-only",
    problem: "The only way to verify a fork works is to spin up Claude Code/Desktop and start chatting. No mcp-cst search 'kafka lag' to sanity-check, no mcp-cst ingest --rebuild to force a cold rebuild, no mcp-cst inspect <id> to dump a row.",
    implication: "DX is brutal for the engineer the README is trying to attract. Every experiment requires editing a tool description and reloading an MCP client.",
    fix: "click/typer CLI with subcommands search, get, aggregate, eval, ingest --rebuild, info that import the same impl modules the tools do. Same code path, different UI." },
  { id: 246, title: "Pluggable vector backend — adapter layer over LanceDB (Qdrant/pgvector/Weaviate)", score: 7, band: "P1", effort: "L", crossValidated: 1, category: "Platform/DX",
    location: "data/store.py + retrieval/hybrid.py:68-82 (LanceDB-specific calls)",
    problem: "TicketStore is a thin wrapper around LanceDB and the retrieval layer reaches into store.table directly with query_type='fts'|'vector'. Forkers with existing Qdrant or pgvector must rewrite the entire retrieval module.",
    implication: "Orgs with existing vector infrastructure cannot adopt without burning retrieval. The 'platform' story shrinks to 'embedded LanceDB or nothing.'",
    fix: "VectorStore Protocol with bm25_search/vector_search/fetch_rows. Move LanceDB into stores/lancedb.py as default impl. Document stores/qdrant.py skeleton. Hybrid fusion stays pure logic over (id, rank) tuples." },
  { id: 247, title: "Hot-reload / incremental ingest — file-watch the source, no restart", score: 7, band: "P1", effort: "M", crossValidated: 1, category: "Platform/DX",
    location: "data/ingest.py:49 (all-or-nothing); server.py:104-113",
    problem: "No incremental ingest. Bumping MCP_CST_DATASET_REVISION discards the cache and re-embeds all 62k rows. No file-watcher to pick up new CSV rows, no delta-merge against a Postgres updated_at cursor.",
    implication: "Production-adjacent forks must restart and re-embed on every refresh, losing locally-created data. README's 'live ticket source with incremental delta-ingest' doesn't exist.",
    fix: "store.upsert_rows(rows) (mostly there) and mcp-cst sync subcommand that diffs adapter output vs store.all_ids(). For HF Parquet, key by (revision, row_index); for CSV/SQL, declare a primary-key column." },
  { id: 248, title: "FTS-rebuild bottleneck blocks scale-out and concurrent writes (platform angle)", score: 7, band: "P1", effort: "M", crossValidated: 1, category: "Platform/DX",
    location: "data/store.py (every add/update/delete rebuilds FTS) — already in scope as approved #1 but called out here for the multi-writer story",
    problem: "Every write rebuilds the entire BM25 index from scratch. Single-writer in practice — cannot run two HTTP server replicas against the same LanceDB directory. (Approved #1 is already in scope; this is the platform PM's framing about scale-out.)",
    implication: "Read replicas are impossible. Bulk import (10k tickets) takes 10k × O(n) rebuilds. Multi-tenant deploy is gated on fixing this.",
    fix: "(Same as approved #1.) Batch FTS rebuilds: skip per-write rebuilds when batch=True is passed, expose store.reindex_fts() to fire once at end-of-batch. Long-term: LanceDB incremental FTS or move BM25 to a separate tantivy index." },
  { id: 249, title: "Plugin system for tools/prompts — drop a module in plugins/, no fork required", score: 6, band: "P1", effort: "M", crossValidated: 1, category: "Platform/DX",
    location: "server.py:118-248 (every tool/prompt hand-wired)",
    problem: "Adding a new tool requires editing server.py and writing a module in tools/. No entry-points-based plugin registry, no plugins/ directory scan, no way for a deploying team to ship a private extension without forking.",
    implication: "Every customization is a fork. Internal teams diverge from upstream and never re-merge. OSS users can't share mcp-cst-jira-plugin as side-loaded packages.",
    fix: "MCPCstPlugin Protocol (def register(mcp, store, embedder)) discovered via importlib.metadata.entry_points(group='mcp_cst.plugins'). Move the seven built-in tools to an internal default plugin." },
  { id: 250, title: "Cross-encoder rerank as opt-in flag — promised in design, not implemented", score: 6, band: "P2", effort: "S", crossValidated: 1, category: "Platform/DX",
    location: "retrieval/hybrid.py:84 (flat RRF only); design-one-page.html:196",
    problem: "Design doc says reranking is on the roadmap; today RRF is the final ranker. No rerank=true flag, no ms-marco-MiniLM integration, no way to A/B compare fused-only vs fused+reranked. Highest-leverage quality knob.",
    implication: "Forkers who care about quality have no on-ramp to the standard 'fuse then rerank top-N' pipeline. The eval harness (#241) needs this flag.",
    fix: "MCP_CST_RERANKER=cross-encoder://... config + rerank: bool = False parameter on search_tickets. Wire sentence_transformers.CrossEncoder against fused top-50, return top-limit." },
  { id: 251, title: "Snapshot / restore — portable store export to skip 2-min cold start", score: 6, band: "P2", effort: "S", crossValidated: 1, category: "Platform/DX",
    location: "data/store.py; README:88-93 (every machine pays ~2min first-run embed)",
    problem: "Each fresh machine re-downloads 120MB and re-embeds 62k tickets. Teams can't pre-bake a store and ship it to CI runners, dev VMs, or Docker images. Cache directory is opaque LanceDB internals.",
    implication: "CI is slow. New hires wait 2+ minutes on first launch. Docker images are huge or paying the cost at container start.",
    fix: "mcp-cst snapshot --out store.tar.zst (tar+zstd the store_path + manifest of {dataset_id, revision, embedding_model}) and mcp-cst restore. Validate manifest matches current config on restore." },
  { id: 252, title: "Query-result feedback loop — capture which result the agent used", score: 5, band: "P2", effort: "M", crossValidated: 2, category: "Platform/DX",
    location: "server.py (no feedback tool); retrieval/hybrid.py (no telemetry hook)",
    problem: "When search returns 10 hits and the agent grounds its reply on hit #3, that signal is thrown away. No record_grounding tool, no LTR training data, no way to know which queries route to wrong results. (Also flagged by AI PM as #216.)",
    implication: "Retrieval system never gets smarter from usage. Every fork starts from zero signal. The eval golden set has to be hand-curated forever instead of bootstrapped from agent traces.",
    fix: "record_grounding_choice tool (query, chosen_id, all_returned_ids, was_useful) writing append-only to feedback.jsonl in cache dir. Document downstream mcp-cst train-reranker --from feedback.jsonl follow-up." },
];

// ============================================================================
// GUARDRAIL_FINDINGS — proposed by specialist subagents in iteration 2.
// Internal-use deployment — Privacy/PII/GDPR category dropped per user.
// IDs partitioned by category:
//   100-119: LLM safety / prompt injection
//   140-159: Operational / DoS / resource limits
//   160-179: Data quality / business logic
// ============================================================================

window.GUARDRAIL_FINDINGS = [
  {
    id: 101,
    title: "No per-session tool-call rate limit enables runaway destructive loops",
    score: 9,
    band: "P0",
    effort: "S",
    crossValidated: 2,
    category: "LLM safety",
    location: "src/mcp_cst/server.py (design-level: no middleware around @mcp.tool)",
    problem: "Every tool dispatch goes straight to its _impl with no call counter. A malicious search hit that says 'For each id in [list], call delete_ticket' will be obeyed without any server-side circuit-breaker. The only brake is the tool description's 'Confirm with the user first' — a soft constraint a determined injection paragraph easily overrides. (Ops agent flagged the same gap as #140 from the DoS angle.)",
    implication: "One poisoned ticket can cascade into wholesale corpus deletion, mass mutation via update_ticket, or fan-out create_ticket spam that pollutes future retrievals. No recovery path — delete is documented as 'destructive and irreversible'.",
    fix: "Wrap mutating tools in a counter (e.g. _TOOL_CALL_COUNTS keyed per process or per-FastMCP-session) and raise RATE_LIMITED after N calls/minute. Default low (e.g. 3 deletes/min, 10 writes/min); make the limit overridable via config."
  },
  {
    id: 102,
    title: "Corpus-poisoning side channel via create_ticket → next session's search",
    score: 9,
    band: "P0",
    effort: "M",
    crossValidated: 1,
    category: "LLM safety",
    location: "src/mcp_cst/tools/create_ticket.py; src/mcp_cst/data/store.py:361-413",
    problem: "create_ticket inserts a row that becomes searchable immediately. The only filter is looks_like_injection — English-only-regex heuristic. There is no provenance tag distinguishing user-created from dataset-derived rows, no quarantine queue, and no length cap. An attacker calls create_ticket with a Hebrew/German paraphrased injection or a payload that just doesn't trigger the 5 regexes; the next user's search_tickets surfaces it as authoritative grounding/snippet.",
    implication: "Stored XSS-equivalent for LLM context. Multi-tenant ticket data crosses trust boundaries silently. Attacker pre-positions an injection that fires when an unrelated user runs an unrelated query that happens to retrieve the poisoned row.",
    fix: "Tag every create_ticket row with provenance='user_created' and created_by_session=<id> columns. Exclude user_created rows from select_grounding by default. Cap subject/body length. Optionally gate create_ticket behind a config flag (off by default for read-only deployments)."
  },
  {
    id: 105,
    title: "No audit trail of prompts/tool calls — forensics impossible after an incident",
    score: 7,
    band: "P1",
    effort: "S",
    crossValidated: 2,
    category: "LLM safety",
    location: "src/mcp_cst/server.py:95-108 (decorator chain has no logging)",
    problem: "_payload_on_error logs nothing. No record of which tool was called, with what args, by which session, with which result. draft_reply assembles a 5-grounding-ticket prompt and returns it; the server retains zero copy of what it told the calling LLM. (Data-quality agent flagged the same as #166 from the forensics-on-mutations angle.)",
    implication: "Cannot detect, investigate, or remediate prompt-injection incidents. Cannot satisfy compliance ('show me every prompt that referenced ticket X'). Cannot build a feedback loop to harden patterns.",
    fix: "Add a structured JSONL audit logger behind MCP_CST_AUDIT_LOG. Log: timestamp, tool name, redacted args (hash long bodies), error code, grounding ids selected for draft_reply, and looks_like_injection hits even when rejected. Rotate and cap size."
  },
  {
    id: 107,
    title: "delete_ticket has no server-side confirmation/challenge gate",
    score: 7,
    band: "P1",
    effort: "S",
    crossValidated: 1,
    category: "LLM safety",
    location: "src/mcp_cst/tools/delete_ticket.py:26-29; src/mcp_cst/server.py:262-265",
    problem: "delete_ticket takes one arg (the id), runs store.delete_ticket(id), and returns. The description tells the LLM 'Confirm with the user first — deletion is irreversible' but enforcement is purely the model's discretion. There's no two-step protocol, no time-window pause, no batch limit.",
    implication: "A single tool-poisoning injection that gets one delete_ticket call through bypasses all data protection. With no rate limit (#101) and no audit log (#105), an attacker who wins one round wins permanently.",
    fix: "Require a confirmation_token arg. Server returns a one-shot token from a new prepare_delete_ticket(id) tool that ALSO returns the ticket content for human review; delete_ticket validates the token (same process, single-use, 60s TTL) before executing."
  },
  {
    id: 140,
    title: "No per-session or per-minute tool-call rate limit",
    score: 9,
    band: "P1",
    effort: "M",
    crossValidated: 2,
    category: "Operational",
    location: "src/mcp_cst/server.py:141-265 (all tool registrations)",
    problem: "Every tool wired directly to FastMCP with no call-rate counter. A runaway or adversarial LLM agent can invoke search_tickets, create_ticket, or aggregate_tickets in a tight loop at thousands of calls per minute. No sliding-window counter, token-bucket, or back-off gate anywhere in the dispatch path. (LLM-safety angle: #101.)",
    implication: "CPU and GPU (embedding) saturation, LanceDB write amplification from repeated FTS rebuilds, and process OOM — all without any operator signal until the process dies.",
    fix: "Add a thread-safe token-bucket (e.g., 60 calls/minute per session id from FastMCP's lifespan context) enforced in a shared decorator applied to every @mcp.tool registration; return a structured RATE_LIMITED error payload on breach."
  },
  {
    id: 143,
    title: "No per-tool wall-clock timeout — slow embedding or LanceDB I/O hangs the server forever",
    score: 8,
    band: "P1",
    effort: "M",
    crossValidated: 1,
    category: "Operational",
    location: "src/mcp_cst/server.py:283-286; retrieval/hybrid.py:88",
    problem: "Embedder call in hybrid_search and in store.add_ticket/update_ticket runs on the calling thread with no timeout. On a cold CUDA init, memory-pressured host, or a model weight page-fault cascade, model.encode() can stall for tens of seconds or block indefinitely. FastMCP's stdio loop has no watchdog.",
    implication: "A single slow embedding call stalls the entire server; the MCP client sees no response and eventually disconnects, leaving the server process permanently busy with an orphaned encode call.",
    fix: "Wrap embedding calls and LanceDB queries in concurrent.futures.ThreadPoolExecutor with a per-call timeout (30s); raise structured TIMEOUT error so the tool returns an error payload rather than hanging."
  },
  {
    id: 144,
    title: "No disk-quota guard on the LanceDB cache directory",
    score: 7,
    band: "P1",
    effort: "S",
    crossValidated: 1,
    category: "Operational",
    location: "src/mcp_cst/data/store.py:409 (add_ticket); config.py:77-78 (store_path)",
    problem: "create_ticket writes a new LanceDB row (plus a full FTS index rebuild) without checking available disk space. Store path is under user-cache directory with no configured size ceiling. An agent looping on create_ticket can fill the disk, which will also corrupt the LanceDB WAL and manifest for all other readers.",
    implication: "Disk exhaustion crashes the server process and leaves the store in an unrecoverable partially-written state; recovery requires manual deletion and a full re-ingest.",
    fix: "Before each add_ticket, check shutil.disk_usage(cfg.store_path).free against a configurable MCP_CST_MIN_FREE_BYTES threshold (default 500MB) and raise QUOTA_EXCEEDED if headroom is insufficient."
  },
  {
    id: 145,
    title: "CI workflow has no step-level timeouts — a hung smoke test blocks the runner indefinitely",
    score: 7,
    band: "P2",
    effort: "S",
    crossValidated: 1,
    category: "Operational",
    location: ".github/workflows/ci.yml:18-78",
    problem: "None of the four CI jobs (lint, typecheck, test, smoke) set timeout-minutes at job level, and no step sets it either. The smoke job starts the stdio server as a subprocess; if embedder download hangs or server deadlocks, the GitHub Actions runner is consumed until the default 6-hour job timeout.",
    implication: "A single stuck CI run ties up a paid runner for up to 6 hours and blocks all queued PRs behind it; at org concurrency limits this can halt the entire pipeline.",
    fix: "Add timeout-minutes: 15 at job level for lint/typecheck/test and timeout-minutes: 30 for smoke; add step-level timeout on the 'Run smoke test' step specifically."
  },
  {
    id: 146,
    title: "No health/liveness tool for operator or orchestrator probing",
    score: 6,
    band: "P2",
    effort: "S",
    crossValidated: 1,
    category: "Operational",
    location: "src/mcp_cst/server.py:141-146 (server_info is closest analog)",
    problem: "server_info returns dataset metadata but requires a fully-initialized store and embedder; there is no lightweight ping that can be called before _init() completes or after a partial failure. Orchestrator restarting a crashed server has no way to distinguish 'still starting up' from 'deadlocked' without parsing log output.",
    implication: "Operators and health-check sidecars cannot reliably detect a hung or partially-initialized server, increasing mean-time-to-detection for outages.",
    fix: "Register a health tool that returns {status, uptime_s, store_ready, embedder_ready} from module-level flags set during _init(). Must not block on initialization itself, answering immediately with partial state if called early."
  },
  {
    id: 147,
    title: "No circuit breaker around LanceDB — repeated failures pay full retry cost",
    score: 6,
    band: "P2",
    effort: "M",
    crossValidated: 1,
    category: "Operational",
    location: "src/mcp_cst/data/store.py:328-354; hybrid.py:80-93",
    problem: "Every tool call that touches LanceDB (get, search, add, update, delete) issues a live query with no circuit-breaker state. If LanceDB begins returning errors (corrupted table, file lock contention, full disk), every subsequent tool call attempts the same I/O and fails after a full timeout. No open/half-open/closed state machine.",
    implication: "A sustained LanceDB fault causes every tool call to block for a full timeout duration before returning an error, multiplying CPU and I/O load while the store is already degraded.",
    fix: "Wrap store access in a simple circuit breaker (pybreaker or hand-rolled counter) that opens after 5 consecutive errors and resets after a 60-second cool-down, returning STORE_UNAVAILABLE immediately while open."
  },
  {
    id: 160,
    title: "create_ticket trusts caller's `language` — no body/language consistency check",
    score: 9,
    band: "P0",
    effort: "M",
    crossValidated: 1,
    category: "Data quality",
    location: "src/mcp_cst/tools/create_ticket.py:58-69; data/store.py:361-413",
    problem: "create_ticket_impl accepts whatever string the caller passes for language and writes it straight to the row. A German body with language='en' is stored as-is, and there is no language detector anywhere in the ingest or write path. The HF source rows are equally trusted at ingest time.",
    implication: "Language-filtered search silently misses tickets and surfaces wrong-language tickets. draft_reply then composes an English reply scaffold for a German customer, or vice versa. The language facet in aggregate_tickets becomes meaningless.",
    fix: "Run langdetect/fasttext-ld on subject+body at create/update time. If caller-supplied language disagrees, either reject with LANGUAGE_MISMATCH or overwrite + return a warning. Also store language_confidence for auditability."
  },
  {
    id: 161,
    title: "delete_ticket is hard-delete with no tombstone, undo window, or audit trail",
    score: 9,
    band: "P0",
    effort: "M",
    crossValidated: 1,
    category: "Data quality",
    location: "src/mcp_cst/data/store.py:489-503; tools/delete_ticket.py:26-29",
    problem: "delete_ticket issues table.delete(...) and rebuilds FTS in one shot. No deleted_at, no soft-delete flag, no trash table, no actor recorded. The DESCRIPTION even advertises this as 'irreversible within the running store'. A single misfired tool call from an over-eager LLM is unrecoverable, and downstream consumers that cached the row never learn it vanished.",
    implication: "Accidental or adversarial deletion of curated tickets (especially ones used as draft_reply grounding) is permanent. No way to answer 'who deleted ticket X and when' for compliance or post-mortems. Re-ingesting doesn't restore user-added tickets at all.",
    fix: "Add deleted_at (nullable timestamp) + deleted_by column, default all reads/searches to WHERE deleted_at IS NULL, and reserve a separate purge_ticket admin path for hard deletes. Keep tombstones for N days so consumers can detect removals."
  },
  {
    id: 163,
    title: "Tags are uncontrolled free strings — no taxonomy, no synonyms, no validation",
    score: 8,
    band: "P1",
    effort: "L",
    crossValidated: 1,
    category: "Data quality",
    location: "src/mcp_cst/data/store.py:147-150, 361-413; aggregates.py:58-66",
    problem: "Approved tag normalization (strip/lower/dedupe) doesn't address controlled vocabulary. Anyone can write tags=['billing', 'payment-issue', 'refund', 'Refund ', 'billings'] and they all survive as distinct facets. The fixture rows already show meaningless tags ('invoice', 'feature', 'refund') stamped onto a login-bug ticket.",
    implication: "aggregate_tickets(group_by='tags') returns a long-tail histogram of near-synonyms masking the real distribution. Tag filters in search_tickets miss half the relevant tickets because the caller picked the wrong synonym. Ops dashboards built on these counts mislead leadership.",
    fix: "Ship a JSON taxonomy file ({canonical → [synonyms]}). On create/update, map each tag through the synonym table and reject anything not in the canonical set (with soft-mode that warns and keeps unmapped under tags_unmapped). Expose taxonomy via schema://tags for autocomplete."
  },
  {
    id: 166,
    title: "No actor/audit field — mutations are anonymous and untraceable",
    score: 7,
    band: "P1",
    effort: "M",
    crossValidated: 2,
    category: "Data quality",
    location: "src/mcp_cst/data/store.py:96-114, 361-503; tools/{create,update,delete}_ticket.py",
    problem: "TicketRecord has no created_at, created_by, updated_at, updated_by fields. Every create/update/delete is recorded with zero attribution. MCP session does carry an identity (client name/transport), but it is never threaded through to the write path or persisted. (LLM agent #105 flagged the same forensics gap from the prompt-injection-incident angle.)",
    implication: "Impossible to answer 'which agent created this ticket', 'when was this row last edited', or 'is this anomaly spike correlated with a specific client'. Forensics on a poisoned corpus (#102 corpus poisoning) is reduced to grep-the-bodies.",
    fix: "Add created_at, updated_at (UTC ISO-8601), created_by, updated_by, mutation_source ('dataset' | 'create_ticket' | 'update_ticket'). Thread MCP session identity into the impl functions. Bump STORE_SCHEMA_VERSION to 3 with a real migration."
  },
  {
    id: 168,
    title: "Subject/body have no paste-detection — bodies pasted into subject distort BM25",
    score: 6,
    band: "P1",
    effort: "S",
    crossValidated: 1,
    category: "Data quality",
    location: "src/mcp_cst/tools/create_ticket.py:45-46; tools/update_ticket.py:44-47",
    problem: "Beyond approved #4 (max-length bounds), there's no sanity check that subject and body are different things. Dataset fixture shows real subjects are ~30 chars; values >80 chars where the first 80 chars equal the first 80 chars of body almost always indicate the caller pasted the body into both fields. text_search column concatenates them, so a bloated subject dominates the BM25 score and the embedding is biased toward the wrong field.",
    implication: "BM25 ranking quality degrades. Search snippets become unusable. Vector embedding is dominated by repeated text.",
    fix: "Warn-but-accept when len(subject) > 80 and the first 80 chars of subject equal the first 80 chars of body (paste detection). Return a `subject_truncated_from_body: true` flag in the response."
  },
  {
    id: 171,
    title: "No embedding-distribution drift detector across ingests / model swaps",
    score: 5,
    band: "P2",
    effort: "L",
    crossValidated: 1,
    category: "Data quality",
    location: "src/mcp_cst/data/store.py:208-268, 276-319",
    problem: "Manifest records embedding_model and embedding_dim; is_valid triggers a rebuild on mismatch. But there is no continuous check that the embedding distribution itself hasn't shifted. A model upgrade (same name, weights-tuned), upstream dataset contamination, or a corrupted batch can pass all structural checks while silently degrading retrieval quality.",
    implication: "Slow, invisible regression of search_tickets and draft_reply quality. By the time customers complain, weeks of bad answers have shipped. Hard to diagnose because the manifest looks healthy.",
    fix: "On every successful build, compute and persist a summary of the vector distribution (mean norm, mean pairwise cosine on a sampled 1k subset, k-NN graph degree histogram). On open, recompute on the same sample and refuse to serve if Jensen-Shannon divergence vs the manifest exceeds a threshold."
  },
  {
    id: 172,
    title: "search_tickets does not auto-prefer the query language",
    score: 5,
    band: "P2",
    effort: "S",
    crossValidated: 1,
    category: "Data quality",
    location: "src/mcp_cst/tools/search_tickets.py:33-58; retrieval/hybrid.py",
    problem: "When language=None, hybrid retrieval searches the full corpus. The German half and English half share an embedding space, but BM25 is tokenization-sensitive and German queries pulled toward English results (or vice versa) score poorly. No detection of q's language, no auto-bias toward same-language hits.",
    implication: "A German user typing 'Anmeldung funktioniert nicht' gets mostly English results because the English half is bigger and BM25 lexical overlap dominates. They cannot tell the language filter would have helped — they have to know to set it.",
    fix: "Detect q's language (langdetect, cached). If caller did not pass language, soft-bias the RRF: keep cross-language hits but boost same-language ones (+0.1 to RRF score). Expose detected language in response envelope so callers see why ranking changed."
  },
];

window.OPEN_FINDINGS = [
  {
    id: 315,
    title: "CI workflow reverted — broken on first run, needs rework before re-adding",
    score: 7,
    band: "P1",
    effort: "M",
    crossValidated: 1,
    location: "(reverted) .github/workflows/ci.yml",
    problem: "The proposed CI workflow shipped with multiple P0 issues: references a non-existent scripts/mcp_smoke_test.py (smoke job crashes), invokes ruff/mypy/pytest-cov which are not declared in [dependency-groups].dev, `uv run mypy` has no target and no [tool.mypy] config, the torch CUDA wheel index pin still applies on Linux runners (so each job downloads ~2 GB), coverage is computed but never gated or uploaded, and no `-m 'not integration'` exclude is set so collection imports integration test modules.",
    implication: "Landing it would turn main red on every push. Reverting buys time to rebuild correctly.",
    fix: "Before re-adding: (1) add ruff + mypy + pytest-cov to dev deps, (2) add [tool.mypy] strict block and pass a path, (3) add CPU-only torch index for non-Darwin in CI env, (4) add HuggingFace cache step, (5) gate unit tests with `-m 'not integration'`, (6) add a real scripts/mcp_smoke_test.py stdio client and only then enable the smoke job, (7) align ruff version with the .pre-commit pin.",
  },
  {
    id: 316,
    title: "scripts/build_smoke_store.py reverted — multiple bugs (kwarg crash, non-deterministic hash, silent overwrite)",
    score: 6,
    band: "P2",
    effort: "S",
    crossValidated: 1,
    location: "(reverted) scripts/build_smoke_store.py",
    problem: "(a) Called TicketStore.create(embedding_model=...) which is not a parameter on the current signature — crashes with TypeError on first run. (b) Used Python's hash() for the 'deterministic' fake embedder, which is salted per-process by PYTHONHASHSEED — vectors differ across runs/machines, contradicting the 'reproducible fixture' docstring. (c) Wrote the store with mode='overwrite' without any --force flag or is_valid() short-circuit — a local run pointed at the real cache silently destroys a real store.",
    implication: "Cannot be used as designed; if the kwarg were dropped it would still produce non-reproducible fixtures and risk wiping a developer's real store.",
    fix: "Drop the embedding_model kwarg, replace hash() with hashlib.blake2b digests for stable hashing, default the output to a `smoke/` subdir or require --force when overwriting, and align the fake embedder with the unit-test fake_embed helper so smoke and unit fixtures are bit-identical.",
  },
  {
    id: 317,
    title: ".pre-commit mypy hook removed — needs proper [tool.mypy] config + project-aware hook",
    score: 5,
    band: "P2",
    effort: "S",
    crossValidated: 1,
    location: "(reverted block) .pre-commit-config.yaml",
    problem: "The mirrors-mypy block ran in an isolated env that only installed pydantic/numpy/types-pyyaml — none of the real heavy deps (mcp, lancedb, polars, pyarrow, datasets, sentence-transformers, torch) — so mypy would emit 'Cannot find implementation' errors on essentially every file. It also passed --config-file=pyproject.toml, but no [tool.mypy] block exists.",
    implication: "Either spams import errors on every commit (and gets ignored) or blocks all commits with false positives until people learn to bypass — the worst of both worlds.",
    fix: "Add a [tool.mypy] strict section, then re-add the hook as `language: system` calling `uv run mypy src tests` so it shares the project's venv. Alternatively, drop pre-commit mypy and run it only in CI.",
  },
];

// Iteration-3 simplification pass: three senior-architect agents swept the
// 2.3k LoC source tree for behavior-preserving simplifications. Findings are
// strictly *no observable change* — every entry carries a behaviorPreserved
// justification. Headline result: ~321 LoC across four files is dead code.
window.SIMPLIFY_FINDINGS = [];
