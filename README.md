# mcp-customer-support-tickets

A read-only Model Context Protocol (MCP) server that exposes the
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
dataset to MCP-capable clients (Claude Code, Claude Desktop, Codex, Cursor).

- **Search** by hybrid BM25 + vector across ~62k EN/DE support tickets.
- **Fetch** any ticket verbatim by id.
- **Aggregate** counts by queue, priority, language, type, or tags.
- **Draft** a reply grounded in up to 5 prior similar tickets+answers (>=70% cosine similarity).

## Install (via MCP client)

```json
{
  "mcpServers": {
    "customer-support-tickets": {
      "command": "uvx",
      "args": ["mcp-customer-support-tickets"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "MCP_CST_DATASET_REVISION": "main"
      }
    }
  }
}
```

## Local development

```sh
git clone <repo-url> mcp-customer-support-tickets
cd mcp-customer-support-tickets
uv sync
uv run mcp-customer-support-tickets   # stdio server
uv run pytest                          # tests
```

## Environment variables

| Var | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | one of these for `draft_reply` | Drafting via Claude (preferred). |
| `OPENAI_API_KEY` | as above | Drafting via GPT (fallback). |
| `MCP_CST_DATASET_REVISION` | no | HF dataset revision pin. |
| `MCP_CST_CACHE_DIR` | no | Override cache dir. |
| `RERANK` | no | `true` enables cross-encoder rerank (deferred; stub for now). |

## First-run notes

The first time the server starts, it downloads the embedding model
(~120 MB) and the HF Parquet, then runs an embedding pass over ~62k
tickets (~2 min on CPU). Everything is cached on disk, keyed by
dataset revision and model id. Subsequent starts are sub-second.

## License

Server code: MIT. Dataset (CC-BY-NC-4.0) is non-commercial — see
`server_info` for the license string surfaced at runtime.
