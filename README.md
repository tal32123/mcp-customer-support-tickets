# mcp-customer-support-tickets

A read-only Model Context Protocol (MCP) server that exposes the
[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
dataset to MCP-capable clients (Claude Code, Claude Desktop, Codex, Cursor).

- **Search** by hybrid BM25 + vector across ~62k EN/DE support tickets.
- **Fetch** any ticket verbatim by id.
- **Aggregate** counts by queue, priority, language, type, or tags.
- **Assemble** a grounded draft-reply prompt: target ticket + up to 5 prior tickets+answers (cosine >= 0.70) + a type-aware scaffold the caller's LLM fills in. No API key needed.

## Install (via MCP client)

```json
{
  "mcpServers": {
    "customer-support-tickets": {
      "command": "uvx",
      "args": ["mcp-customer-support-tickets"],
      "env": {
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
