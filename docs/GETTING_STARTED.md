# Getting Started (VS Code + Dev-Remote)

**Documentation:** [README](../README.md) · [Getting Started](GETTING_STARTED.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

This guide is for developers who want the lowest-friction way to try Context-Engine:

- Run a single Docker Compose stack
- Install one VS Code extension
- Open a project and start asking questions about your code

---

## 1. Prerequisites

- **Docker** (Docker Desktop or equivalent)
- **Git**
- **VS Code** (to use the Context Engine Uploader extension)
- **An MCP-enabled IDE or client** to talk to Context-Engine via MCP, for example:
  - Claude Code, Windsurf, Cursor, Roo, Cline, Zed (via `mcp-remote`), etc.

CLI-only workflows using `ctx.py` and hybrid search tools are supported but are documented separately. This guide assumes you will talk to Context-Engine through an MCP-enabled assistant.

You do *not* need to clone this repo into every project. You run Context-Engine once, then point it at whatever code you care about.

---

## 2. Start the dev-remote stack

In a terminal (from wherever you want the stack to live):

```bash
git clone https://github.com/m1rl0k/Context-Engine.git
cd Context-Engine

# Start the dev-remote stack (Qdrant, MCPs, upload service, watcher, etc.)
docker compose -f docker-compose.yml up -d
```

This brings up, on your host machine:

- Qdrant on `http://localhost:6333`
- Memory MCP:
  - SSE: `http://localhost:8000/sse`
  - HTTP / RMCP: `http://localhost:8002/mcp`
- Indexer MCP:
  - SSE: `http://localhost:8001/sse`
  - HTTP / RMCP: `http://localhost:8003/mcp`
- Upload service (used by the VS Code extension) on `http://localhost:8004`

---

## 3. Index your code (via VS Code extension)

In the dev-remote flow, you normally do **not** run the indexer manually.

Instead, the VS Code extension uploads your workspace to the dev-remote stack, and the `indexer` + `watcher` services handle:

- Mirroring your project into the container under `/work` (in dev-workspace folder)
- Walking files, chunking them, and writing vectors + metadata into Qdrant
- Tracking per-file hashes under `.codebase` so unchanged files are skipped

If you prefer CLI-based indexing, see the README and advanced docs (Multi-Repo, Kubernetes, etc.) for `docker compose run --rm indexer` usage.

---

## 4. Connect your IDE

The normal way to use Context-Engine is through an MCP-enabled assistant. The simplest config is via the HTTP MCP endpoints below; the VS Code extension can also scaffold these configs for you.

### Example: Claude Code / generic RMCP client

Add to your MCP config (for example `claude_code_config.json`):

```json
{
  "mcpServers": {
    "memory": { "url": "http://localhost:8002/mcp" },
    "qdrant-indexer": { "url": "http://localhost:8003/mcp" }
  }
}
```

### Example: Windsurf / Cursor (SSE)

If your client prefers SSE:

```json
{
  "mcpServers": {
    "memory": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

See [docs/IDE_CLIENTS.md](IDE_CLIENTS.md) for copy-paste configs for specific IDEs.

---

## 5. Try a few example queries

Once your IDE MCP's are connected and indexing has finished, just *ask your assistant* questions like the ones below; it will call the MCP tools on your behalf.

### Code search examples (qdrant-indexer)

Ask your assistant to run something like:

- "Find places where remote uploads are retried"
- "Show functions that call ingest_code.index_repo"
- "Search for performance bottlenecks in the upload_service script"

Under the hood, the client will call tools such as `repo_search` or `context_answer` on the `qdrant-indexer` server.

### Commit history / lineage examples (qdrant-indexer)

If you have git history ingestion enabled, you can also ask:

- "When did we add git history ingestion to the upload client?"
- "When did we optimize git history collection to fetch only commits since last upload?"
- "What commits mention Windows UnicodeDecodeError in git history collection?"

These eventually call `search_commits_for` and related tools, which use the commit index and lineage summaries.

---

## 6. VS Code extension (recommended)

For this dev-remote flow, the **Context Engine Uploader** VS Code extension is the primary way to sync and index code:

- Install from the Marketplace: <https://marketplace.visualstudio.com/items?itemName=context-engine.context-engine-uploader>
- Point it at your project (or let it auto-detect the current workspace root)
- Configure the upload endpoint to `http://localhost:8004`
- Start the uploader; it will force an initial upload and then watch for changes

Under **Settings → Extensions → Context Engine Uploader** you will typically use:

- `endpoint`: `http://localhost:8004` (dev-remote upload_service)
- Optional MCP settings: `mcpIndexerUrl`, `mcpMemoryUrl`, and `mcpTransportMode` (`sse-remote` or `http`) pointing at the dev-remote memory/indexer URLs listed above
- Optional auto-config: enable `mcpClaudeEnabled` / `mcpWindsurfEnabled` and `autoWriteMcpConfigOnStartup` to have the extension write Claude Code/Windsurf MCP configs (and an optional `/ctx` hook) for you

Once running, your code is kept in sync with the dev-remote stack without any manual indexer commands.

## 7. Where to go next

Once you have the basic flow working (dev-remote stack up → VS Code extension syncing → IDE connected via MCP → run a few queries), you can explore:

- [Configuration](CONFIGURATION.md) — environment variables and tuning knobs
- [IDE Clients](IDE_CLIENTS.md) — detailed configs for specific IDEs
- [Multi-Repo](MULTI_REPO_COLLECTIONS.md) — multi-repo collections, remote servers, Kubernetes
- [Memory Guide](MEMORY_GUIDE.md) — how to use the Memory MCP server alongside the indexer
- [Architecture](ARCHITECTURE.md) — deeper dive into how the components fit together
- [ctx CLI](CTX_CLI.md) — CLI workflows and prompt hooks; see `ctx/claude-hook-example.json` for a Claude Code `/ctx` hook wired to `ctx.py`
- [VS Code Extension](vscode-extension.md) — full extension capabilities and settings
