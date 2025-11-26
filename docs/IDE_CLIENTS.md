# IDE & Client Configuration

Connect your IDE to a running Context-Engine stack. No need to clone this repo into your project.

**Documentation:** [README](../README.md) · [Configuration](CONFIGURATION.md) · [IDE Clients](IDE_CLIENTS.md) · [MCP API](MCP_API.md) · [ctx CLI](CTX_CLI.md) · [Memory Guide](MEMORY_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Multi-Repo](MULTI_REPO_COLLECTIONS.md) · [Kubernetes](../deploy/kubernetes/README.md) · [VS Code Extension](vscode-extension.md) · [Troubleshooting](TROUBLESHOOTING.md) · [Development](DEVELOPMENT.md)

---

**On this page:**
- [Quick Start](#quick-start)
- [Supported Clients](#supported-clients)
- [SSE Clients](#sse-clients-port-80008001)
- [RMCP Clients](#rmcp-clients-port-80028003)
- [Mixed Transport](#mixed-transport-examples)
- [Remote Server](#remote-server)
- [Verification](#verification)

---

## Quick Start

**Prerequisites:** Context-Engine running somewhere (localhost, remote server, or Kubernetes).

**Minimal config** — add to your IDE's MCP settings:
```json
{
  "mcpServers": {
    "context-engine": { "url": "http://localhost:8001/sse" }
  }
}
```

Replace `localhost` with your server IP/hostname for remote setups.

---

## Supported Clients

| Client | Transport | Notes |
|--------|-----------|-------|
| Roo | SSE/RMCP | Both SSE and RMCP connections |
| Cline | SSE/RMCP | Both SSE and RMCP connections |
| Windsurf | SSE/RMCP | Both SSE and RMCP connections |
| Zed | SSE | Uses mcp-remote bridge |
| Kiro | SSE | Uses mcp-remote bridge |
| Qodo | RMCP | Direct HTTP endpoints |
| OpenAI Codex | RMCP | TOML config |
| Augment | SSE | Simple JSON configs |
| AmpCode | SSE | Simple URL for SSE endpoints |
| Claude Code CLI | SSE | Simple JSON configs |

---

## SSE Clients (port 8000/8001)

### Roo / Cline / Windsurf

```json
{
  "mcpServers": {
    "memory": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

### Augment

```json
{
  "mcpServers": {
    "memory": { "type": "sse", "url": "http://localhost:8000/sse", "disabled": false },
    "qdrant-indexer": { "type": "sse", "url": "http://localhost:8001/sse", "disabled": false }
  }
}
```

### Kiro

Create `.kiro/settings/mcp.json` in your workspace:

```json
{
  "mcpServers": {
    "qdrant-indexer": { "command": "npx", "args": ["mcp-remote", "http://localhost:8001/sse", "--transport", "sse-only"] },
    "memory": { "command": "npx", "args": ["mcp-remote", "http://localhost:8000/sse", "--transport", "sse-only"] }
  }
}
```

**Notes:**
- Kiro expects command/args (stdio). `mcp-remote` bridges to remote SSE endpoints.
- If `npx` prompts in your environment, add `-y` right after `npx`.
- Workspace config overrides user-level config (`~/.kiro/settings/mcp.json`).

**Troubleshooting:**
- Error: "Enabled MCP Server must specify a command, ignoring." → Use the command/args form; do not use type:url in Kiro.

### Zed

Add to your Zed `settings.json` (Command Palette → "Settings: Open Settings (JSON)"):

```json
{
  "qdrant-indexer": {
    "command": "npx",
    "args": ["mcp-remote", "http://localhost:8001/sse", "--transport", "sse-only"],
    "env": {}
  }
}
```

**Notes:**
- Zed expects MCP servers at the root level of settings.json
- Uses command/args (stdio). mcp-remote bridges to remote SSE endpoints
- If npx prompts, add `-y` right after npx: `"args": ["-y", "mcp-remote", ...]`

**Alternative (direct HTTP):**
```json
{
  "qdrant-indexer": {
    "type": "http",
    "url": "http://localhost:8001/sse"
  }
}
```

---

## RMCP Clients (port 8002/8003)

### Qodo

Add each MCP tool separately through the UI:

**Tool 1 - memory:**
```json
{
  "memory": { "url": "http://localhost:8002/mcp" }
}
```

**Tool 2 - qdrant-indexer:**
```json
{
  "qdrant-indexer": { "url": "http://localhost:8003/mcp" }
}
```

**Note:** Qodo can talk to RMCP endpoints directly, no `mcp-remote` wrapper needed.

### OpenAI Codex

TOML configuration:

```toml
experimental_use_rmcp_client = true

[mcp_servers.memory_http]
url = "http://127.0.0.1:8002/mcp"

[mcp_servers.qdrant_indexer_http]
url = "http://127.0.0.1:8003/mcp"
```

---

## Mixed Transport (stdio + SSE)

### Windsurf/Cursor

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uvx",
      "args": ["mcp-server-qdrant"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTION_NAME": "my-collection",
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5"
      },
      "disabled": false
    }
  }
}
```

---

## Remote Server

When Context-Engine runs on a remote server (e.g., `context.yourcompany.com`):

```json
{
  "mcpServers": {
    "context-engine": { "url": "http://context.yourcompany.com:8001/sse" }
  }
}
```

**Indexing your local project to the remote server:**
```bash
# Using VS Code extension (recommended)
# Install vscode-context-engine, configure server URL, click "Upload Workspace"

# Using CLI
scripts/remote_upload_client.py --server http://context.yourcompany.com:9090 --path /your/project
```

> See [docs/MULTI_REPO_COLLECTIONS.md](MULTI_REPO_COLLECTIONS.md) for multi-repo and Kubernetes deployment.

---

## Important Notes for IDE Agents

- **Do not send null values** to MCP tools. Omit the field or pass an empty string "" instead.
- **qdrant-index examples:**
  - `{"subdir":"","recreate":false,"collection":"my-collection","repo_name":"workspace"}`
  - `{"subdir":"scripts","recreate":true}`
- For indexing repo root with no params, use `qdrant_index_root` (zero-arg) or call `qdrant-index` with `subdir:""`.

---

## Verification

After configuring, you should see tools from both servers:
- `store`, `find` (Memory)
- `repo_search`, `code_search`, `context_search`, `context_answer` (Indexer)
- `qdrant_list`, `qdrant_index`, `qdrant_prune`, `qdrant_status` (Indexer)

Test connectivity:
- Call `qdrant_list` to confirm Qdrant connectivity
- Call `qdrant_index` with `{ "subdir": "scripts", "recreate": true }` to test indexing
- Call `context_search` with `{ "include_memories": true }` to test memory blending

