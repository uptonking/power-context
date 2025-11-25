Context Engine Uploader
=======================

Features
--------
- Runs a force sync (`Index Codebase`) followed by watch mode to keep a remote Context Engine instance in sync with your workspace.
- Auto-detects the first workspace folder as the default target path, storing it in workspace settings so the extension is portable.
- Provides commands and a status-bar button:
  - `Context Engine Uploader: Index Codebase` – force sync + watch with spinner feedback.
  - `Context Engine Uploader: Start/Stop/Restart` for manual lifecycle control.
- Streams detailed logs into the `Context Engine Upload` output channel for visibility into both force sync and watch phases.
- Status bar shows current state (indexing spinner, purple watching state) so you always know if uploads are active.

Configuration
-------------
- `Run On Startup` auto-triggers force sync + watch after VS Code finishes loading.
- `Python Path`, `Endpoint`, `Extra Force Args`, `Extra Watch Args`, and `Interval Seconds` can be tuned via standard VS Code settings.
- `Target Path` is auto-filled from the workspace but can be overridden if you need to upload a different folder.
- **Python dependencies:** the extension runs the standalone upload client via your configured `pythonPath`. Ensure the interpreter has `requests`, `urllib3`, and `charset_normalizer` installed. Run `python3 -m pip install requests urllib3 charset_normalizer` (or replace `python3` with your configured path) before starting the uploader.
- **Path mapping:** `Host Root` + `Container Root` control how local paths are rewritten before reaching the remote service. By default the host root mirrors your `Target Path` and the container root is `/work`, which keeps Windows paths working without extra config.
- **Prompt+ decoder:** set `Context Engine Uploader: Decoder Url` (default `http://localhost:8081`, auto-appends `/completion`) to point at your local llama.cpp decoder. For Ollama, set it to `http://localhost:11434/api/chat`. Enable `Context Engine Uploader: Use Glm Decoder` to set `REFRAG_RUNTIME=glm` for GLM backends. Turn on `Use Gpu Decoder` to set `USE_GPU_DECODER=1` so ctx.py prefers the GPU llama.cpp sidecar.
- **Claude Code MCP config:** `MCP Indexer Url` and `MCP Memory Url` control the URLs written into the project-local `.mcp.json` when you run the `Write MCP Config` command. This is only for configuring Claude Code MCP clients; other MCP integrations can be added separately later.
- **CTX + GLM settings:**
  - `contextEngineUploader.ctxIndexerUrl` is copied into `.env` (as `MCP_INDEXER_URL`) so the embedded `ctx.py` knows which MCP indexer to call when enhancing prompts.
  - `contextEngineUploader.glmApiKey`, `glmApiBase`, and `glmModel` are used when scaffolding `ctx_config.json`/`.env` to pre-fill GLM decoder options. Existing non-placeholder values are preserved, so you can override them in the files at any time.
- **Context scaffolding:**
  - `contextEngineUploader.scaffoldCtxConfig` (default `true`) controls whether the extension keeps a minimal `ctx_config.json` + `.env` in sync with your workspace. When enabled, running `Write MCP Config` or `Write CTX Config` will seed the files from the bundled `env.example` and inferred collection name, only overwriting placeholder/empty values.
  - `contextEngineUploader.surfaceQdrantCollectionHint` gates whether the Claude hook adds a hint line with the Qdrant collection ID when ctx is enhancing prompts. This setting is also respected when the extension writes `.claude/settings.local.json`.

Workspace-level ctx integration
-------------------------------
- The VSIX bundles an `env.example` template plus the ctx hook/CLI so you can dogfood the workflow without copying files manually.
- When scaffolding is enabled (see above), running the `Context Engine Uploader: Write CTX Config (ctx_config.json/.env)` command will:
  - Infer the collection name from the standalone upload client (`--show-mapping`).
  - Create or update `ctx_config.json` with that collection and sensible defaults (GLM runtime, `default_mode`, `require_context`, etc.).
  - Create or update `.env` from the bundled template, ensuring CTX-critical values such as `MULTI_REPO_MODE=1`, `REFRAG_RUNTIME=glm`, and `REFRAG_DECODER=1` are set. Non-placeholder values (e.g., a real `GLM_API_KEY`) are left alone.
- You still own the files: if you need a custom value, edit `.env` or `ctx_config.json` directly. The scaffolder only touches keys that are missing, empty, or obviously placeholders.
- The Claude hook + ctx prompt enhancement is currently wired for Linux/dev-remote environments only. On other platforms, MCP config and uploading still work, but the automatic prompt rewrite hook is disabled.

Commands
--------
- Command Palette → “Context Engine Uploader” to access Start/Stop/Restart/Index Codebase.
- Status-bar button (`Index Codebase`) mirrors the same behavior and displays progress.
- `Context Engine Uploader: Write MCP Config (.mcp.json)` writes or updates a project-local `.mcp.json` with MCP server entries for the Qdrant indexer and memory/search endpoints, using the configured MCP URLs.
- `Context Engine Uploader: Write CTX Config (ctx_config.json/.env)` scaffolds the ctx config + env files as described above. This command runs automatically after `Write MCP Config` if scaffolding is enabled, but it is also exposed in the Command Palette for manual use.

Logs
----
Open `View → Output → Context Engine Upload` to see the remote uploader’s stdout/stderr, including any errors from the Python client.
