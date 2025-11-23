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

Commands
--------
- Command Palette → “Context Engine Uploader” to access Start/Stop/Restart/Index Codebase.
- Status-bar button (`Index Codebase`) mirrors the same behavior and displays progress.
- Status-bar button (`Prompt+`) runs the bundled `scripts/ctx.py --unicorn` on your current selection and replaces it with the enhanced prompt.

Logs
----
Open `View → Output → Context Engine Upload` to see the remote uploader’s stdout/stderr, including any errors from the Python client.
