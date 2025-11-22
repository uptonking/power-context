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

Commands
--------
- Command Palette → “Context Engine Uploader” to access Start/Stop/Restart/Index Codebase.
- Status-bar button (`Index Codebase`) mirrors the same behavior and displays progress.

Logs
----
Open `View → Output → Context Engine Upload` to see the remote uploader’s stdout/stderr, including any errors from the Python client.
