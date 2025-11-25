Context Engine Uploader VS Code Extension
=========================================

Build Prerequisites
-------------------
- Node.js 18+ and npm
- Python 3 available on PATH for runtime testing
- VS Code Extension Manager `vsce` (`npm install -g @vscode/vsce`) or run via `npx`

Install Dependencies
--------------------
```bash
cd vscode-extension/context-engine-uploader
npm install
```

Package the Extension
---------------------
```bash
cd vscode-extension/context-engine-uploader
npx vsce package
```
This emits a `.vsix` file such as `context-engine-uploader-0.1.0.vsix`.

Test Locally
------------
1. In VS Code, open the command palette and select `Developer: Install Extension from Location...`.
2. Pick the generated `.vsix`.
3. Reload the window when prompted.

Key Settings After Install
--------------------------
- `Context Engine Upload` output channel shows force-sync and watch logs.
- `Context Engine Uploader: Index Codebase` command or status bar button runs a force sync followed by watch.
- Configure `contextEngineUploader.targetPath`, `endpoint`, and other options under Settings → Extensions → Context Engine Uploader.

## Prerequisites
Python 3.8+ must be available on the host so the bundled client can run.

## Configuration

All settings live under `Context Engine Uploader` in the VS Code settings UI or `settings.json`.

| Setting | Description |
| --- | --- |
| `contextEngineUploader.runOnStartup` | Runs the force sync automatically after VS Code starts, then starts watch mode. Leave enabled to mirror the old manual workflow. |
| `contextEngineUploader.pythonPath` | Python executable to use (`python3` by default). |
| `contextEngineUploader.scriptWorkingDirectory` | Optional override for the folder that contains `standalone_upload_client.py`. Leave blank to use the extension’s own copy. |
| `contextEngineUploader.decoderUrl` | Override `DECODER_URL` passed into `scripts/ctx.py` when running Prompt+. Defaults to local llama.cpp (`http://localhost:8081`, auto-appends `/completion`). Use `http://localhost:11434/api/chat` for Ollama. |
| `contextEngineUploader.useGlmDecoder` | Set `REFRAG_RUNTIME=glm` for Prompt+ to hit GLM instead of Ollama/llama.cpp. |
| `contextEngineUploader.useGpuDecoder` | Set `USE_GPU_DECODER=1` so ctx.py prefers the GPU llama.cpp sidecar. |
| `contextEngineUploader.targetPath` | Absolute path that should be passed to `--path` (for example `/users/mycode`). |
| `contextEngineUploader.endpoint` | Remote endpoint passed to `--endpoint`, defaulting to `http://localhost:8004`. |
| `contextEngineUploader.intervalSeconds` | Poll interval for watch mode. Set to `5` to match the previous command file. |
| `contextEngineUploader.extraForceArgs` | Optional string array appended to the force invocation. Leave empty for the standard workflow. |
| `contextEngineUploader.extraWatchArgs` | Optional string array appended to the watch invocation. |

## Commands and lifecycle

- `Context Engine Uploader: Start` — executes the initial `--force` followed by `--watch` using the configured settings.
- `Context Engine Uploader: Stop` — terminates any running upload client processes.
- `Context Engine Uploader: Restart` — stops current processes and re-runs the startup sequence.
- `Context Engine Uploader: Show Upload Service Logs` — opens a terminal and tails `docker compose logs -f upload_service`.
- `Context Engine Uploader: Prompt+ (Unicorn Mode)` — runs `scripts/ctx.py --unicorn` on your current selection and replaces it with the enhanced prompt (status bar button).

The extension logs all subprocess output to the **Context Engine Upload** output channel so you can confirm uploads without leaving VS Code. The watch process shuts down automatically when VS Code exits or when you run the Stop command.
