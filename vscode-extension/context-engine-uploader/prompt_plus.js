function createPromptPlusManager(deps) {
  const vscode = deps.vscode;
  const spawn = deps.spawn;
  const path = deps.path;
  const fs = deps.fs;
  const log = deps.log;

  const extensionRoot = deps.extensionRoot;
  const getEffectiveConfig = deps.getEffectiveConfig;
  const getTargetPath = deps.getTargetPath;
  const getWorkspaceFolderPath = deps.getWorkspaceFolderPath;
  const detectDefaultTargetPath = deps.detectDefaultTargetPath;
  const resolveBridgeHttpUrl = deps.resolveBridgeHttpUrl;
  const getPythonOverridePath = deps.getPythonOverridePath;
  const appendOutput = deps.appendOutput;

  function dispose() {
    // No timers/processes owned.
  }

  function getConfiguredPythonPath() {
    try {
      const cfg = getEffectiveConfig();
      const configured = (cfg.get('pythonPath') || 'python3').trim();
      const override = typeof getPythonOverridePath === 'function' ? getPythonOverridePath() : undefined;
      if (override && fs.existsSync(override)) {
        return override;
      }
      return configured || 'python3';
    } catch (_) {
      const override = typeof getPythonOverridePath === 'function' ? getPythonOverridePath() : undefined;
      if (override && fs.existsSync(override)) {
        return override;
      }
      return 'python3';
    }
  }

  function getConfiguredDecoderUrl() {
    try {
      const cfg = getEffectiveConfig();
      const configured = (cfg.get('decoderUrl') || '').trim();
      return configured || 'http://localhost:8081';
    } catch (_) {
      return 'http://localhost:8081';
    }
  }

  function resolveCtxScriptPath() {
    const candidates = [];
    candidates.push(path.join(extensionRoot, 'scripts', 'ctx.py'));
    candidates.push(path.join(extensionRoot, 'ctx.py'));
    const wsFolder = getWorkspaceFolderPath();
    if (wsFolder) {
      candidates.push(path.join(wsFolder, 'scripts', 'ctx.py'));
      candidates.push(path.join(wsFolder, 'ctx.py'));
    }
    candidates.push(path.resolve(extensionRoot, '..', '..', 'scripts', 'ctx.py'));

    for (const candidate of candidates) {
      if (candidate && fs.existsSync(candidate)) {
        return path.resolve(candidate);
      }
    }

    vscode.window.showErrorMessage('Context Engine Uploader: ctx.py not found (expected scripts/ctx.py).');
    return undefined;
  }

  async function enhanceSelectionWithUnicorn() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
      vscode.window.showWarningMessage('Open a file and select text to enhance with Prompt+.');
      return;
    }
    const selection = editor.selections && editor.selections.length ? editor.selections[0] : editor.selection;
    const text = selection && !selection.isEmpty ? editor.document.getText(selection) : '';
    if (!text || !text.trim()) {
      vscode.window.showWarningMessage('Select text to enhance with Prompt+.');
      return;
    }

    const ctxScript = resolveCtxScriptPath();
    if (!ctxScript) {
      return;
    }

    const pythonPath = getConfiguredPythonPath();
    const projectRoot = path.dirname(path.dirname(ctxScript));
    const env = { ...process.env };
    env.PYTHONUNBUFFERED = '1';
    const decoderUrl = getConfiguredDecoderUrl();
    if (decoderUrl) {
      env.DECODER_URL = decoderUrl;
    }

    try {
      const cfg = getEffectiveConfig();
      const transportModeRaw = cfg.get('mcpTransportMode') || 'sse-remote';
      const serverModeRaw = cfg.get('mcpServerMode') || 'bridge';
      const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
      const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
      let idxUrlRaw = (cfg.get('ctxIndexerUrl') || cfg.get('mcpIndexerUrl') || '').trim();
      if (serverMode === 'bridge' && transportMode === 'http') {
        const bridgeUrl = resolveBridgeHttpUrl();
        if (bridgeUrl) {
          idxUrlRaw = bridgeUrl;
        }
      }
      if (idxUrlRaw) {
        env.MCP_INDEXER_URL = idxUrlRaw;
      }
    } catch (_) {
      // ignore config read failures; fall back to defaults
    }

    try {
      const cfg = getEffectiveConfig();
      const useGpuDecoder = cfg.get('useGpuDecoder', false);
      if (useGpuDecoder) {
        env.USE_GPU_DECODER = '1';
      }
      let ctxWorkspaceDir;
      try {
        ctxWorkspaceDir = getTargetPath(cfg);
      } catch (error) {
        ctxWorkspaceDir = undefined;
      }
      if (!ctxWorkspaceDir) {
        const wsFolder = getWorkspaceFolderPath();
        if (wsFolder) {
          ctxWorkspaceDir = detectDefaultTargetPath(wsFolder);
        }
      }
      if (ctxWorkspaceDir && typeof ctxWorkspaceDir === 'string' && fs.existsSync(ctxWorkspaceDir)) {
        env.CTX_WORKSPACE_DIR = ctxWorkspaceDir;
      }
    } catch (_) {
      // ignore config read failures; fall back to defaults
    }

    const existingPyPath = env.PYTHONPATH || '';
    env.PYTHONPATH = existingPyPath ? `${projectRoot}${path.delimiter}${existingPyPath}` : projectRoot;

    log(`Running Prompt+ via ctx.py at ${ctxScript}`);

    return new Promise((resolve) => {
      const args = [ctxScript, '--unicorn', text];
      const child = spawn(pythonPath, args, { cwd: projectRoot, env });
      let stdout = '';
      let stderr = '';

      if (child.stdout) {
        child.stdout.on('data', data => {
          stdout += data.toString();
        });
      }
      if (child.stderr) {
        child.stderr.on('data', data => {
          const chunk = data.toString();
          stderr += chunk;
          try {
            if (typeof appendOutput === 'function') {
              appendOutput(`[prompt+] ${chunk}`);
            }
          } catch (_) {
            // ignore
          }
        });
      }

      child.on('error', error => {
        log(`Prompt+ spawn failed: ${error instanceof Error ? error.message : String(error)}`);
        vscode.window.showErrorMessage('Prompt+ failed to start. Check Python path.');
        resolve();
      });

      child.on('close', code => {
        if (code !== 0) {
          log(`Prompt+ exited with code ${code}${stderr ? `: ${stderr.trim()}` : ''}`);
          vscode.window.showErrorMessage('Prompt+ failed. See output for details.');
          return resolve();
        }
        const enhanced = (stdout || '').trim();
        if (!enhanced) {
          vscode.window.showWarningMessage('Prompt+ returned no output.');
          return resolve();
        }
        editor.edit(editBuilder => editBuilder.replace(selection, enhanced)).then(ok => {
          if (ok) {
            vscode.window.showInformationMessage('Prompt+ applied (Unicorn Mode).');
          } else {
            vscode.window.showErrorMessage('Prompt+ could not update the selection.');
          }
          resolve();
        });
      });
    });
  }

  return {
    enhanceSelectionWithUnicorn,
    dispose,
  };
}

module.exports = {
  createPromptPlusManager,
};
