function createLogsTerminalManager(deps) {
  const vscode = deps.vscode;
  const fs = deps.fs;
  const log = deps.log;

  const getEffectiveConfig = deps.getEffectiveConfig;
  const getWorkspaceFolderPath = deps.getWorkspaceFolderPath;

  let logsTerminal;
  let logTailActive = false;

  function open() {
    try {
      const cfg = getEffectiveConfig();
      const wsPath = getWorkspaceFolderPath() || (cfg.get('targetPath') || '');
      const cwd = (wsPath && typeof wsPath === 'string' && fs.existsSync(wsPath)) ? wsPath : undefined;
      if (logsTerminal && logsTerminal.exitStatus) {
        logsTerminal = undefined;
        logTailActive = false;
      }
      if (!logsTerminal) {
        logsTerminal = vscode.window.createTerminal({ name: 'Context Engine Upload Service Logs (Docker)', cwd: cwd ? vscode.Uri.file(cwd) : undefined });
        logTailActive = false;
      }
      logsTerminal.show(true);
      if (!logTailActive) {
        logsTerminal.sendText('docker compose logs -f upload_service', true);
        logTailActive = true;
      }
    } catch (e) {
      log(`Unable to open logs terminal: ${e && e.message ? e.message : String(e)}`);
    }
  }

  function handleDidCloseTerminal(term) {
    if (term === logsTerminal) {
      logsTerminal = undefined;
      logTailActive = false;
    }
  }

  function dispose() {
    try {
      logTailActive = false;
      logsTerminal = undefined;
    } catch (_) {
      // ignore
    }
  }

  return {
    open,
    handleDidCloseTerminal,
    dispose,
  };
}

module.exports = {
  createLogsTerminalManager,
};
