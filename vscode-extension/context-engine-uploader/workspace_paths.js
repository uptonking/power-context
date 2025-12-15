function createWorkspacePathUtils(deps) {
  const vscode = deps.vscode;
  const path = deps.path;
  const fs = deps.fs;
  const log = deps.log;
  const updateStatusBarTooltip = deps.updateStatusBarTooltip;

  let lastAutoDetectLogKey = '';
  let lastResolvedTargetLogKey = '';

  function getWorkspaceFolderPath() {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders || !folders.length) {
      return undefined;
    }
    return folders[0].uri.fsPath;
  }

  function looksLikeRepoRoot(dirPath) {
    try {
      const codebaseStatePath = path.join(dirPath, '.codebase', 'state.json');
      const gitDir = path.join(dirPath, '.git');
      if (fs.existsSync(codebaseStatePath) || fs.existsSync(gitDir)) {
        return true;
      }
    } catch (error) {
      log(`Repo root detection failed for ${dirPath}: ${error instanceof Error ? error.message : String(error)}`);
    }
    return false;
  }

  function detectDefaultTargetPath(workspaceFolderPath) {
    try {
      const resolved = path.resolve(workspaceFolderPath);
      if (!fs.existsSync(resolved)) {
        return workspaceFolderPath;
      }
      const rootLooksLikeRepo = looksLikeRepoRoot(resolved);
      let entries;
      try {
        entries = fs.readdirSync(resolved);
      } catch (error) {
        log(`Auto targetPath discovery failed to read workspace folder: ${error instanceof Error ? error.message : String(error)}`);
        return resolved;
      }
      const candidates = [];
      for (const name of entries) {
        const fullPath = path.join(resolved, name);
        let stats;
        try {
          stats = fs.statSync(fullPath);
        } catch (_) {
          continue;
        }
        if (!stats.isDirectory()) {
          continue;
        }
        if (looksLikeRepoRoot(fullPath)) {
          candidates.push(path.resolve(fullPath));
        }
      }
      if (candidates.length === 1) {
        const detected = candidates[0];
        const key = `${resolved}::${detected}`;
        if (key !== lastAutoDetectLogKey) {
          lastAutoDetectLogKey = key;
          log(`Target path auto-detected as ${detected} (under workspace folder).`);
        }
        return detected;
      }
      if (rootLooksLikeRepo) {
        if (candidates.length > 1) {
          const key = `${resolved}::multiple`;
          if (key !== lastAutoDetectLogKey) {
            lastAutoDetectLogKey = key;
            log('Auto targetPath discovery found multiple candidate repos under workspace; using workspace folder instead.');
          }
        }
        return resolved;
      }
      if (candidates.length > 1) {
        const key = `${resolved}::multiple`;
        if (key !== lastAutoDetectLogKey) {
          lastAutoDetectLogKey = key;
          log('Auto targetPath discovery found multiple candidate repos under workspace; using workspace folder instead.');
        }
      }
      return resolved;
    } catch (error) {
      log(`Auto targetPath discovery failed: ${error instanceof Error ? error.message : String(error)}`);
      return workspaceFolderPath;
    }
  }

  function resolveTargetPathFromConfig(config) {
    let inspected;
    try {
      if (typeof config.inspect === 'function') {
        inspected = config.inspect('targetPath');
      }
    } catch (error) {
      inspected = undefined;
    }
    let targetPath = (config.get('targetPath') || '').trim();
    const metadata = inspected || {};
    if (targetPath) {
      return { path: targetPath, inspected: metadata };
    }
    const folderPath = getWorkspaceFolderPath();
    if (!folderPath) {
      return { path: undefined, inspected: metadata };
    }
    const autoTarget = detectDefaultTargetPath(folderPath);
    return { path: autoTarget, inspected: metadata, inferred: true };
  }

  function getTargetPath(config) {
    const result = resolveTargetPathFromConfig(config);
    let targetPath = result.path;
    const inspected = result.inspected;
    if (inspected && targetPath) {
      let sourceLabel = 'default';
      if (inspected.workspaceFolderValue !== undefined) {
        sourceLabel = 'workspaceFolder';
      } else if (inspected.workspaceValue !== undefined) {
        sourceLabel = 'workspace';
      } else if (inspected.globalValue !== undefined) {
        sourceLabel = 'user';
      }
      const key = `${sourceLabel}::${targetPath}`;
      if (key !== lastResolvedTargetLogKey) {
        lastResolvedTargetLogKey = key;
        log(`Target path resolved to ${targetPath} (source: ${sourceLabel} settings)`);
      }
      if (inspected.globalValue !== undefined && inspected.workspaceValue !== undefined && inspected.globalValue !== inspected.workspaceValue) {
        log('Target path has different user and workspace values; using workspace value. Update workspace settings (e.g. .vscode/settings.json) to change it.');
      }
    }
    if (targetPath) {
      const folderPath = getWorkspaceFolderPath();
      if (folderPath && !path.isAbsolute(targetPath)) {
        targetPath = path.resolve(folderPath, targetPath);
      }
      updateStatusBarTooltip(targetPath);
      return targetPath;
    }
    vscode.window.showErrorMessage('Context Engine Uploader: open a folder or set contextEngineUploader.targetPath.');
    updateStatusBarTooltip();
    return undefined;
  }

  function saveTargetPath(config, targetPath) {
    const hasWorkspace = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length;
    const target = hasWorkspace ? vscode.ConfigurationTarget.Workspace : vscode.ConfigurationTarget.Global;
    config.update('targetPath', targetPath, target).catch(error => {
      log(`Target path save failed: ${error instanceof Error ? error.message : String(error)}`);
    });
  }

  return {
    getWorkspaceFolderPath,
    looksLikeRepoRoot,
    detectDefaultTargetPath,
    resolveTargetPathFromConfig,
    getTargetPath,
    saveTargetPath,
  };
}

module.exports = {
  createWorkspacePathUtils,
};
