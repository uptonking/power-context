const vscode = require('vscode');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
let outputChannel;
let watchProcess;
let forceProcess;
let extensionRoot;
let statusBarItem;
let logsTerminal;
let logTailActive = false;
let statusMode = 'idle';
let workspaceWatcher;
let watchedTargetPath;
let indexedWatchDisposables = [];
let globalStoragePath;
let pythonOverridePath;
const REQUIRED_PYTHON_MODULES = ['requests', 'urllib3', 'charset_normalizer'];
const DEFAULT_CONTAINER_ROOT = '/work';
function activate(context) {
  outputChannel = vscode.window.createOutputChannel('Context Engine Upload');
  context.subscriptions.push(outputChannel);
  extensionRoot = context.extensionPath;
  globalStoragePath = context.globalStorageUri && context.globalStorageUri.fsPath ? context.globalStorageUri.fsPath : undefined;
  try {
    const venvPy = resolvePrivateVenvPython();
    if (venvPy) {
      pythonOverridePath = venvPy;
      log(`Detected existing private venv interpreter: ${venvPy}`);
    }
  } catch (_) {}
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = 'contextEngineUploader.indexCodebase';
  context.subscriptions.push(statusBarItem);
  statusBarItem.show();
  setStatusBarState('idle');
  updateStatusBarTooltip();

  // Ensure an output channel is visible early for user feedback
  if (outputChannel) { outputChannel.show(true); }
  const startDisposable = vscode.commands.registerCommand('contextEngineUploader.start', () => {
    runSequence('auto').catch(error => log(`Start failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const stopDisposable = vscode.commands.registerCommand('contextEngineUploader.stop', () => {
    stopProcesses().catch(error => log(`Stop failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const restartDisposable = vscode.commands.registerCommand('contextEngineUploader.restart', () => {
    stopProcesses().then(() => runSequence('auto')).catch(error => log(`Restart failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const indexDisposable = vscode.commands.registerCommand('contextEngineUploader.indexCodebase', () => {
    vscode.window.showInformationMessage('Context Engine indexing started.');
    if (outputChannel) { outputChannel.show(true); }
    try {
      const cfg = vscode.workspace.getConfiguration('contextEngineUploader');
      if (cfg.get('autoTailUploadLogs', true)) {
        openUploadServiceLogsTerminal();
      }
    } catch (e) {
      log(`Auto-tail logs failed: ${e && e.message ? e.message : String(e)}`);
    }
    runSequence('force').catch(error => log(`Index failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const showLogsDisposable = vscode.commands.registerCommand('contextEngineUploader.showUploadServiceLogs', () => {
    try { openUploadServiceLogsTerminal(); } catch (e) { log(`Show logs failed: ${e && e.message ? e.message : String(e)}`); }
  });
  const configDisposable = vscode.workspace.onDidChangeConfiguration(event => {
    if (event.affectsConfiguration('contextEngineUploader') && watchProcess) {
      runSequence('auto').catch(error => log(`Auto-restart failed: ${error instanceof Error ? error.message : String(error)}`));
    }
    if (event.affectsConfiguration('contextEngineUploader.targetPath')) {
      updateStatusBarTooltip();
    }
  });
  const workspaceDisposable = vscode.workspace.onDidChangeWorkspaceFolders(() => {
    ensureTargetPathConfigured();
  });
  const terminalCloseDisposable = vscode.window.onDidCloseTerminal(term => {
    if (term === logsTerminal) {
      logsTerminal = undefined;
      logTailActive = false;
    }
  });
  context.subscriptions.push(startDisposable, stopDisposable, restartDisposable, indexDisposable, showLogsDisposable, configDisposable, workspaceDisposable, terminalCloseDisposable);
  const config = vscode.workspace.getConfiguration('contextEngineUploader');
  ensureTargetPathConfigured();
  if (config.get('runOnStartup')) {
    runSequence('auto').catch(error => log(`Startup run failed: ${error instanceof Error ? error.message : String(error)}`));
  }
}
async function runSequence(mode = 'auto') {
  const options = resolveOptions();
  if (!options) {
    return;
  }
  const depsSatisfied = await ensurePythonDependencies(options.pythonPath);
  if (!depsSatisfied) {
    setStatusBarState('idle');
    return;
  }
  // Re-resolve options in case ensurePythonDependencies switched to a private venv interpreter
  const reoptions = resolveOptions();
  if (reoptions) {
    Object.assign(options, reoptions);
  }
  await stopProcesses();
  const needsForce = mode === 'force' || needsForceSync(options.targetPath);
  if (needsForce) {
    setStatusBarState('indexing');
    if (outputChannel) { outputChannel.show(true); }
    const code = await runOnce(options);
    if (code === 0) {
      setStatusBarState('indexed');
      ensureIndexedWatcher(options.targetPath);
      if (options.startWatchAfterForce) {
        startWatch(options);
      }
    } else {
      setStatusBarState('idle');
    }
    return;
  }
  startWatch(options);
}
function resolveOptions() {
  const config = vscode.workspace.getConfiguration('contextEngineUploader');
  let pythonPath = (config.get('pythonPath') || 'python3').trim();
  if (pythonOverridePath && fs.existsSync(pythonOverridePath)) {
    pythonPath = pythonOverridePath;
  }
  const endpoint = (config.get('endpoint') || '').trim();
  const targetPath = getTargetPath(config);
  const interval = config.get('intervalSeconds') || 5;
  const extraForceArgs = config.get('extraForceArgs') || [];
  const extraWatchArgs = config.get('extraWatchArgs') || [];
  const hostRootOverride = (config.get('hostRoot') || '').trim();
  const containerRoot = (config.get('containerRoot') || DEFAULT_CONTAINER_ROOT).trim() || DEFAULT_CONTAINER_ROOT;
  const startWatchAfterForce = config.get('startWatchAfterForce', true);
  const configuredScriptDir = (config.get('scriptWorkingDirectory') || '').trim();
  const candidates = [];
  if (configuredScriptDir) {
    candidates.push(configuredScriptDir);
  }
  // Prefer packaged script; also try workspace ./scripts fallback for dev
  candidates.push(extensionRoot);
  const wsRoot = getWorkspaceFolderPath();
  if (wsRoot) {
    candidates.push(path.join(wsRoot, 'scripts'));
  }
  candidates.push(path.join(extensionRoot, '..', 'out'));
  let workingDirectory;
  let scriptPath;
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const resolved = path.resolve(candidate);
    const testPath = path.join(resolved, 'standalone_upload_client.py');
    if (fs.existsSync(testPath)) {
      workingDirectory = resolved;
      scriptPath = testPath;
      break;
    }
  }
  if (!workingDirectory || !scriptPath) {
    vscode.window.showErrorMessage('Context Engine Uploader: extension path unavailable.');
    return undefined;
  }
  const scriptSource = workingDirectory === extensionRoot ? 'packaged' : (workingDirectory.includes('\\out') || workingDirectory.endsWith('/out') ? 'staged out' : 'custom');
  if (!endpoint) {
    vscode.window.showErrorMessage('Context Engine Uploader: set contextEngineUploader.endpoint.');
    return undefined;
  }
  if (!targetPath) {
    return undefined;
  }
  const resolvedTarget = path.resolve(targetPath);
  let derivedHostRoot = path.dirname(resolvedTarget);
  if (!derivedHostRoot || derivedHostRoot === resolvedTarget) {
    derivedHostRoot = resolvedTarget;
  }
  const hostRoot = hostRootOverride || derivedHostRoot;
  log(`Using ${scriptSource} standalone_upload_client.py at ${scriptPath}`);
  log(`Uploader path mapping hostRoot=${hostRoot || 'n/a'} -> containerRoot=${containerRoot}`);
  return {
    pythonPath,
    workingDirectory,
    scriptPath,
    targetPath,
    endpoint,
    interval,
    extraForceArgs,
    extraWatchArgs,
    hostRoot,
    containerRoot,
    startWatchAfterForce
  };
}
function getTargetPath(config) {
  let targetPath = (config.get('targetPath') || '').trim();
  if (targetPath) {
    updateStatusBarTooltip(targetPath);
    return targetPath;
  }
  const folderPath = getWorkspaceFolderPath();
  if (!folderPath) {
    vscode.window.showErrorMessage('Context Engine Uploader: open a folder or set contextEngineUploader.targetPath.');
    updateStatusBarTooltip();
    return undefined;
  }
  updateStatusBarTooltip(folderPath);
  return folderPath;
}
function saveTargetPath(config, targetPath) {
  const hasWorkspace = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length;
  const target = hasWorkspace ? vscode.ConfigurationTarget.Workspace : vscode.ConfigurationTarget.Global;
  config.update('targetPath', targetPath, target).catch(error => {
    log(`Target path save failed: ${error instanceof Error ? error.message : String(error)}`);
  });
}
function getWorkspaceFolderPath() {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || !folders.length) {
    return undefined;
  }
  return folders[0].uri.fsPath;
}
function ensureTargetPathConfigured() {
  const config = vscode.workspace.getConfiguration('contextEngineUploader');
  const current = (config.get('targetPath') || '').trim();
  if (current) {
    updateStatusBarTooltip(current);
    return;
  }
  const folderPath = getWorkspaceFolderPath();
  if (!folderPath) {
    updateStatusBarTooltip();
    return;
  }
  updateStatusBarTooltip(folderPath);
}
function updateStatusBarTooltip(targetPath) {
  if (!statusBarItem) {
    return;
  }
  if (targetPath) {
    statusBarItem.tooltip = `Index Codebase (${targetPath})`;
  } else {
    statusBarItem.tooltip = 'Index Codebase';
  }
}
function needsForceSync(targetPath) {
  try {
    const cachePath = path.join(targetPath, '.context-engine', 'file_cache.json');
    if (!fs.existsSync(cachePath)) {
      return true;
    }
    const stats = fs.statSync(cachePath);
    return stats.size === 0;
  } catch (error) {
    log(`Force detection failed: ${error instanceof Error ? error.message : String(error)}`);
    return true;
  }
}
async function ensurePythonDependencies(pythonPath) {
  // Probe current interpreter; if modules missing, offer to create a private venv and install deps
  const ok = await checkPythonDeps(pythonPath);
  if (ok) return true;
  const choice = await vscode.window.showErrorMessage(
    'Context Engine Uploader: missing Python modules. Create isolated environment and auto-install?',
    'Auto-install to private venv',
    'Cancel'
  );
  if (choice !== 'Auto-install to private venv') {
    return false;
  }
  const created = await ensurePrivateVenv();
  if (!created) return false;
  const venvPython = resolvePrivateVenvPython();
  if (!venvPython) {
    vscode.window.showErrorMessage('Context Engine Uploader: failed to locate private venv python.');
    return false;
  }
  const installed = await installDepsInto(venvPython);
  if (!installed) return false;
  pythonOverridePath = venvPython;
  log(`Using private venv interpreter: ${pythonOverridePath}`);
  return await checkPythonDeps(pythonOverridePath);
}

async function checkPythonDeps(pythonPath) {
  const missing = [];
  let pythonError;
  for (const moduleName of REQUIRED_PYTHON_MODULES) {
    const check = spawnSync(pythonPath, ['-c', `import ${moduleName}`], { encoding: 'utf8' });
    if (check.error) {
      pythonError = check.error;
      break;
    }
    if (check.status !== 0) {
      missing.push(moduleName);
    }
  }
  if (pythonError) {
    const message = `Context Engine Uploader: failed to run ${pythonPath}. Update contextEngineUploader.pythonPath.`;
    vscode.window.showErrorMessage(message);
    log(`Dependency check failed: ${pythonError.message || pythonError}`);
    return false;
  }
  if (missing.length) {
    log(`Missing Python modules for ${pythonPath}: ${missing.join(', ')}`);
    return false;
  }
  return true;
}

function venvRootDir() {
  // Prefer workspace storage; fallback to extension storage
  try {
    const ws = getWorkspaceFolderPath();
    const base = ws && fs.existsSync(ws) ? path.join(ws, '.vscode', '.context-engine-uploader')
      : (globalStoragePath || path.join(extensionRoot, '.storage'));
    if (!fs.existsSync(base)) fs.mkdirSync(base, { recursive: true });
    return base;
  } catch (e) {
    return extensionRoot;
  }
}

function privateVenvPath() {
  return path.join(venvRootDir(), 'py-venv');
}

function resolvePrivateVenvPython() {
  const venvPath = privateVenvPath();
  const bin = process.platform === 'win32' ? path.join(venvPath, 'Scripts', 'python.exe') : path.join(venvPath, 'bin', 'python');
  return fs.existsSync(bin) ? bin : undefined;
}

async function ensurePrivateVenv() {
  try {
    const python = resolvePrivateVenvPython();
    if (python) {
      log('Private venv already exists.');
      return true;
    }
    const venvPath = privateVenvPath();
    const basePy = await detectSystemPython();
    if (!basePy) {
      vscode.window.showErrorMessage('Context Engine Uploader: no Python interpreter found to bootstrap venv.');
      return false;
    }
    log(`Creating private venv at ${venvPath} using ${basePy}`);
    const res = spawnSync(basePy, ['-m', 'venv', venvPath], { encoding: 'utf8' });
    if (res.status !== 0) {
      log(`venv creation failed: ${res.stderr || res.stdout}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to create private venv.');
      return false;
    }
    return true;
  } catch (e) {
    log(`ensurePrivateVenv error: ${e && e.message ? e.message : String(e)}`);
    return false;
  }
}

async function installDepsInto(pythonBin) {
  try {
    log(`Installing Python deps into private venv via ${pythonBin}`);
    const args = ['-m', 'pip', 'install', ...REQUIRED_PYTHON_MODULES];
    const res = spawnSync(pythonBin, args, { encoding: 'utf8' });
    if (res.status !== 0) {
      log(`pip install failed: ${res.stderr || res.stdout}`);
      vscode.window.showErrorMessage('Context Engine Uploader: pip install failed. See Output for details.');
      return false;
    }
    return true;
  } catch (e) {
    log(`installDepsInto error: ${e && e.message ? e.message : String(e)}`);
    return false;
  }
}

async function detectSystemPython() {
  // Try configured pythonPath, then common names
  const candidates = [];
  try {
    const cfg = vscode.workspace.getConfiguration('contextEngineUploader');
    const configured = (cfg.get('pythonPath') || '').trim();
    if (configured) candidates.push(configured);
  } catch {}
  if (process.platform === 'win32') {
    candidates.push('py', 'python3', 'python');
  } else {
    candidates.push('python3', 'python');
    // Add common Homebrew path on Apple Silicon
    candidates.push('/opt/homebrew/bin/python3');
  }
  for (const cmd of candidates) {
    const probe = spawnSync(cmd, ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' });
    if (!probe.error && probe.status === 0) {
      const p = (probe.stdout || '').trim();
      if (p) return p;
    }
  }
  return undefined;
}
function setStatusBarState(mode) {
  if (!statusBarItem) {
    return;
  }
  statusMode = mode;
  if (mode === 'indexing') {
    statusBarItem.text = '$(sync~spin) Indexing...';
    statusBarItem.color = undefined;
  } else if (mode === 'indexed') {
    statusBarItem.text = '$(check) Indexed';
    statusBarItem.color = new vscode.ThemeColor('charts.green');
  } else if (mode === 'watch') {
    statusBarItem.text = '$(sync) Watching (Click Force Index)';
    statusBarItem.color = new vscode.ThemeColor('charts.purple');
  } else {
    statusBarItem.text = '$(sync) Index Codebase';
    statusBarItem.color = undefined;
  }
}
function runOnce(options) {
  return new Promise(resolve => {
    const args = buildArgs(options, 'force');
    const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory, env: buildChildEnv(options) });
    forceProcess = child;
    attachOutput(child, 'force');
    let finished = false;
    const finish = code => {
      if (finished) {
        return;
      }
      finished = true;
      log(`Force sync exited with code ${code}`);
      if (forceProcess === child) {
        forceProcess = undefined;
      }
      resolve(typeof code === 'number' ? code : 1);
    };
    child.on('close', finish);
    child.on('error', error => {
      log(`Force sync failed: ${error.message}`);
      finish(1);
    });
  });
}
function startWatch(options) {
  disposeIndexedWatcher();
  const args = buildArgs(options, 'watch');
  const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory, env: buildChildEnv(options) });
  watchProcess = child;
  attachOutput(child, 'watch');
  if (outputChannel) { outputChannel.show(true); }
  setStatusBarState('watch');
  child.on('close', code => {
    log(`Watch exited with code ${code}`);
    if (watchProcess === child) {
      watchProcess = undefined;
      if (statusMode !== 'indexing') {
        setStatusBarState('idle');
      }
    }
  });
  child.on('error', error => {
    log(`Watch failed: ${error.message}`);
    if (watchProcess === child) {
      watchProcess = undefined;
      if (statusMode !== 'indexing') {
        setStatusBarState('idle');
      }
    }
  });
  vscode.window.showInformationMessage('Context Engine remote watch started.');
}
function buildArgs(options, mode) {
  const args = ['-u', options.scriptPath, '--path', options.targetPath, '--endpoint', options.endpoint];
  if (mode === 'force') {
    args.push('--force');
    if (options.extraForceArgs && options.extraForceArgs.length) {
      args.push(...options.extraForceArgs);
    }
  } else {
    args.push('--watch', '--interval', String(options.interval));
    if (options.extraWatchArgs && options.extraWatchArgs.length) {
      args.push(...options.extraWatchArgs);
    }
  }
  return args;
}
function attachOutput(child, label) {
  if (!outputChannel) {
    return;
  }
  if (child.stdout) {
    child.stdout.on('data', data => {
      outputChannel.append(`[${label}] ${data}`);
    });
  }
  if (child.stderr) {
    child.stderr.on('data', data => {
      outputChannel.append(`[${label} err] ${data}`);
    });
  }
}
function openUploadServiceLogsTerminal() {
  try {
    const cfg = vscode.workspace.getConfiguration('contextEngineUploader');
    const wsPath = getWorkspaceFolderPath() || (cfg.get('targetPath') || '');
    const cwd = (wsPath && typeof wsPath === 'string' && fs.existsSync(wsPath)) ? wsPath : undefined;
    if (logsTerminal && logsTerminal.exitStatus) {
      logsTerminal = undefined;
      logTailActive = false;
    }
    if (!logsTerminal) {
      logsTerminal = vscode.window.createTerminal({ name: 'Context Engine Upload Logs', cwd: cwd ? vscode.Uri.file(cwd) : undefined });
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

function ensureIndexedWatcher(targetPath) {
  try {
    disposeIndexedWatcher();
    watchedTargetPath = targetPath;
    let pattern;
    if (targetPath && fs.existsSync(targetPath)) {
      pattern = new vscode.RelativePattern(targetPath, '**/*');
    } else {
      const folder = getWorkspaceFolderPath();
      if (folder && fs.existsSync(folder)) {
        pattern = new vscode.RelativePattern(folder, '**/*');
      } else {
        pattern = '**/*';
      }
    }
    workspaceWatcher = vscode.workspace.createFileSystemWatcher(pattern, false, false, false);
    const flipToIdle = () => {
      if (statusMode === 'indexed') {
        setStatusBarState('idle');
      }
    };
    indexedWatchDisposables.push(workspaceWatcher);
    indexedWatchDisposables.push(workspaceWatcher.onDidCreate(flipToIdle));
    indexedWatchDisposables.push(workspaceWatcher.onDidChange(flipToIdle));
    indexedWatchDisposables.push(workspaceWatcher.onDidDelete(flipToIdle));
    indexedWatchDisposables.push(vscode.workspace.onDidChangeTextDocument(() => flipToIdle()));
    log('Indexed watcher armed; any file change will return status bar to "Index Codebase".');
  } catch (e) {
    log(`Failed to arm indexed watcher: ${e && e.message ? e.message : String(e)}`);
  }
}

function disposeIndexedWatcher() {
  try {
    for (const d of indexedWatchDisposables) {
      try { if (d && typeof d.dispose === 'function') d.dispose(); } catch (_) {}
    }
    indexedWatchDisposables = [];
    if (workspaceWatcher && typeof workspaceWatcher.dispose === 'function') {
      workspaceWatcher.dispose();
    }
    workspaceWatcher = undefined;
    watchedTargetPath = undefined;
  } catch (e) {
    // ignore
  }
}

async function stopProcesses() {
  await Promise.all([terminateProcess(forceProcess, 'force'), terminateProcess(watchProcess, 'watch')]);
  if (!forceProcess && !watchProcess && statusMode !== 'indexing') {
    setStatusBarState('idle');
  }
}
function terminateProcess(proc, label) {
  if (!proc) {
    return Promise.resolve();
  }
  return new Promise(resolve => {
    let finished = false;
    let termTimer;
    let killTimer;
    const cleanup = () => {
      if (termTimer) clearTimeout(termTimer);
      if (killTimer) clearTimeout(killTimer);
    };
    const finalize = (reason) => {
      if (finished) return;
      finished = true;
      cleanup();
      if (proc === forceProcess) {
        forceProcess = undefined;
      }
      if (proc === watchProcess) {
        watchProcess = undefined;
      }
      log(`${label} process stopped${reason ? ` (${reason})` : ''}.`);
      resolve();
    };

    // Resolve only after the child actually exits (or after forced kill path)
    const onExit = (code, signal) => {
      finalize(`exit code=${code} signal=${signal || ''}`.trim());
    };
    proc.once('exit', onExit);
    proc.once('close', onExit);

    try {
      proc.kill(); // default SIGTERM
    } catch (error) {
      finalize('kill() threw');
      return;
    }

    const waitSigtermMs = 4000;
    const waitSigkillMs = 2000;

    // If process doesn't exit after SIGTERM, escalate to SIGKILL and then force-resolve
    termTimer = setTimeout(() => {
      try {
        if (proc && !proc.killed) {
          proc.kill('SIGKILL');
          log(`${label} process did not exit after ${waitSigtermMs}ms; sent SIGKILL.`);
        }
      } catch (_) {
        // ignore
      }
      killTimer = setTimeout(() => {
        finalize(`forced after ${waitSigtermMs + waitSigkillMs}ms`);
      }, waitSigkillMs);
    }, waitSigtermMs);
  });
}
function log(message) {
  if (!outputChannel) {
    return;
  }
  const timestamp = new Date().toISOString();
  outputChannel.appendLine(`[${timestamp}] ${message}`);
}
function buildChildEnv(options) {
  const env = {
    ...process.env,
    WORKSPACE_PATH: options.targetPath,
    WATCH_ROOT: options.targetPath
  };
  if (options.hostRoot) {
    env.HOST_ROOT = options.hostRoot;
  }
  if (options.containerRoot) {
    env.CONTAINER_ROOT = options.containerRoot;
  }
   try {
     const libsPath = path.join(options.workingDirectory, 'python_libs');
     if (fs.existsSync(libsPath)) {
       const existing = process.env.PYTHONPATH || '';
       env.PYTHONPATH = existing ? `${libsPath}${path.delimiter}${existing}` : libsPath;
       log(`Detected bundled python_libs at ${libsPath}; setting PYTHONPATH for child process.`);
     }
   } catch (error) {
     log(`Failed to configure PYTHONPATH for bundled deps: ${error instanceof Error ? error.message : String(error)}`);
  }
  return env;
}
function deactivate() {
  disposeIndexedWatcher();
  return stopProcesses();
}
module.exports = {
  activate,
  deactivate
};
