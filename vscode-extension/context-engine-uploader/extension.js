const vscode = require('vscode');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
let outputChannel;
let watchProcess;
let forceProcess;
let extensionRoot;
let statusBarItem;
let statusMode = 'idle';
function activate(context) {
  outputChannel = vscode.window.createOutputChannel('Context Engine Upload');
  context.subscriptions.push(outputChannel);
  extensionRoot = context.extensionPath;
  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = 'contextEngineUploader.indexCodebase';
  context.subscriptions.push(statusBarItem);
  statusBarItem.show();
  setStatusBarState('idle');
  updateStatusBarTooltip();
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
    runSequence('force').catch(error => log(`Index failed: ${error instanceof Error ? error.message : String(error)}`));
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
  context.subscriptions.push(startDisposable, stopDisposable, restartDisposable, indexDisposable, configDisposable, workspaceDisposable);
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
  await stopProcesses();
  const needsForce = mode === 'force' || needsForceSync(options.targetPath);
  if (needsForce) {
    setStatusBarState('indexing');
    const code = await runOnce(options);
    if (code === 0) {
      startWatch(options);
    } else {
      setStatusBarState('idle');
    }
    return;
  }
  startWatch(options);
}
function resolveOptions() {
  const config = vscode.workspace.getConfiguration('contextEngineUploader');
  const pythonPath = (config.get('pythonPath') || 'python3').trim();
  const endpoint = (config.get('endpoint') || '').trim();
  const targetPath = getTargetPath(config);
  const interval = config.get('intervalSeconds') || 5;
  const extraForceArgs = config.get('extraForceArgs') || [];
  const extraWatchArgs = config.get('extraWatchArgs') || [];
  let workingDirectory = (config.get('scriptWorkingDirectory') || '').trim();
  if (!workingDirectory) {
    workingDirectory = extensionRoot;
  }
  if (!workingDirectory) {
    vscode.window.showErrorMessage('Context Engine Uploader: extension path unavailable.');
    return undefined;
  }
  const scriptPath = path.join(workingDirectory, 'standalone_upload_client.py');
  if (!fs.existsSync(scriptPath)) {
    vscode.window.showErrorMessage('Context Engine Uploader: standalone_upload_client.py not found. Reinstall extension or set scriptWorkingDirectory.');
    return undefined;
  }
  if (!endpoint) {
    vscode.window.showErrorMessage('Context Engine Uploader: set contextEngineUploader.endpoint.');
    return undefined;
  }
  if (!targetPath) {
    return undefined;
  }
  return {
    pythonPath,
    workingDirectory,
    scriptPath,
    targetPath,
    endpoint,
    interval,
    extraForceArgs,
    extraWatchArgs
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
  targetPath = folderPath;
  saveTargetPath(config, targetPath);
  updateStatusBarTooltip(targetPath);
  return targetPath;
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
  saveTargetPath(config, folderPath);
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
function setStatusBarState(mode) {
  if (!statusBarItem) {
    return;
  }
  statusMode = mode;
  if (mode === 'indexing') {
    statusBarItem.text = '$(sync~spin) Indexing...';
    statusBarItem.color = undefined;
  } else if (mode === 'watch') {
    statusBarItem.text = '$(sync) Watching';
    statusBarItem.color = new vscode.ThemeColor('charts.purple');
  } else {
    statusBarItem.text = '$(sync) Index Codebase';
    statusBarItem.color = undefined;
  }
}
function runOnce(options) {
  return new Promise(resolve => {
    const args = buildArgs(options, 'force');
    const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory });
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
  const args = buildArgs(options, 'watch');
  const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory });
  watchProcess = child;
  attachOutput(child, 'watch');
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
    const finalize = () => {
      if (finished) {
        return;
      }
      finished = true;
      if (proc === forceProcess) {
        forceProcess = undefined;
      }
      if (proc === watchProcess) {
        watchProcess = undefined;
      }
      log(`${label} process stopped.`);
      resolve();
    };
    proc.once('exit', finalize);
    proc.once('close', finalize);
    try {
      proc.kill();
    } catch (error) {
      finalize();
      return;
    }
    setTimeout(finalize, 2000);
  });
}
function log(message) {
  if (!outputChannel) {
    return;
  }
  const timestamp = new Date().toISOString();
  outputChannel.appendLine(`[${timestamp}] ${message}`);
}
function deactivate() {
  return stopProcesses();
}
module.exports = {
  activate,
  deactivate
};
