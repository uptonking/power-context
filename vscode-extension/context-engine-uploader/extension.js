const vscode = require('vscode');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const { ensureAuthIfRequired, runAuthLoginFlow } = require('./auth_utils');
const profiles = require('./profiles');
const sidebar = require('./sidebar');
const { createBridgeManager } = require('./mcp_bridge');
const { createMcpConfigManager } = require('./mcp_config');
const { createCtxConfigManager } = require('./ctx_config');
const { createWorkspacePathUtils } = require('./workspace_paths');
const { createLogsTerminalManager } = require('./logs_terminal');
const { createPromptPlusManager } = require('./prompt_plus');
const { createOnboardingManager } = require('./onboarding');
let outputChannel;
let watchProcess;
let forceProcess;
let extensionRoot;
let statusBarItem;
let promptStatusBarItem;
let logsTerminalManager;
let statusMode = 'idle';
let workspaceWatcher;
let watchedTargetPath;
let indexedWatchDisposables = [];
let globalStoragePath;
let pythonOverridePath;
let bridgeManager;
let mcpConfigManager;
let ctxConfigManager;
let workspacePathUtils;
let promptPlusManager;
let onboardingManager;
let pendingProfileRestartTimer;
const REQUIRED_PYTHON_MODULES = ['requests', 'urllib3', 'charset_normalizer'];
const DEFAULT_CONTAINER_ROOT = '/work';
const ONBOARDING_PROMPT_KEY = 'contextEngineUploader.onboardingPrompted';
// const CLAUDE_HOOK_COMMAND = '/home/coder/project/Context-Engine/ctx-hook-simple.sh';

function getEffectiveConfig() {
  try {
    return profiles.getUploaderConfig();
  } catch (_) {
    return vscode.workspace.getConfiguration('contextEngineUploader');
  }
}

function getResolvedTargetPathForSidebar() {
  try {
    const config = getEffectiveConfig();
    const result = resolveTargetPathFromConfig(config);
    let target = result && result.path ? result.path : undefined;
    let source = (result && result.inferred) ? 'inferred' : 'settings';
    if (!target && watchedTargetPath) {
      target = watchedTargetPath;
      source = 'runtime';
    }
    return { path: target, source };
  } catch (_) {
    return { path: watchedTargetPath, source: watchedTargetPath ? 'runtime' : undefined };
  }
}

function scheduleRestartAfterProfileChange() {
  try {
    if (pendingProfileRestartTimer) {
      clearTimeout(pendingProfileRestartTimer);
      pendingProfileRestartTimer = undefined;
    }
  } catch (_) {
  }
  try {
    pendingProfileRestartTimer = setTimeout(() => {
      pendingProfileRestartTimer = undefined;
      if (!watchProcess) {
        return;
      }
      runSequence('auto').catch(error => log(`Auto-restart after profile change failed: ${error instanceof Error ? error.message : String(error)}`));
    }, 250);
  } catch (_) {
  }
}

function activate(context) {
  outputChannel = vscode.window.createOutputChannel('Context Engine Upload');
  context.subscriptions.push(outputChannel);
  extensionRoot = context.extensionPath;
  globalStoragePath = context.globalStorageUri && context.globalStorageUri.fsPath ? context.globalStorageUri.fsPath : undefined;
  try {
    profiles.init({
      vscode,
      context,
      log,
      onProfileChanged: () => {
        try { ensureTargetPathConfigured(); } catch (_) {}
        try {
          if (watchProcess) {
            scheduleRestartAfterProfileChange();
          }
        } catch (_) {}
      },
    });
  } catch (error) {
    log(`Profiles init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    workspacePathUtils = createWorkspacePathUtils({
      vscode,
      path,
      fs,
      log,
      updateStatusBarTooltip,
    });
  } catch (error) {
    workspacePathUtils = undefined;
    log(`Workspace path utils init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    logsTerminalManager = createLogsTerminalManager({
      vscode,
      fs,
      log,
      getEffectiveConfig,
      getWorkspaceFolderPath,
    });
  } catch (error) {
    logsTerminalManager = undefined;
    log(`Logs terminal manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    promptPlusManager = createPromptPlusManager({
      vscode,
      spawn,
      path,
      fs,
      log,
      extensionRoot,
      getEffectiveConfig,
      getTargetPath,
      getWorkspaceFolderPath,
      detectDefaultTargetPath,
      resolveBridgeHttpUrl,
      getPythonOverridePath: () => pythonOverridePath,
      appendOutput: (text) => {
        if (outputChannel) {
          outputChannel.append(text);
        }
      },
    });
  } catch (error) {
    promptPlusManager = undefined;
    log(`Prompt+ manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    onboardingManager = createOnboardingManager({
      vscode,
      context,
      log,
      appendOutput: text => {
        if (outputChannel) {
          outputChannel.append(text);
        }
      },
      showOutput: () => {
        if (outputChannel) {
          outputChannel.show(true);
        }
      },
    });
  } catch (error) {
    onboardingManager = undefined;
    log(`Onboarding manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    bridgeManager = createBridgeManager({
      vscode,
      spawn,
      log,
      getEffectiveConfig,
      resolveBridgeWorkspacePath,
      normalizeBridgeUrl,
      normalizeWorkspaceForBridge,
      resolveBridgeCliInvocation,
      attachOutput,
      terminateProcess,
      scheduleMcpConfigRefreshAfterBridge,
    });
  } catch (error) {
    bridgeManager = undefined;
    log(`Bridge manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    ctxConfigManager = createCtxConfigManager({
      vscode,
      spawnSync,
      log,
      extensionRoot,
      getEffectiveConfig,
      resolveOptions,
      ensurePythonDependencies,
      buildChildEnv,
      resolveBridgeHttpUrl,
    });
  } catch (error) {
    ctxConfigManager = undefined;
    log(`CTX config manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    mcpConfigManager = createMcpConfigManager({
      vscode,
      log,
      extensionRoot,
      getEffectiveConfig,
      getWorkspaceFolderPath,
      resolveBridgeWorkspacePath,
      normalizeBridgeUrl,
      normalizeWorkspaceForBridge,
      resolveBridgeCliInvocation,
      resolveBridgeHttpUrl,
      requiresHttpBridge,
      ensureHttpBridgeReadyForConfigs,
      getBridgeIsRunning: () => (bridgeManager && typeof bridgeManager.isRunning === 'function' ? bridgeManager.isRunning() : false),
      writeCtxConfig,
    });
  } catch (error) {
    mcpConfigManager = undefined;
    log(`MCP config manager init failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    // Ensure manager resources are cleaned up when the extension deactivates.
    const managerDisposable = {
      dispose: () => {
        try { if (mcpConfigManager && typeof mcpConfigManager.dispose === 'function') mcpConfigManager.dispose(); } catch (_) {}
        try { if (ctxConfigManager && typeof ctxConfigManager.dispose === 'function') ctxConfigManager.dispose(); } catch (_) {}
        try { if (bridgeManager && typeof bridgeManager.dispose === 'function') bridgeManager.dispose(); } catch (_) {}
        try { if (logsTerminalManager && typeof logsTerminalManager.dispose === 'function') logsTerminalManager.dispose(); } catch (_) {}
        try { if (promptPlusManager && typeof promptPlusManager.dispose === 'function') promptPlusManager.dispose(); } catch (_) {}
        try { if (onboardingManager && typeof onboardingManager.dispose === 'function') onboardingManager.dispose(); } catch (_) {}
      }
    };
    context.subscriptions.push(managerDisposable);
  } catch (_) {
    // ignore
  }
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
  promptStatusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 90);
  promptStatusBarItem.command = 'contextEngineUploader.promptEnhance';
  promptStatusBarItem.text = '$(sparkle) Prompt+';
  promptStatusBarItem.tooltip = 'Enhance selection with Unicorn Mode via ctx.py';
  context.subscriptions.push(promptStatusBarItem);
  promptStatusBarItem.show();


  try {
    const disposables = profiles.registerCommands({
      resolveTargetPathFromConfig,
      getWorkspaceFolderPath,
      detectDefaultTargetPath,
      normalizeWorkspaceForBridge,
      runSequence,
      writeMcpConfig,
      writeCtxConfig,
      fetch: (typeof fetch === 'function' ? fetch : undefined),
    });
    if (Array.isArray(disposables)) {
      context.subscriptions.push(...disposables);
    }
  } catch (error) {
    log(`Profiles command registration failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    sidebar.register(context, {
      profiles,
      getEffectiveConfig,
      getResolvedTargetPath: getResolvedTargetPathForSidebar,
      getState: () => ({
        statusMode,
        httpBridgeProcess: bridgeManager ? bridgeManager.getState().process : undefined,
        httpBridgePort: bridgeManager ? bridgeManager.getState().port : undefined,
      }),
      onboarding: onboardingManager,
    });
  } catch (error) {
    log(`Sidebar registration failed: ${error instanceof Error ? error.message : String(error)}`);
  }

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
    runSequence('force').catch(error => log(`Index failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const uploadGitHistoryDisposable = vscode.commands.registerCommand('contextEngineUploader.uploadGitHistory', () => {
    vscode.window.showInformationMessage('Context Engine git history upload (force sync bundle) started.');
    if (outputChannel) { outputChannel.show(true); }
    runSequence('force').catch(error => log(`Git history upload failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const ctxConfigDisposable = vscode.commands.registerCommand('contextEngineUploader.writeCtxConfig', () => {
    writeCtxConfig().catch(error => log(`CTX config write failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const mcpConfigDisposable = vscode.commands.registerCommand('contextEngineUploader.writeMcpConfig', () => {
    writeMcpConfig().catch(error => log(`MCP config write failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const mcpConfigSelectDisposable = vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigSelect', async () => {
    try {
      const cfg = getEffectiveConfig();
      const claudeEnabled = !!cfg.get('mcpClaudeEnabled', true);
      const windsurfEnabled = !!cfg.get('mcpWindsurfEnabled', false);
      const augmentEnabled = !!cfg.get('mcpAugmentEnabled', false);

      const items = [
        {
          label: 'All enabled targets',
          description: 'Writes MCP config for all enabled clients',
          id: 'all',
        },
        {
          label: 'Claude Code (.mcp.json)',
          description: claudeEnabled ? 'Enabled' : 'Disabled in settings',
          id: 'claude',
        },
        {
          label: 'Windsurf (mcp_config.json)',
          description: windsurfEnabled ? 'Enabled' : 'Disabled in settings',
          id: 'windsurf',
        },
        {
          label: 'Augment Code (~/.augment/settings.json)',
          description: augmentEnabled ? 'Enabled' : 'Disabled in settings',
          id: 'augment',
        },
      ];

      const picked = await vscode.window.showQuickPick(items, { placeHolder: 'Select which MCP config to write' });
      if (!picked) {
        return;
      }

      if (picked.id === 'all') {
        await writeMcpConfig();
      } else if (picked.id === 'claude') {
        await writeMcpConfig({ targets: ['claude'] });
      } else if (picked.id === 'windsurf') {
        await writeMcpConfig({ targets: ['windsurf'] });
      } else if (picked.id === 'augment') {
        await writeMcpConfig({ targets: ['augment'] });
      }
    } catch (error) {
      log(`MCP config select failed: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to select MCP config target. See output for details.');
    }
  });
  const mcpConfigClaudeDisposable = vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigClaude', () => {
    writeMcpConfig({ targets: ['claude'] }).catch(error => log(`Claude MCP config write failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const mcpConfigWindsurfDisposable = vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigWindsurf', () => {
    writeMcpConfig({ targets: ['windsurf'] }).catch(error => log(`Windsurf MCP config write failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const mcpConfigAugmentDisposable = vscode.commands.registerCommand('contextEngineUploader.writeMcpConfigAugment', () => {
    writeMcpConfig({ targets: ['augment'] }).catch(error => log(`Augment MCP config write failed: ${error instanceof Error ? error.message : String(error)}`));
  });
  const cloneStackDisposable = vscode.commands.registerCommand('contextEngineUploader.cloneAndStartStack', () => {
    if (!onboardingManager || typeof onboardingManager.cloneAndStartStack !== 'function') {
      vscode.window.showErrorMessage('Context Engine onboarding is unavailable in this session.');
      return;
    }
    onboardingManager.cloneAndStartStack();
  });
  const startStackDisposable = vscode.commands.registerCommand('contextEngineUploader.startSavedStack', () => {
    if (!onboardingManager || typeof onboardingManager.startSavedStack !== 'function') {
      vscode.window.showErrorMessage('Context Engine onboarding is unavailable in this session.');
      return;
    }
    onboardingManager.startSavedStack();
  });
  const showLogsDisposable = vscode.commands.registerCommand('contextEngineUploader.showUploadServiceLogs', () => {
    try {
      if (outputChannel) {
        outputChannel.show(true);
      } else {
        vscode.window.showErrorMessage('Context Engine Uploader: output channel is unavailable.');
      }
    } catch (e) {
      log(`Show logs failed: ${e && e.message ? e.message : String(e)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to show logs. See output for details.');
    }
  });
  const tailDockerLogsDisposable = vscode.commands.registerCommand('contextEngineUploader.tailUploadServiceLogs', () => {
    try {
      if (logsTerminalManager && typeof logsTerminalManager.open === 'function') {
        logsTerminalManager.open();
      } else {
        vscode.window.showErrorMessage('Context Engine Uploader: log tailing is unavailable (extension failed to initialize logs terminal manager). See output for details.');
      }
    } catch (e) {
      log(`Tail docker logs failed: ${e && e.message ? e.message : String(e)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to tail upload service logs. See output for details.');
    }
  });
  const startBridgeDisposable = vscode.commands.registerCommand('contextEngineUploader.startMcpHttpBridge', () => {
    startHttpBridgeProcess().catch(error => {
      log(`HTTP MCP bridge start failed: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to start HTTP MCP bridge. Check Output for details.');
    });
  });
  const stopBridgeDisposable = vscode.commands.registerCommand('contextEngineUploader.stopMcpHttpBridge', () => {
    stopHttpBridgeProcess().catch(error => {
      log(`HTTP MCP bridge stop failed: ${error instanceof Error ? error.message : String(error)}`);
    });
  });
  const promptEnhanceDisposable = vscode.commands.registerCommand('contextEngineUploader.promptEnhance', () => {
    try {
      if (promptPlusManager && typeof promptPlusManager.enhanceSelectionWithUnicorn === 'function') {
        promptPlusManager.enhanceSelectionWithUnicorn().catch(error => {
          log(`Prompt+ failed: ${error instanceof Error ? error.message : String(error)}`);
          vscode.window.showErrorMessage('Prompt+ failed. See Context Engine Upload output.');
        });
      } else {
        vscode.window.showErrorMessage('Context Engine Uploader: Prompt+ is unavailable (extension failed to initialize Prompt+ manager). See output for details.');
      }
    } catch (error) {
      log(`Prompt+ failed: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Prompt+ failed. See Context Engine Upload output.');
    }
  });
  const authLoginDisposable = vscode.commands.registerCommand('contextEngineUploader.authLogin', () => {
    try {
      const cfg = getEffectiveConfig();
      const endpoint = (cfg.get('endpoint') || '').trim();
      runAuthLoginFlow(endpoint || undefined, buildAuthDeps()).catch(error => {
        log(`Auth login failed: ${error instanceof Error ? error.message : String(error)}`);
        vscode.window.showErrorMessage('Context Engine Uploader: auth login failed. See output for details.');
      });
    } catch (error) {
      runAuthLoginFlow(undefined, buildAuthDeps()).catch(error2 => {
        log(`Auth login failed: ${error2 instanceof Error ? error2.message : String(error2)}`);
        vscode.window.showErrorMessage('Context Engine Uploader: auth login failed. See output for details.');
      });
    }
  });
  const configDisposable = vscode.workspace.onDidChangeConfiguration(event => {
    if (event.affectsConfiguration('contextEngineUploader') && watchProcess) {
      runSequence('auto').catch(error => log(`Auto-restart failed: ${error instanceof Error ? error.message : String(error)}`));
    }
    if (event.affectsConfiguration('contextEngineUploader.targetPath')) {
      updateStatusBarTooltip();
    }
    if (
      event.affectsConfiguration('contextEngineUploader.mcpIndexerUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpMemoryUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpClaudeEnabled') ||
      event.affectsConfiguration('contextEngineUploader.mcpWindsurfEnabled') ||
      event.affectsConfiguration('contextEngineUploader.mcpAugmentEnabled') ||
      event.affectsConfiguration('contextEngineUploader.mcpTransportMode') ||
      event.affectsConfiguration('contextEngineUploader.windsurfMcpPath') ||
      event.affectsConfiguration('contextEngineUploader.augmentMcpPath') ||
      event.affectsConfiguration('contextEngineUploader.claudeHookEnabled') ||
      event.affectsConfiguration('contextEngineUploader.surfaceQdrantCollectionHint')
    ) {
      // Best-effort auto-update of MCP + hook configurations when settings change
      writeMcpConfig().catch(error => log(`Auto MCP config write failed: ${error instanceof Error ? error.message : String(error)}`));
    }
    if (
      event.affectsConfiguration('contextEngineUploader.autoStartMcpBridge') ||
      event.affectsConfiguration('contextEngineUploader.mcpBridgePort') ||
      event.affectsConfiguration('contextEngineUploader.mcpBridgeBinPath') ||
      event.affectsConfiguration('contextEngineUploader.mcpBridgeLocalOnly') ||
      event.affectsConfiguration('contextEngineUploader.mcpIndexerUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpMemoryUrl') ||
      event.affectsConfiguration('contextEngineUploader.mcpServerMode') ||
      event.affectsConfiguration('contextEngineUploader.mcpTransportMode')
    ) {
      handleHttpBridgeSettingsChanged().catch(error => log(`HTTP MCP bridge restart failed: ${error instanceof Error ? error.message : String(error)}`));
    }
  });
  const workspaceDisposable = vscode.workspace.onDidChangeWorkspaceFolders(() => {
    ensureTargetPathConfigured();
  });
  const terminalCloseDisposable = vscode.window.onDidCloseTerminal(term => {
    try {
      if (logsTerminalManager && typeof logsTerminalManager.handleDidCloseTerminal === 'function') {
        logsTerminalManager.handleDidCloseTerminal(term);
      }
    } catch (_) {}
  });
  context.subscriptions.push(
    startDisposable,
    stopDisposable,
    restartDisposable,
    indexDisposable,
    uploadGitHistoryDisposable,
    cloneStackDisposable,
    startStackDisposable,
    showLogsDisposable,
    tailDockerLogsDisposable,
    promptEnhanceDisposable,
    authLoginDisposable,
    startBridgeDisposable,
    stopBridgeDisposable,
    mcpConfigDisposable,
    mcpConfigSelectDisposable,
    mcpConfigClaudeDisposable,
    mcpConfigWindsurfDisposable,
    mcpConfigAugmentDisposable,
    ctxConfigDisposable,
    configDisposable,
    workspaceDisposable,
    terminalCloseDisposable
  );
  const config = getEffectiveConfig();
  ensureTargetPathConfigured();
// TODO: organise in another modulenise in another module
  try {
    const endpoint = (config.get('endpoint') || '').trim();
    const resolved = resolveTargetPathFromConfig(config);
    const targetPath = resolved && resolved.path ? String(resolved.path).trim() : '';
    const needsSetup = !endpoint || !targetPath;
    if (needsSetup && context && context.workspaceState) {
      const alreadyPrompted = !!context.workspaceState.get(ONBOARDING_PROMPT_KEY);
      if (!alreadyPrompted) {
        context.workspaceState.update(ONBOARDING_PROMPT_KEY, true).catch(() => {});
        vscode.window.showInformationMessage(
          'Context Engine Uploader: finish setup for this workspace to start indexing/uploading.',
          'Setup Workspace',
          'Later'
        ).then(choice => {
          if (choice === 'Setup Workspace') {
            vscode.commands.executeCommand('contextEngineUploader.setupWorkspace');
          }
        });
      }
    }
  } catch (_) {
  }
  if (config.get('runOnStartup')) {
    runSequence('auto').catch(error => log(`Startup run failed: ${error instanceof Error ? error.message : String(error)}`));
  }

  // Optionally keep MCP + hook + ctx config in sync on activation
  if (config.get('autoWriteMcpConfigOnStartup')) {
    writeMcpConfig().catch(error => log(`MCP config auto-write on activation failed: ${error instanceof Error ? error.message : String(error)}`));
  } else if (config.get('scaffoldCtxConfig', true)) {
    // Legacy behavior: scaffold ctx_config.json/.env directly when MCP auto-write is disabled
    writeCtxConfig().catch(error => log(`CTX config auto-scaffold on activation failed: ${error instanceof Error ? error.message : String(error)}`));
  }
  if (config.get('autoStartMcpBridge', false)) {
    const transportModeRaw = config.get('mcpTransportMode') || 'sse-remote';
    const serverModeRaw = config.get('mcpServerMode') || 'bridge';
    const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
    const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
    if (requiresHttpBridge(serverMode, transportMode)) {
      startHttpBridgeProcess().catch(error => log(`Auto-start HTTP MCP bridge failed: ${error instanceof Error ? error.message : String(error)}`));
    } else {
      log('Context Engine Uploader: autoStartMcpBridge is enabled, but current MCP wiring does not use the HTTP bridge; skipping auto-start.');
    }
  }
}
function buildAuthDeps() {
  return {
    vscode,
    spawn,
    spawnSync,
    resolveBridgeCliInvocation,
    getWorkspaceFolderPath,
    attachOutput,
    log,
    getEffectiveConfig,
    fetchGlobal: (typeof fetch === 'function' ? fetch : undefined),
  };
}
async function runSequence(mode = 'auto') {
  const options = resolveOptions();
  if (!options) {
    return;
  }

  try {
    await ensureAuthIfRequired(options.endpoint, buildAuthDeps());
  } catch (error) {
    log(`Auth preflight check failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  const depsSatisfied = await ensurePythonDependencies(options.pythonPath);
  if (!depsSatisfied) {
    setStatusBarState('idle');
    return;
  }
  // Re-resolve options in case ensurePythonDependencies switched to a better interpreter
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
  const config = getEffectiveConfig();
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
function resolveTargetPathFromConfig(config) {
  if (workspacePathUtils && typeof workspacePathUtils.resolveTargetPathFromConfig === 'function') {
    return workspacePathUtils.resolveTargetPathFromConfig(config);
  }
  return { path: (config.get('targetPath') || '').trim(), inspected: {}, inferred: false };
}

function getTargetPath(config) {
  if (workspacePathUtils && typeof workspacePathUtils.getTargetPath === 'function') {
    return workspacePathUtils.getTargetPath(config);
  }
  return undefined;
}
function saveTargetPath(config, targetPath) {
  if (workspacePathUtils && typeof workspacePathUtils.saveTargetPath === 'function') {
    return workspacePathUtils.saveTargetPath(config, targetPath);
  }
}
function getWorkspaceFolderPath() {
  if (workspacePathUtils && typeof workspacePathUtils.getWorkspaceFolderPath === 'function') {
    return workspacePathUtils.getWorkspaceFolderPath();
  }
  const folders = vscode.workspace.workspaceFolders;
  return (folders && folders.length) ? folders[0].uri.fsPath : undefined;
}
function scheduleMcpConfigRefreshAfterBridge(delayMs = 1500) {
  try {
    if (mcpConfigManager && typeof mcpConfigManager.scheduleMcpConfigRefreshAfterBridge === 'function') {
      return mcpConfigManager.scheduleMcpConfigRefreshAfterBridge(delayMs);
    }
  } catch (error) {
    log(`Context Engine Uploader: failed to schedule MCP config refresh: ${error instanceof Error ? error.message : String(error)}`);
  }
}
function detectDefaultTargetPath(workspaceFolderPath) {
  if (workspacePathUtils && typeof workspacePathUtils.detectDefaultTargetPath === 'function') {
    return workspacePathUtils.detectDefaultTargetPath(workspaceFolderPath);
  }
  return workspaceFolderPath;
}

function resolveBridgeWorkspacePath() {
  try {
    const settings = getEffectiveConfig();
    const target = getTargetPath(settings);
    if (target) {
      return path.resolve(target);
    }
  } catch (error) {
    log(`Context Engine Uploader: failed to resolve bridge workspace path via getTargetPath: ${error instanceof Error ? error.message : String(error)}`);
  }
  const fallbackFolder = getWorkspaceFolderPath();
  if (!fallbackFolder) {
    return undefined;
  }
  try {
    const autoTarget = detectDefaultTargetPath(fallbackFolder);
    return autoTarget ? path.resolve(autoTarget) : path.resolve(fallbackFolder);
  } catch (error) {
    log(`Context Engine Uploader: failed fallback bridge workspace path detection: ${error instanceof Error ? error.message : String(error)}`);
    return undefined;
  }
}

function ensureTargetPathConfigured() {
  const config = getEffectiveConfig();
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
  const autoTarget = detectDefaultTargetPath(folderPath);
  updateStatusBarTooltip(autoTarget);
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
  // Probe current interpreter with bundled python_libs first
  let ok = await checkPythonDeps(pythonPath);
  if (ok) {
    return true;
  }

  // If that fails, try to auto-detect a better system Python before falling back to a venv
  const autoPython = await detectSystemPython();
  if (autoPython && autoPython !== pythonPath) {
    log(`Falling back to auto-detected Python interpreter: ${autoPython}`);
    ok = await checkPythonDeps(autoPython);
    if (ok) {
      pythonOverridePath = autoPython;
      return true;
    }
  }

  // As a last resort, offer to create a private venv and install deps via pip
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
  return await checkPythonDeps(venvPython);
}

async function checkPythonDeps(pythonPath) {
  const missing = [];
  let pythonError;
  const env = { ...process.env };
  try {
    const libsPath = path.join(extensionRoot, 'python_libs');
    if (fs.existsSync(libsPath)) {
      const existing = env.PYTHONPATH || '';
      env.PYTHONPATH = existing ? `${libsPath}${path.delimiter}${existing}` : libsPath;
      log(`Using bundled python_libs at ${libsPath} for dependency check.`);
    }
  } catch (error) {
    log(`Failed to configure PYTHONPATH for dependency check: ${error instanceof Error ? error.message : String(error)}`);
  }
  for (const moduleName of REQUIRED_PYTHON_MODULES) {
    const check = spawnSync(pythonPath, ['-c', `import ${moduleName}`], { encoding: 'utf8', env });
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
    const cfg = getEffectiveConfig();
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

function requiresHttpBridge(serverMode, transportMode) {
  try {
    if (bridgeManager && typeof bridgeManager.requiresHttpBridge === 'function') {
      return bridgeManager.requiresHttpBridge(serverMode, transportMode);
    }
  } catch (_) {
    // ignore
  }
  return serverMode === 'bridge' && transportMode === 'http';
}

async function ensureHttpBridgeReadyForConfigs() {
  try {
    if (bridgeManager && typeof bridgeManager.ensureReadyForConfigs === 'function') {
      return await bridgeManager.ensureReadyForConfigs();
    }
  } catch (error) {
    log(`Failed to ensure HTTP bridge is ready: ${error instanceof Error ? error.message : String(error)}`);
  }
  return false;
}

function resolveBridgeHttpUrl() {
  try {
    if (bridgeManager && typeof bridgeManager.resolveBridgeHttpUrl === 'function') {
      return bridgeManager.resolveBridgeHttpUrl();
    }
  } catch (_) {
    // ignore
  }
  return undefined;
}

async function startHttpBridgeProcess() {
  if (bridgeManager && typeof bridgeManager.start === 'function') {
    return bridgeManager.start();
  }
  return undefined;
}

function stopHttpBridgeProcess() {
  if (bridgeManager && typeof bridgeManager.stop === 'function') {
    return bridgeManager.stop();
  }
  return Promise.resolve();
}

async function handleHttpBridgeSettingsChanged() {
  if (bridgeManager && typeof bridgeManager.handleSettingsChanged === 'function') {
    return bridgeManager.handleSettingsChanged();
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
    const baseEnv = buildChildEnv(options);
    const childEnv = { ...baseEnv, REMOTE_UPLOAD_GIT_FORCE: '1' };
    const child = spawn(options.pythonPath, args, { cwd: options.workingDirectory, env: childEnv });
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
      const chunk = data.toString();
      outputChannel.append(`[${label} err] ${chunk}`);
    });
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
function terminateProcess(proc, label, afterStop) {
  if (!proc) {
    return Promise.resolve();
  }
  return new Promise(resolve => {
    let finished = false;
    let termTimer;
    let killTimer;
    const clearTimers = () => {
      if (termTimer) clearTimeout(termTimer);
      if (killTimer) clearTimeout(killTimer);
    };
    const finalize = (reason) => {
      if (finished) return;
      finished = true;
      clearTimers();
      if (typeof afterStop === 'function') {
        afterStop();
      }
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
  try {
    const settings = getEffectiveConfig();
    const devRemoteMode = settings.get('devRemoteMode', false);
    if (devRemoteMode) {
      // Enable dev-remote upload mode for the standalone upload client.
      // This causes standalone_upload_client.py to ignore any 'dev-workspace'
      // directories when scanning for files to upload.
      env.REMOTE_UPLOAD_MODE = 'development';
      env.DEV_REMOTE_MODE = '1';
      log('Context Engine Uploader: devRemoteMode enabled (REMOTE_UPLOAD_MODE=development, DEV_REMOTE_MODE=1).');
    }
    const gitMaxCommits = settings.get('gitMaxCommits');
    if (typeof gitMaxCommits === 'number' && !Number.isNaN(gitMaxCommits)) {
      env.REMOTE_UPLOAD_GIT_MAX_COMMITS = String(gitMaxCommits);
    }
    const gitSinceRaw = settings.get('gitSince');
    const gitSince = typeof gitSinceRaw === 'string' ? gitSinceRaw.trim() : '';
    if (gitSince) {
      env.REMOTE_UPLOAD_GIT_SINCE = gitSince;
    }
  } catch (error) {
    log(`Failed to read devRemoteMode setting: ${error instanceof Error ? error.message : String(error)}`);
  }
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
function normalizeBridgeUrl(url) {
  if (!url || typeof url !== 'string') {
    return '';
  }
  const trimmed = url.trim();
  if (!trimmed) {
    return '';
  }
  return trimmed;
}

function normalizeWorkspaceForBridge(workspacePath) {
  if (!workspacePath || typeof workspacePath !== 'string') {
    return '';
  }
  try {
    const resolved = path.resolve(workspacePath);
    if (process.platform === 'win32') {
      return resolved.replace(/\//g, '\\');
    }
    return resolved;
  } catch (_) {
    return workspacePath;
  }
}

async function writeMcpConfig() {
  const options = arguments.length ? arguments[0] : undefined;
  if (mcpConfigManager && typeof mcpConfigManager.writeMcpConfig === 'function') {
    return mcpConfigManager.writeMcpConfig(options);
  }
}

async function writeCtxConfig() {
  if (ctxConfigManager && typeof ctxConfigManager.writeCtxConfig === 'function') {
    return ctxConfigManager.writeCtxConfig();
  }
}
function deactivate() {
  disposeIndexedWatcher();
  return Promise.all([stopProcesses()]);
}
module.exports = {
  activate,
  deactivate
};

function resolveBridgeCliInvocation() {
  const binPath = findLocalBridgeBin();
  if (binPath) {
    return {
      command: 'node',
      args: [binPath],
      kind: 'local'
    };
  }
  const isWindows = process.platform === 'win32';
  if (isWindows) {
    return {
      command: 'cmd',
      args: ['/c', 'npx', '@context-engine-bridge/context-engine-mcp-bridge'],
      kind: 'npx'
    };
  }
  return {
    command: 'npx',
    args: ['@context-engine-bridge/context-engine-mcp-bridge'],
    kind: 'npx'
  };
}

function findLocalBridgeBin() {
  let localOnly = true;
  let configured = '';
  try {
    const settings = getEffectiveConfig();
    localOnly = settings.get('mcpBridgeLocalOnly', true);
    configured = (settings.get('mcpBridgeBinPath') || '').trim();
  } catch (_) {
    // ignore config lookup failures and fall back to env/npx behavior
  }

  // When local-only is disabled, skip local resolution and always fall back to npx
  if (localOnly === false) {
    return undefined;
  }

  if (configured && fs.existsSync(configured)) {
    return path.resolve(configured);
  }

  const envOverride = (process.env.CTXCE_BRIDGE_BIN || '').trim();
  if (envOverride && fs.existsSync(envOverride)) {
    return path.resolve(envOverride);
  }

  return undefined;
}
