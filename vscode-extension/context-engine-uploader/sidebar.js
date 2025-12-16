const vscode = require('vscode');
const fs = require('fs');
const path = require('path');

function makeTreeItem(label, opts = {}) {
  const item = new vscode.TreeItem(label, opts.collapsibleState || vscode.TreeItemCollapsibleState.None);
  if (opts.description !== undefined) {
    item.description = opts.description;
  }
  if (opts.tooltip !== undefined) {
    item.tooltip = opts.tooltip;
  }
  if (opts.icon !== undefined) {
    item.iconPath = opts.icon;
  }
  if (opts.command !== undefined) {
    item.command = opts.command;
  }
  if (opts.contextValue !== undefined) {
    item.contextValue = opts.contextValue;
  }
  return item;
}

function createProvider(getChildrenFn) {
  const emitter = new vscode.EventEmitter();
  return {
    onDidChangeTreeData: emitter.event,
    refresh: () => emitter.fire(undefined),
    getTreeItem: element => element,
    getChildren: async element => getChildrenFn(element),
  };
}

function resolveMcpMode(cfg) {
  try {
    const transportModeRaw = cfg.get('mcpTransportMode') || 'sse-remote';
    const serverModeRaw = cfg.get('mcpServerMode') || 'bridge';
    const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
    const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
    return `${serverMode}/${transportMode}`;
  } catch (_) {
    return 'unknown';
  }
}

function getWorkspaceRoot() {
  try {
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length) {
      return folders[0].uri.fsPath;
    }
  } catch (_) {
    // ignore
  }
  return undefined;
}

function pathExists(p, expectDir = false) {
  try {
    if (!p) {
      return false;
    }
    const stat = fs.statSync(p);
    return expectDir ? stat.isDirectory() : stat.isFile();
  } catch (_) {
    return false;
  }
}

function findConfigFile(bases, filename) {
  for (const base of bases) {
    if (!base) {
      continue;
    }
    const candidate = path.join(base, filename);
    if (pathExists(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

const _authStatusCache = new Map();
const _AUTH_STATUS_TTL_MS = 30_000;
const _AUTH_STATUS_TIMEOUT_MS = 750;

async function probeAuthEnabled(endpoint) {
  const base = (endpoint || '').trim().replace(/\/+$/, '');
  if (!base) {
    return undefined;
  }

  const fetchFn = (typeof fetch === 'function' ? fetch : undefined);
  if (!fetchFn) {
    return undefined;
  }

  let timer;
  let controller;
  try {
    controller = (typeof AbortController === 'function') ? new AbortController() : undefined;
    if (controller) {
      timer = setTimeout(() => {
        try { controller.abort(); } catch (_) { }
      }, _AUTH_STATUS_TIMEOUT_MS);
    }
    const res = await fetchFn(`${base}/auth/status`, { method: 'GET', signal: controller ? controller.signal : undefined });
    if (!res || !res.ok) {
      return undefined;
    }
    const body = await res.json();
    if (body && typeof body.enabled === 'boolean') {
      return body.enabled;
    }
    return undefined;
  } catch (_) {
    return undefined;
  } finally {
    if (timer) {
      clearTimeout(timer);
    }
  }
}

async function getCachedAuthEnabled(endpoint) {
  const key = (endpoint || '').trim();
  if (!key) {
    return undefined;
  }
  const now = Date.now();
  const cached = _authStatusCache.get(key);
  if (cached && cached.ts && (now - cached.ts) < _AUTH_STATUS_TTL_MS) {
    return cached.enabled;
  }
  const enabled = await probeAuthEnabled(key);
  _authStatusCache.set(key, { enabled, ts: now });
  return enabled;
}

function register(context, deps) {
  const profiles = deps && deps.profiles;
  const getEffectiveConfig = deps && deps.getEffectiveConfig;
  const getResolvedTargetPath = deps && deps.getResolvedTargetPath;
  const getState = deps && deps.getState;

  const providers = [];

  const profilesProvider = createProvider(async element => {
    if (!profiles || typeof profiles.listProfiles !== 'function') {
      return [];
    }

    if (!element) {
      return [
        makeTreeItem('Active Profile', { collapsibleState: vscode.TreeItemCollapsibleState.Expanded, contextValue: 'ctxceProfilesActiveRoot' }),
        makeTreeItem('Manage Profiles', { collapsibleState: vscode.TreeItemCollapsibleState.Collapsed, contextValue: 'ctxceProfilesManageRoot' }),
      ];
    }

    if (element.contextValue === 'ctxceProfilesActiveRoot') {
      const active = typeof profiles.getActiveProfileSummary === 'function' ? profiles.getActiveProfileSummary() : { id: undefined, name: undefined };
      const list = profiles.listProfiles();
      const items = [];

      items.push(makeTreeItem(`Current: ${active && active.id ? (active.name || active.id) : 'None'}`, {
        description: active && active.id ? active.id : '',
        icon: new vscode.ThemeIcon(active && active.id ? 'check' : 'circle-slash'),
        contextValue: 'ctxceProfilesCurrent',
      }));

      items.push(makeTreeItem('None (use VS Code settings)', {
        icon: new vscode.ThemeIcon(!active || !active.id ? 'check' : 'circle-slash'),
        command: {
          command: 'contextEngineUploader._setActiveProfile',
          title: 'Set Active Profile',
          arguments: [''],
        },
        contextValue: 'ProfilesChoice',
      }));

      for (const p of list) {
        const isActive = !!(active && active.id && p.id === active.id);
        items.push(makeTreeItem(p.name || p.id, {
          description: p.id,
          icon: new vscode.ThemeIcon(isActive ? 'check' : 'circle-outline'),
          command: {
            command: 'contextEngineUploader._setActiveProfile',
            title: 'Set Active Profile',
            arguments: [p.id],
          },
          contextValue: 'ProfilesChoice',
        }));
      }

      return items;
    }

    if (element.contextValue === 'ctxceProfilesManageRoot') {
      return [
        makeTreeItem('Setup Workspace', { icon: new vscode.ThemeIcon('rocket'), command: { command: 'contextEngineUploader.setupWorkspace', title: 'Setup Workspace' } }),
        makeTreeItem('Switch Profile', { icon: new vscode.ThemeIcon('sync'), command: { command: 'contextEngineUploader.switchProfile', title: 'Switch Profile' } }),
        makeTreeItem('Create Profile From Current Settings', { icon: new vscode.ThemeIcon('add'), command: { command: 'contextEngineUploader.createProfileFromCurrentSettings', title: 'Create Profile From Current Settings' } }),
        makeTreeItem('Import Profiles', { icon: new vscode.ThemeIcon('cloud-download'), command: { command: 'contextEngineUploader.importProfiles', title: 'Import Profiles' } }),
        makeTreeItem('Export Profiles', { icon: new vscode.ThemeIcon('cloud-upload'), command: { command: 'contextEngineUploader.exportProfiles', title: 'Export Profiles' } }),
      ];
    }

    return [];
  });

  const statusProvider = createProvider(async () => {
    const cfg = getEffectiveConfig ? getEffectiveConfig() : vscode.workspace.getConfiguration('contextEngineUploader');
    const state = typeof getState === 'function' ? getState() : {};

    const endpoint = (() => {
      try { return (cfg.get('endpoint') || '').trim(); } catch (_) { return ''; }
    })();

    const resolvedTarget = typeof getResolvedTargetPath === 'function' ? getResolvedTargetPath() : undefined;
    const targetDescription = resolvedTarget && resolvedTarget.path ? resolvedTarget.path : '(unset)';
    const targetTooltip = resolvedTarget && resolvedTarget.source ? `Source: ${resolvedTarget.source}` : undefined;

    const mcpMode = resolveMcpMode(cfg);
    const bridge = state && state.httpBridgeProcess ? `running:${state.httpBridgePort || ''}` : 'stopped';

    return [
      makeTreeItem('Run State', { description: state && state.statusMode ? state.statusMode : 'unknown', icon: new vscode.ThemeIcon('pulse') }),
      makeTreeItem('Endpoint', { description: endpoint || '(unset)', icon: new vscode.ThemeIcon('link') }),
      makeTreeItem('Target Path', { description: targetDescription, tooltip: targetTooltip, icon: new vscode.ThemeIcon('folder') }),
      makeTreeItem('MCP Mode', { description: mcpMode, icon: new vscode.ThemeIcon('plug') }),
      makeTreeItem('HTTP Bridge', { description: bridge, icon: new vscode.ThemeIcon('server-process') }),
    ];
  });

  const actionsProvider = createProvider(async (element) => {
    if (!element) {
      return [
        makeTreeItem('Getting Started', { collapsibleState: vscode.TreeItemCollapsibleState.Expanded, contextValue: 'ctxceActionsGettingStartedRoot' }),
        makeTreeItem('Upload & Watch', { collapsibleState: vscode.TreeItemCollapsibleState.Expanded, contextValue: 'ctxceActionsUploadRoot' }),
        makeTreeItem('MCP Bridge & Config', { collapsibleState: vscode.TreeItemCollapsibleState.Collapsed, contextValue: 'ctxceActionsBridgeRoot' }),
        makeTreeItem('Utilities', { collapsibleState: vscode.TreeItemCollapsibleState.Collapsed, contextValue: 'ctxceActionsUtilitiesRoot' }),
      ];
    }

    if (element.contextValue === 'ctxceActionsGettingStartedRoot') {
      const cfg = getEffectiveConfig ? getEffectiveConfig() : vscode.workspace.getConfiguration('contextEngineUploader');

      const endpoint = (() => {
        try { return (cfg.get('endpoint') || '').trim(); } catch (_) { return ''; }
      })();

      const resolvedTarget = typeof getResolvedTargetPath === 'function' ? getResolvedTargetPath() : undefined;
      const targetDescription = resolvedTarget && resolvedTarget.path ? resolvedTarget.path : '(unset)';
      const targetPath = resolvedTarget && resolvedTarget.path ? String(resolvedTarget.path) : '';
      const targetExists = pathExists(targetPath, true);
      const workspaceRoot = getWorkspaceRoot();
      const configSearchBases = [targetPath, workspaceRoot].filter(Boolean);
      const mcpConfigPath = findConfigFile(configSearchBases, '.mcp.json');
      const ctxConfigPath = findConfigFile(configSearchBases, 'ctx_config.json');

      const bridgeMode = (() => {
        const mode = resolveMcpMode(cfg);
        return typeof mode === 'string' && mode.startsWith('bridge/');
      })();

      const authEnabled = bridgeMode && endpoint ? await getCachedAuthEnabled(endpoint) : undefined;
      const showAuth = !!(bridgeMode && endpoint && (authEnabled === true || authEnabled === undefined));

      const missingEndpoint = !endpoint;
      const missingTarget = !targetPath || !targetExists;
      const missingMcpConfig = !mcpConfigPath;
      const missingCtxConfig = !ctxConfigPath;

      const items = [
        makeTreeItem('Setup Workspace', {
          icon: new vscode.ThemeIcon('rocket'),
          command: { command: 'contextEngineUploader.setupWorkspace', title: 'Setup Workspace' },
          tooltip: 'Create and configure a profile for this workspace (recommended for first-time setup).',
        }),
        makeTreeItem('Open Settings', {
          icon: new vscode.ThemeIcon('gear'),
          command: { command: 'workbench.action.openSettings', title: 'Open Settings', arguments: ['contextEngineUploader'] },
          tooltip: 'Open VS Code settings filtered to Context Engine Uploader.',
        }),
      ];

      if (missingEndpoint) {
        items.push(makeTreeItem('Check Endpoint', {
          description: '(unset)',
          icon: new vscode.ThemeIcon('warning'),
          command: { command: 'workbench.action.openSettings', title: 'Open Settings', arguments: ['contextEngineUploader.endpoint'] },
          tooltip: 'Configure the upload service endpoint.',
        }));
      }

      if (missingTarget) {
        items.push(makeTreeItem('Check Target Path', {
          description: targetPath ? `${targetDescription} (missing)` : '(unset)',
          icon: new vscode.ThemeIcon('warning'),
          command: { command: 'workbench.action.openSettings', title: 'Open Settings', arguments: ['contextEngineUploader.targetPath'] },
          tooltip: 'Configure the target path to index/upload.',
        }));
      }

      if (missingMcpConfig) {
        items.push(makeTreeItem('Write MCP Config (.mcp.json)', {
          description: 'Missing',
          icon: new vscode.ThemeIcon('warning'),
          command: { command: 'contextEngineUploader.writeMcpConfig', title: 'Write MCP Config (.mcp.json)' },
          tooltip: 'Writes project MCP configuration for IDE clients (Claude/Windsurf) based on current settings.',
        }));
      }

      if (missingCtxConfig) {
        items.push(makeTreeItem('Write CTX Config (ctx_config.json)', {
          description: 'Missing',
          icon: new vscode.ThemeIcon('warning'),
          command: { command: 'contextEngineUploader.writeCtxConfig', title: 'Write CTX Config' },
          tooltip: 'Scaffold ctx_config.json/.env for local tools (Prompt+, etc.).',
        }));
      }

      if (!missingEndpoint && !missingTarget && !missingMcpConfig && !missingCtxConfig) {
        items.push(makeTreeItem('All set', {
          description: 'Ready',
          icon: new vscode.ThemeIcon('check'),
          tooltip: 'Workspace looks ready. You can start indexing/uploading now.',
        }));
      }

      if (showAuth) {
        items.push(makeTreeItem('Sign In', {
          icon: new vscode.ThemeIcon('account'),
          command: { command: 'contextEngineUploader.authLogin', title: 'Sign In' },
          tooltip: 'Runs ctxce auth login for the configured endpoint.',
        }));
      }

      return items;
    }

    if (element.contextValue === 'ctxceActionsUploadRoot') {
      return [
        makeTreeItem('Force Index Now', {
          icon: new vscode.ThemeIcon('sync'),
          command: { command: 'contextEngineUploader.indexCodebase', title: 'Index Codebase' },
          tooltip: 'Runs the force upload once (then watch if enabled).',
        }),
        makeTreeItem('Start Upload / Watch', {
          icon: new vscode.ThemeIcon('play'),
          command: { command: 'contextEngineUploader.start', title: 'Start Upload / Watch' },
        }),
        makeTreeItem('Stop Upload / Watch', {
          icon: new vscode.ThemeIcon('stop'),
          command: { command: 'contextEngineUploader.stop', title: 'Stop Upload / Watch' },
        }),
        makeTreeItem('Restart Upload / Watch', {
          icon: new vscode.ThemeIcon('refresh'),
          command: { command: 'contextEngineUploader.restart', title: 'Restart Upload / Watch' },
        }),
        makeTreeItem('Show Extension Output', {
          icon: new vscode.ThemeIcon('output'),
          command: { command: 'contextEngineUploader.showUploadServiceLogs', title: 'Show Extension Output' },
        }),
        makeTreeItem('Tail Upload Service Logs (Docker)', {
          icon: new vscode.ThemeIcon('terminal'),
          command: { command: 'contextEngineUploader.tailUploadServiceLogs', title: 'Tail Upload Service Logs (Docker)' },
        }),
      ];
    }

    if (element.contextValue === 'ctxceActionsBridgeRoot') {
      return [
        makeTreeItem('Write MCP Config (.mcp.json)', {
          icon: new vscode.ThemeIcon('file-code'),
          command: { command: 'contextEngineUploader.writeMcpConfig', title: 'Write MCP Config (.mcp.json)' },
        }),
        makeTreeItem('Write CTX Config (ctx_config.json)', {
          icon: new vscode.ThemeIcon('file-text'),
          command: { command: 'contextEngineUploader.writeCtxConfig', title: 'Write CTX Config' },
        }),
        makeTreeItem('Start MCP HTTP Bridge', {
          icon: new vscode.ThemeIcon('broadcast'),
          command: { command: 'contextEngineUploader.startMcpHttpBridge', title: 'Start MCP HTTP Bridge' },
        }),
        makeTreeItem('Stop MCP HTTP Bridge', {
          icon: new vscode.ThemeIcon('debug-stop'),
          command: { command: 'contextEngineUploader.stopMcpHttpBridge', title: 'Stop MCP HTTP Bridge' },
        }),
      ];
    }

    if (element.contextValue === 'ctxceActionsUtilitiesRoot') {
      const cfg = getEffectiveConfig ? getEffectiveConfig() : vscode.workspace.getConfiguration('contextEngineUploader');
      const endpoint = (() => {
        try { return (cfg.get('endpoint') || '').trim(); } catch (_) { return ''; }
      })();
      const bridgeMode = (() => {
        const mode = resolveMcpMode(cfg);
        return typeof mode === 'string' && mode.startsWith('bridge/');
      })();
      const authEnabled = bridgeMode && endpoint ? await getCachedAuthEnabled(endpoint) : undefined;
      const showAuth = !!(bridgeMode && endpoint && (authEnabled === true || authEnabled === undefined));

      const items = [
        makeTreeItem('Prompt+', { icon: new vscode.ThemeIcon('sparkle'), command: { command: 'contextEngineUploader.promptEnhance', title: 'Prompt+' } }),
      ];

      if (showAuth) {
        items.push(makeTreeItem('Sign In', { icon: new vscode.ThemeIcon('account'), command: { command: 'contextEngineUploader.authLogin', title: 'Sign In' } }));
      }

      return items;
    }

    return [];
  });

  providers.push(profilesProvider, statusProvider, actionsProvider);

  const setActiveDisposable = vscode.commands.registerCommand('contextEngineUploader._setActiveProfile', async (profileId) => {
    if (!profiles || typeof profiles.setActiveProfileId !== 'function') {
      return;
    }
    await profiles.setActiveProfileId(profileId || undefined);
    profilesProvider.refresh();
    statusProvider.refresh();
  });

  context.subscriptions.push(setActiveDisposable);

  const profilesTree = vscode.window.createTreeView('contextEngineUploaderProfiles', { treeDataProvider: profilesProvider, showCollapseAll: false });
  const statusTree = vscode.window.createTreeView('contextEngineUploaderStatus', { treeDataProvider: statusProvider, showCollapseAll: false });
  const actionsTree = vscode.window.createTreeView('contextEngineUploaderActions', { treeDataProvider: actionsProvider, showCollapseAll: false });

  context.subscriptions.push(profilesTree, statusTree, actionsTree);

  let refreshTimer;
  const bumpTimer = () => {
    const visible = profilesTree.visible || statusTree.visible || actionsTree.visible;
    if (!visible) {
      if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = undefined;
      }
      return;
    }
    if (!refreshTimer) {
      refreshTimer = setInterval(() => {
        profilesProvider.refresh();
        statusProvider.refresh();
        actionsProvider.refresh();
      }, 2000);
    }
  };

  profilesTree.onDidChangeVisibility(bumpTimer, null, context.subscriptions);
  statusTree.onDidChangeVisibility(bumpTimer, null, context.subscriptions);
  actionsTree.onDidChangeVisibility(bumpTimer, null, context.subscriptions);

  bumpTimer();

  return {
    refresh: () => {
      for (const p of providers) {
        p.refresh();
      }
    },
  };
}

module.exports = {
  register,
};
