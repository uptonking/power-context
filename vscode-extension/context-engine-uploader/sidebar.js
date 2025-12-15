const vscode = require('vscode');

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
        makeTreeItem('Upload & Watch', { collapsibleState: vscode.TreeItemCollapsibleState.Expanded, contextValue: 'ctxceActionsUploadRoot' }),
        makeTreeItem('MCP Bridge & Config', { collapsibleState: vscode.TreeItemCollapsibleState.Collapsed, contextValue: 'ctxceActionsBridgeRoot' }),
        makeTreeItem('Utilities', { collapsibleState: vscode.TreeItemCollapsibleState.Collapsed, contextValue: 'ctxceActionsUtilitiesRoot' }),
      ];
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
      return [
        makeTreeItem('Prompt+', { icon: new vscode.ThemeIcon('sparkle'), command: { command: 'contextEngineUploader.promptEnhance', title: 'Prompt+' } }),
        makeTreeItem('Sign In', { icon: new vscode.ThemeIcon('account'), command: { command: 'contextEngineUploader.authLogin', title: 'Sign In' } }),
      ];
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
