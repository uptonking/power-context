const path = require('path');
const fs = require('fs');
const os = require('os');

function createMcpConfigManager(deps) {
  const vscode = deps.vscode;
  const log = deps.log;

  const extensionRoot = deps.extensionRoot;
  const getEffectiveConfig = deps.getEffectiveConfig;
  const getWorkspaceFolderPath = deps.getWorkspaceFolderPath;
  const resolveBridgeWorkspacePath = deps.resolveBridgeWorkspacePath;
  const normalizeBridgeUrl = deps.normalizeBridgeUrl;
  const normalizeWorkspaceForBridge = deps.normalizeWorkspaceForBridge;
  const resolveBridgeCliInvocation = deps.resolveBridgeCliInvocation;

  const resolveBridgeHttpUrl = deps.resolveBridgeHttpUrl;
  const requiresHttpBridge = deps.requiresHttpBridge;
  const ensureHttpBridgeReadyForConfigs = deps.ensureHttpBridgeReadyForConfigs;
  const getBridgeIsRunning = deps.getBridgeIsRunning;

  const writeCtxConfig = deps.writeCtxConfig;

  let pendingBridgeConfigTimer;

  function dispose() {
    try {
      if (pendingBridgeConfigTimer) {
        clearTimeout(pendingBridgeConfigTimer);
        pendingBridgeConfigTimer = undefined;
      }
    } catch (_) {
      // ignore
    }
  }

  function getClaudeHookCommand() {
    const isLinux = process.platform === 'linux';
    if (!isLinux) {
      return '';
    }
    if (!extensionRoot) {
      log('Claude hook command resolution failed: extensionRoot is undefined.');
      return '';
    }
    try {
      const embeddedPath = path.join(extensionRoot, 'ctx-hook-simple.sh');
      if (fs.existsSync(embeddedPath)) {
        log(`Using embedded Claude hook at ${embeddedPath}`);
        return embeddedPath;
      }
      log(`Claude hook command resolution failed: ctx-hook-simple.sh not found at ${embeddedPath}`);
    } catch (error) {
      log(`Failed to resolve embedded Claude hook path: ${error instanceof Error ? error.message : String(error)}`);
    }
    return '';
  }

  function getDefaultWindsurfMcpPath() {
    return path.join(os.homedir(), '.codeium', 'windsurf', 'mcp_config.json');
  }

  function buildBridgeServerConfig(workspacePath, indexerUrl, memoryUrl) {
    const invocation = resolveBridgeCliInvocation();
    const args = [...invocation.args, 'mcp-serve'];
    if (workspacePath) {
      args.push('--workspace', normalizeWorkspaceForBridge(workspacePath));
    }
    if (indexerUrl) {
      args.push('--indexer-url', indexerUrl);
    }
    if (memoryUrl) {
      args.push('--memory-url', memoryUrl);
    }
    return {
      command: invocation.command,
      args,
      env: {},
    };
  }

  async function removeContextEngineFromWindsurfConfig() {
    try {
      const settings = getEffectiveConfig();
      const customPath = (settings.get('windsurfMcpPath') || '').trim();
      const configPath = customPath || getDefaultWindsurfMcpPath();
      if (!configPath) {
        return;
      }
      if (!fs.existsSync(configPath)) {
        // Nothing to remove yet.
        return;
      }
      let config = { mcpServers: {} };
      try {
        const raw = fs.readFileSync(configPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          config = parsed;
        }
      } catch (error) {
        log(`Context Engine Uploader: failed to parse Windsurf mcp_config.json when removing context-engine: ${error instanceof Error ? error.message : String(error)}`);
        return;
      }
      if (!config.mcpServers || typeof config.mcpServers !== 'object') {
        return;
      }
      if (!config.mcpServers['context-engine']) {
        return;
      }
      delete config.mcpServers['context-engine'];
      try {
        const json = JSON.stringify(config, null, 2) + '\n';
        fs.writeFileSync(configPath, json, 'utf8');
        log(`Context Engine Uploader: removed context-engine server from Windsurf MCP config at ${configPath} before HTTP bridge restart.`);
      } catch (error) {
        log(`Context Engine Uploader: failed to write Windsurf mcp_config.json when removing context-engine: ${error instanceof Error ? error.message : String(error)}`);
      }
    } catch (error) {
      log(`Context Engine Uploader: error while removing context-engine from Windsurf MCP config: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  function scheduleMcpConfigRefreshAfterBridge(delayMs = 1500) {
    try {
      if (pendingBridgeConfigTimer) {
        clearTimeout(pendingBridgeConfigTimer);
        pendingBridgeConfigTimer = undefined;
      }
      // For bridge-http mode started by the extension, Windsurf needs the
      // "context-engine" MCP server entry removed and then re-added once the
      // HTTP bridge is ready. Best-effort removal happens immediately here;
      // writeMcpConfig() below will re-write configs after the bridge comes up.
      try {
        const settings = getEffectiveConfig();
        const windsurfEnabled = settings.get('mcpWindsurfEnabled', false);
        const transportModeRaw = settings.get('mcpTransportMode') || 'sse-remote';
        const serverModeRaw = settings.get('mcpServerMode') || 'bridge';
        const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
        const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
        if (windsurfEnabled && serverMode === 'bridge' && transportMode === 'http') {
          removeContextEngineFromWindsurfConfig().catch(error => {
            log(`Context Engine Uploader: failed to remove context-engine from Windsurf MCP config before HTTP bridge restart: ${error instanceof Error ? error.message : String(error)}`);
          });
        }
      } catch (error) {
        log(`Context Engine Uploader: failed to prepare Windsurf MCP removal before HTTP bridge restart: ${error instanceof Error ? error.message : String(error)}`);
      }
      pendingBridgeConfigTimer = setTimeout(() => {
        pendingBridgeConfigTimer = undefined;
        log('Context Engine Uploader: HTTP bridge ready; refreshing MCP configs.');
        writeMcpConfig().catch(error => {
          log(`Context Engine Uploader: MCP config refresh after bridge start failed: ${error instanceof Error ? error.message : String(error)}`);
        });
      }, delayMs);
    } catch (error) {
      log(`Context Engine Uploader: failed to schedule MCP config refresh: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async function writeClaudeMcpServers(root, indexerUrl, memoryUrl, transportMode, serverMode = 'bridge') {
    const bridgeWorkspace = resolveBridgeWorkspacePath();
    const configPath = path.join(bridgeWorkspace || root, '.mcp.json');
    let config = { mcpServers: {} };
    if (fs.existsSync(configPath)) {
      try {
        const raw = fs.readFileSync(configPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          config = parsed;
        }
      } catch (error) {
        vscode.window.showErrorMessage('Context Engine Uploader: existing .mcp.json is invalid JSON; not modified.');
        log(`Failed to parse .mcp.json: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    }
    if (!config.mcpServers || typeof config.mcpServers !== 'object') {
      config.mcpServers = {};
    }
    log(`Preparing to write .mcp.json at ${configPath} with indexerUrl=${indexerUrl || '""'} memoryUrl=${memoryUrl || '""'}`);
    const servers = config.mcpServers;
    const mode = (typeof transportMode === 'string' ? transportMode.trim() : 'sse-remote') || 'sse-remote';

    if (serverMode === 'bridge') {
      if (mode === 'http') {
        const bridgeUrl = resolveBridgeHttpUrl();
        if (bridgeUrl) {
          servers['context-engine'] = {
            type: 'http',
            url: bridgeUrl,
          };
        } else {
          const bridgeServer = buildBridgeServerConfig(bridgeWorkspace || root, indexerUrl, memoryUrl);
          servers['context-engine'] = bridgeServer;
        }
      } else {
        const bridgeServer = buildBridgeServerConfig(bridgeWorkspace || root, indexerUrl, memoryUrl);
        servers['context-engine'] = bridgeServer;
      }
      delete servers['qdrant-indexer'];
      delete servers.memory;
    } else if (mode === 'http') {
      // Direct HTTP MCP endpoints for Claude (.mcp.json)
      if (indexerUrl) {
        servers['qdrant-indexer'] = {
          type: 'http',
          url: indexerUrl,
        };
      }
      if (memoryUrl) {
        servers.memory = {
          type: 'http',
          url: memoryUrl,
        };
      }
    } else {
      // Legacy/default: stdio via mcp-remote SSE bridge
      const isWindows = process.platform === 'win32';
      const makeServer = url => {
        if (isWindows) {
          return {
            command: 'cmd',
            args: ['/c', 'npx', 'mcp-remote', url, '--transport', 'sse-only'],
            env: {},
          };
        }
        return {
          command: 'npx',
          args: ['mcp-remote', url, '--transport', 'sse-only'],
          env: {},
        };
      };
      if (indexerUrl) {
        servers['qdrant-indexer'] = makeServer(indexerUrl);
      }
      if (memoryUrl) {
        servers.memory = makeServer(memoryUrl);
      }
    }
    try {
      const json = JSON.stringify(config, null, 2) + '\n';
      fs.writeFileSync(configPath, json, 'utf8');
      vscode.window.showInformationMessage('Context Engine Uploader: .mcp.json updated for Context Engine MCP servers.');
      log(`Wrote .mcp.json at ${configPath}`);
      return true;
    } catch (error) {
      vscode.window.showErrorMessage('Context Engine Uploader: failed to write .mcp.json.');
      log(`Failed to write .mcp.json: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  async function writeWindsurfMcpServers(configPath, indexerUrl, memoryUrl, transportMode, serverMode = 'bridge', workspaceHint) {
    try {
      fs.mkdirSync(path.dirname(configPath), { recursive: true });
    } catch (error) {
      log(`Failed to ensure Windsurf MCP directory: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to prepare Windsurf MCP directory.');
      return false;
    }
    let config = { mcpServers: {} };
    if (fs.existsSync(configPath)) {
      try {
        const raw = fs.readFileSync(configPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          config = parsed;
        }
      } catch (error) {
        vscode.window.showErrorMessage('Context Engine Uploader: existing Windsurf mcp_config.json is invalid JSON; not modified.');
        log(`Failed to parse Windsurf mcp_config.json: ${error instanceof Error ? error.message : String(error)}`);
        return false;
      }
    }
    if (!config.mcpServers || typeof config.mcpServers !== 'object') {
      config.mcpServers = {};
    }
    log(`Preparing to write Windsurf mcp_config.json at ${configPath} with indexerUrl=${indexerUrl || '""'} memoryUrl=${memoryUrl || '""'}`);
    const servers = config.mcpServers;
    const mode = (typeof transportMode === 'string' ? transportMode.trim() : 'sse-remote') || 'sse-remote';

    if (serverMode === 'bridge') {
      const bridgeWorkspace = resolveBridgeWorkspacePath() || workspaceHint || '';
      if (mode === 'http') {
        const bridgeUrl = resolveBridgeHttpUrl();
        if (bridgeUrl) {
          servers['context-engine'] = {
            serverUrl: bridgeUrl,
          };
        } else {
          const bridgeServer = buildBridgeServerConfig(bridgeWorkspace, indexerUrl, memoryUrl);
          servers['context-engine'] = bridgeServer;
        }
      } else {
        const bridgeServer = buildBridgeServerConfig(bridgeWorkspace, indexerUrl, memoryUrl);
        servers['context-engine'] = bridgeServer;
      }
      delete servers['qdrant-indexer'];
      delete servers.memory;
    } else if (mode === 'http') {
      // Direct HTTP MCP endpoints for Windsurf mcp_config.json
      if (indexerUrl) {
        servers['qdrant-indexer'] = {
          type: 'http',
          url: indexerUrl,
        };
      }
      if (memoryUrl) {
        servers.memory = {
          type: 'http',
          url: memoryUrl,
        };
      }
    } else {
      // Legacy/default: use mcp-remote SSE bridge
      const makeServer = url => {
        // Default args for local/HTTPS endpoints
        const args = ['mcp-remote', url, '--transport', 'sse-only'];
        try {
          const u = new URL(url);
          const isLocalHost =
            u.hostname === 'localhost' ||
            u.hostname === '127.0.0.1' ||
            u.hostname === '::1';
          // For non-local HTTP URLs, mcp-remote requires --allow-http
          if (u.protocol === 'http:' && !isLocalHost) {
            args.push('--allow-http');
          }
        } catch (e) {
          // If URL parsing fails, fall back to default args without additional flags
        }
        return {
          command: 'npx',
          args,
          env: {},
        };
      };
      if (indexerUrl) {
        servers['qdrant-indexer'] = makeServer(indexerUrl);
      }
      if (memoryUrl) {
        servers.memory = makeServer(memoryUrl);
      }
    }
    try {
      const json = JSON.stringify(config, null, 2) + '\n';
      fs.writeFileSync(configPath, json, 'utf8');
      vscode.window.showInformationMessage(`Context Engine Uploader: Windsurf MCP config updated at ${configPath}.`);
      log(`Wrote Windsurf mcp_config.json at ${configPath}`);
      return true;
    } catch (error) {
      vscode.window.showErrorMessage('Context Engine Uploader: failed to write Windsurf mcp_config.json.');
      log(`Failed to write Windsurf mcp_config.json: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  async function writeClaudeHookConfig(root, commandPath) {
    try {
      const claudeDir = path.join(root, '.claude');
      fs.mkdirSync(claudeDir, { recursive: true });
      const settingsPath = path.join(claudeDir, 'settings.local.json');
      let config = {};
      if (fs.existsSync(settingsPath)) {
        try {
          const raw = fs.readFileSync(settingsPath, 'utf8');
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === 'object') {
            config = parsed;
          }
        } catch (error) {
          vscode.window.showErrorMessage('Context Engine Uploader: existing .claude/settings.local.json is invalid JSON; not modified.');
          log(`Failed to parse .claude/settings.local.json: ${error instanceof Error ? error.message : String(error)}`);
          return false;
        }
      }
      if (!config.permissions || typeof config.permissions !== 'object') {
        config.permissions = { allow: [], deny: [], ask: [] };
      } else {
        config.permissions.allow = config.permissions.allow || [];
        config.permissions.deny = config.permissions.deny || [];
        config.permissions.ask = config.permissions.ask || [];
      }
      if (!config.enabledMcpjsonServers) {
        config.enabledMcpjsonServers = [];
      }
      if (!config.hooks || typeof config.hooks !== 'object') {
        config.hooks = {};
      }
      // Derive CTX workspace directory for the hook from extension settings.
      // Collection hint behavior is now driven by ctx_config.json, not hook env.
      let hookEnv;
      try {
        const resolvedTarget = resolveBridgeWorkspacePath ? resolveBridgeWorkspacePath() : undefined;
        if (resolvedTarget) {
          hookEnv = { CTX_WORKSPACE_DIR: resolvedTarget };
        }
      } catch (error) {
        // Best-effort only; if anything fails, fall back to no extra env
        hookEnv = undefined;
      }

      const hook = {
        type: 'command',
        command: commandPath,
      };
      if (hookEnv) {
        hook.env = hookEnv;
      }

      // Append or update our hook under UserPromptSubmit without clobbering existing hooks
      let userPromptHooks = config.hooks['UserPromptSubmit'];
      if (!Array.isArray(userPromptHooks)) {
        userPromptHooks = [];
      }

      const normalizeCommand = value => {
        if (!value) return '';
        const resolved = path.resolve(value);
        return resolved.replace(/context-engine\.context-engine-uploader-[0-9.]+/, 'context-engine.context-engine-uploader');
      };

      const normalizedNewCommand = normalizeCommand(commandPath);
      let updated = false;

      for (const entry of userPromptHooks) {
        if (!entry || !Array.isArray(entry.hooks)) {
          continue;
        }
        for (const existing of entry.hooks) {
          if (!existing || existing.type !== 'command') {
            continue;
          }
          const normalizedExisting = normalizeCommand(existing.command);
          if (normalizedExisting === normalizedNewCommand) {
            existing.command = commandPath;
            if (!existing.env) {
              existing.env = {};
            }
            if (hookEnv) {
              existing.env = { ...existing.env, ...hookEnv };
            }
            updated = true;
          }
        }
      }

      if (!updated) {
        userPromptHooks.push({ hooks: [hook] });
      }

      // Deduplicate any accidental double entries for the same command
      const seenCommands = new Set();
      for (const entry of userPromptHooks) {
        if (!entry || !Array.isArray(entry.hooks)) {
          continue;
        }
        entry.hooks = entry.hooks.filter(existing => {
          if (!existing || existing.type !== 'command') {
            return true;
          }
          const normalized = normalizeCommand(existing.command);
          if (seenCommands.has(normalized)) {
            return false;
          }
          seenCommands.add(normalized);
          return true;
        });
      }

      config.hooks['UserPromptSubmit'] = userPromptHooks.filter(entry => Array.isArray(entry.hooks) && entry.hooks.length);
      fs.writeFileSync(settingsPath, JSON.stringify(config, null, 2) + '\n', 'utf8');
      vscode.window.showInformationMessage('Context Engine Uploader: .claude/settings.local.json updated with Claude hook.');
      log(`Wrote Claude hook config at ${settingsPath}`);
      return true;
    } catch (error) {
      vscode.window.showErrorMessage('Context Engine Uploader: failed to write .claude/settings.local.json.');
      log(`Failed to write .claude/settings.local.json: ${error instanceof Error ? error.message : String(error)}`);
      return false;
    }
  }

  async function writeMcpConfig() {
    const settings = getEffectiveConfig();
    const claudeEnabled = settings.get('mcpClaudeEnabled', true);
    const windsurfEnabled = settings.get('mcpWindsurfEnabled', false);
    const claudeHookEnabled = settings.get('claudeHookEnabled', false);
    const isLinux = process.platform === 'linux';
    if (!claudeEnabled && !windsurfEnabled && !claudeHookEnabled) {
      vscode.window.showInformationMessage('Context Engine Uploader: MCP config writing is disabled in settings.');
      return;
    }
    const transportModeRaw = (settings.get('mcpTransportMode') || 'sse-remote');
    const transportMode = (typeof transportModeRaw === 'string' ? transportModeRaw.trim() : 'sse-remote') || 'sse-remote';
    const serverModeRaw = (settings.get('mcpServerMode') || 'bridge');
    const serverMode = (typeof serverModeRaw === 'string' ? serverModeRaw.trim() : 'bridge') || 'bridge';
    const needsHttpBridge = requiresHttpBridge(serverMode, transportMode);
    const bridgeWasRunning = !!(typeof getBridgeIsRunning === 'function' && getBridgeIsRunning());
    if (needsHttpBridge) {
      const ready = await ensureHttpBridgeReadyForConfigs();
      if (!ready) {
        vscode.window.showErrorMessage('Context Engine Uploader: HTTP MCP bridge failed to start; MCP config not updated.');
        return;
      }
      const bridgeNowRunning = !!(typeof getBridgeIsRunning === 'function' && getBridgeIsRunning());
      if (!bridgeWasRunning && bridgeNowRunning) {
        log('Context Engine Uploader: HTTP MCP bridge launching; delaying MCP config write until bridge signals ready.');
        return;
      }
    }
    const effectiveMode =
      serverMode === 'bridge'
        ? (transportMode === 'http' ? 'bridge-http' : 'bridge-stdio')
        : (transportMode === 'http' ? 'direct-http' : 'direct-sse');
    log(`Context Engine Uploader: MCP wiring mode=${effectiveMode} (serverMode=${serverMode}, transportMode=${transportMode}).`);
    if (effectiveMode === 'bridge-http') {
      const bridgeUrl = resolveBridgeHttpUrl();
      if (bridgeUrl) {
        log(`Context Engine Uploader: bridge HTTP endpoint ${bridgeUrl}`);
      }
    }

    let indexerUrl = (settings.get('mcpIndexerUrl') || 'http://localhost:8003/mcp').trim();
    let memoryUrl = (settings.get('mcpMemoryUrl') || 'http://localhost:8002/mcp').trim();
    if (serverMode === 'bridge') {
      indexerUrl = normalizeBridgeUrl(indexerUrl);
      memoryUrl = normalizeBridgeUrl(memoryUrl);
    }
    let wroteAny = false;
    let hookWrote = false;
    if (claudeEnabled) {
      const root = getWorkspaceFolderPath();
      if (!root) {
        vscode.window.showErrorMessage('Context Engine Uploader: open a folder before writing .mcp.json.');
      } else {
        const result = await writeClaudeMcpServers(root, indexerUrl, memoryUrl, transportMode, serverMode);
        wroteAny = wroteAny || result;
      }
    }
    if (windsurfEnabled) {
      const customPath = (settings.get('windsurfMcpPath') || '').trim();
      const windsPath = customPath || getDefaultWindsurfMcpPath();
      const workspaceHint = getWorkspaceFolderPath();
      const result = await writeWindsurfMcpServers(windsPath, indexerUrl, memoryUrl, transportMode, serverMode, workspaceHint);
      wroteAny = wroteAny || result;
    }
    if (claudeHookEnabled) {
      const root = getWorkspaceFolderPath();
      if (!root) {
        vscode.window.showErrorMessage('Context Engine Uploader: open a folder before writing Claude hook config.');
      } else if (!isLinux) {
        vscode.window.showWarningMessage('Context Engine Uploader: Claude hook auto-config is only wired for Linux/dev-remote at this time.');
      } else {
        const commandPath = getClaudeHookCommand();
        if (!commandPath) {
          vscode.window.showErrorMessage('Context Engine Uploader: embedded Claude hook script not found in extension; .claude/settings.local.json was not updated.');
          log('Claude hook config skipped because embedded ctx-hook-simple.sh could not be resolved.');
        } else {
          const result = await writeClaudeHookConfig(root, commandPath);
          hookWrote = hookWrote || result;
        }
      }
    }
    if (settings.get('scaffoldCtxConfig', true)) {
      try {
        await writeCtxConfig();
      } catch (error) {
        log(`CTX config auto-scaffolding failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    if (!wroteAny && !hookWrote) {
      // noop
    }
  }

  return {
    scheduleMcpConfigRefreshAfterBridge,
    writeMcpConfig,
    dispose,
  };
}

module.exports = {
  createMcpConfigManager,
};
