function registerPromptPlusCommands(deps) {
  const vscode = deps.vscode;
  const fs = deps.fs;
  const path = deps.path;
  const log = deps.log;

  const getEffectiveConfig = deps.getEffectiveConfig;
  const resolveTargetPathFromConfig = deps.resolveTargetPathFromConfig;
  const writeCtxConfig = deps.writeCtxConfig;
  const getPromptPlusManager = deps.getPromptPlusManager;
  const getSidebarApi = deps.getSidebarApi;

  const promptEnhanceDisposable = vscode.commands.registerCommand('contextEngineUploader.promptEnhance', () => {
    try {
      const promptPlusManager = typeof getPromptPlusManager === 'function' ? getPromptPlusManager() : undefined;
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

  const promptEnhanceCopyDisposable = vscode.commands.registerCommand('contextEngineUploader.promptEnhanceCopy', () => {
    try {
      const promptPlusManager = typeof getPromptPlusManager === 'function' ? getPromptPlusManager() : undefined;
      if (promptPlusManager && typeof promptPlusManager.enhancePromptWithUnicornCopy === 'function') {
        promptPlusManager.enhancePromptWithUnicornCopy().catch(error => {
          log(`Prompt+ copy failed: ${error instanceof Error ? error.message : String(error)}`);
          vscode.window.showErrorMessage('Prompt+ copy failed. See Context Engine Upload output.');
        });
      } else {
        vscode.window.showErrorMessage('Context Engine Uploader: Prompt+ is unavailable (extension failed to initialize Prompt+ manager). See output for details.');
      }
    } catch (error) {
      log(`Prompt+ copy failed: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Prompt+ copy failed. See Context Engine Upload output.');
    }
  });

  const promptEnhanceOpenDisposable = vscode.commands.registerCommand('contextEngineUploader.promptEnhanceOpen', () => {
    try {
      const promptPlusManager = typeof getPromptPlusManager === 'function' ? getPromptPlusManager() : undefined;
      if (promptPlusManager && typeof promptPlusManager.enhancePromptWithUnicornOpen === 'function') {
        promptPlusManager.enhancePromptWithUnicornOpen().catch(error => {
          log(`Prompt+ open failed: ${error instanceof Error ? error.message : String(error)}`);
          vscode.window.showErrorMessage('Prompt+ open failed. See Context Engine Upload output.');
        });
      } else {
        vscode.window.showErrorMessage('Context Engine Uploader: Prompt+ is unavailable (extension failed to initialize Prompt+ manager). See output for details.');
      }
    } catch (error) {
      log(`Prompt+ open failed: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Prompt+ open failed. See Context Engine Upload output.');
    }
  });

  const promptDefaultModeDisposable = vscode.commands.registerCommand('contextEngineUploader.setCtxDefaultMode', async () => {
    const cfg = getEffectiveConfig();
    const resolved = resolveTargetPathFromConfig(cfg);
    const targetPath = resolved && resolved.path ? String(resolved.path).trim() : '';
    if (!targetPath) {
      vscode.window.showErrorMessage('Context Engine Uploader: targetPath is not configured (cannot locate ctx_config.json).');
      return;
    }

    const ctxConfigPath = path.join(targetPath, 'ctx_config.json');
    const items = [
      {
        label: 'default',
        description: 'Fast single-pass rewrite (ctx.py default)',
        value: 'default',
      },
      {
        label: 'unicorn',
        description: 'Best quality multi-pass rewrite (ctx.py --unicorn)',
        value: 'unicorn',
      },
    ];

    let existing = {};
    if (fs.existsSync(ctxConfigPath)) {
      try {
        const raw = fs.readFileSync(ctxConfigPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          existing = parsed;
        }
      } catch (error) {
        log(`Failed to parse ctx_config.json at ${ctxConfigPath}: ${error instanceof Error ? error.message : String(error)}`);
        vscode.window.showErrorMessage('Context Engine Uploader: failed to parse ctx_config.json. See output for details.');
        return;
      }
    } else {
      const pickedInit = await vscode.window.showInformationMessage(
        'ctx_config.json not found. Create it now?',
        'Create',
        'Cancel'
      );
      if (pickedInit !== 'Create') {
        return;
      }
      await writeCtxConfig();
      if (!fs.existsSync(ctxConfigPath)) {
        vscode.window.showErrorMessage('Context Engine Uploader: ctx_config.json is still missing after scaffolding.');
        return;
      }
      try {
        const raw = fs.readFileSync(ctxConfigPath, 'utf8');
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') {
          existing = parsed;
        }
      } catch (error) {
        log(`Failed to parse ctx_config.json after scaffolding at ${ctxConfigPath}: ${error instanceof Error ? error.message : String(error)}`);
        vscode.window.showErrorMessage('Context Engine Uploader: failed to parse ctx_config.json after scaffolding. See output for details.');
        return;
      }
    }

    const currentMode = (typeof existing.default_mode === 'string' ? existing.default_mode.trim() : '') || 'default';
    const picked = await vscode.window.showQuickPick(items, {
      placeHolder: `Prompt+ default mode (currently: ${currentMode})`,
      matchOnDescription: true,
      ignoreFocusOut: true,
    });
    if (!picked) {
      return;
    }

    if (picked.value === currentMode) {
      vscode.window.showInformationMessage(`Prompt+ default mode already set to '${currentMode}'.`);
      return;
    }

    try {
      existing.default_mode = picked.value;
      fs.writeFileSync(ctxConfigPath, JSON.stringify(existing, null, 2) + '\n', 'utf8');
      vscode.window.showInformationMessage(`Prompt+ default mode set to '${picked.value}'.`);
      try {
        const sidebarApi = typeof getSidebarApi === 'function' ? getSidebarApi() : undefined;
        if (sidebarApi && typeof sidebarApi.refresh === 'function') {
          sidebarApi.refresh();
        }
      } catch (_) {
      }
    } catch (error) {
      log(`Failed to write ctx_config.json default_mode at ${ctxConfigPath}: ${error instanceof Error ? error.message : String(error)}`);
      vscode.window.showErrorMessage('Context Engine Uploader: failed to update ctx_config.json. See output for details.');
    }
  });

  return [
    promptEnhanceDisposable,
    promptEnhanceCopyDisposable,
    promptEnhanceOpenDisposable,
    promptDefaultModeDisposable,
  ];
}

module.exports = {
  registerPromptPlusCommands,
};
