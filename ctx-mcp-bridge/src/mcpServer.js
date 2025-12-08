function debugLog(message) {
  try {
    const text = typeof message === "string" ? message : String(message);
    console.error(text);
    const dest = process.env.CTXCE_DEBUG_LOG;
    if (dest) {
      fs.appendFileSync(dest, `${new Date().toISOString()} ${text}\n`, "utf8");
    }
  } catch {
  }
}

async function sendSessionDefaults(client, payload, label) {
  if (!client) {
    return;
  }
  try {
    await client.callTool({
      name: "set_session_defaults",
      arguments: payload,
    });
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error(`[ctxce] Failed to call set_session_defaults on ${label}:`, err);
  }
}
function dedupeTools(tools) {
  const seen = new Set();
  const out = [];
  for (const tool of tools) {
    const key = (tool && typeof tool.name === "string" && tool.name) || "";
    if (!key || seen.has(key)) {
      if (key === "" || key !== "set_session_defaults") {
        continue;
      }
      if (seen.has(key)) {
        continue;
      }
    }
    seen.add(key);
    out.push(tool);
  }
  return out;
}

async function listMemoryTools(client) {
  if (!client) {
    return [];
  }
  try {
    const remote = await withTimeout(
      client.listTools(),
      5000,
      "memory tools/list",
    );
    return Array.isArray(remote?.tools) ? remote.tools.slice() : [];
  } catch (err) {
    debugLog("[ctxce] Error calling memory tools/list: " + String(err));
    return [];
  }
}

function withTimeout(promise, ms, label) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      const errorMessage =
        label != null
          ? `[ctxce] Timeout after ${ms}ms in ${label}`
          : `[ctxce] Timeout after ${ms}ms`;
      reject(new Error(errorMessage));
    }, ms);
    promise
      .then((value) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        resolve(value);
      })
      .catch((err) => {
        if (settled) {
          return;
        }
        settled = true;
        clearTimeout(timer);
        reject(err);
      });
  });
}

function selectClientForTool(name, indexerClient, memoryClient) {
  if (!name) {
    return indexerClient;
  }
  const lowered = name.toLowerCase();
  if (memoryClient && (lowered.startsWith("memory.") || lowered.startsWith("mcp_memory_") || lowered.includes("memory"))) {
    return memoryClient;
  }
  return indexerClient;
}
// MCP stdio server implemented using the official MCP TypeScript SDK.
// Acts as a low-level proxy for tools, forwarding tools/list and tools/call
// to the remote qdrant-indexer MCP server while adding a local `ping` tool.

import process from "node:process";
import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

export async function runMcpServer(options) {
  const workspace = options.workspace || process.cwd();
  const indexerUrl = options.indexerUrl;
  const memoryUrl = options.memoryUrl;

  const config = loadConfig(workspace);
  const defaultCollection =
    config && typeof config.default_collection === "string"
      ? config.default_collection
      : null;
  const defaultMode =
    config && typeof config.default_mode === "string" ? config.default_mode : null;
  const defaultUnder =
    config && typeof config.default_under === "string" ? config.default_under : null;

  debugLog(
    `[ctxce] MCP low-level stdio bridge starting: workspace=${workspace}, indexerUrl=${indexerUrl}`,
  );

  if (defaultCollection) {
    // eslint-disable-next-line no-console
    console.error(
      `[ctxce] Using default collection from ctx_config.json: ${defaultCollection}`,
    );
  }

  // High-level MCP client for the remote HTTP /mcp indexer
  const indexerTransport = new StreamableHTTPClientTransport(indexerUrl);
  const indexerClient = new Client(
    {
      name: "ctx-context-engine-bridge-http-client",
      version: "0.0.1",
    },
    {
      capabilities: {
        tools: {},
        resources: {},
        prompts: {},
      },
    },
  );

  try {
    await indexerClient.connect(indexerTransport);
  } catch (err) {
    debugLog("[ctxce] Failed to connect MCP HTTP client to indexer: " + String(err));
    throw err;
  }

  let memoryClient = null;
  if (memoryUrl) {
    try {
      const memoryTransport = new StreamableHTTPClientTransport(memoryUrl);
      memoryClient = new Client(
        {
          name: "ctx-context-engine-bridge-memory-client",
          version: "0.0.1",
        },
        {
          capabilities: {
            tools: {},
            resources: {},
            prompts: {},
          },
        },
      );
      await memoryClient.connect(memoryTransport);
      debugLog(`[ctxce] Connected memory MCP client: ${memoryUrl}`);
    } catch (err) {
      debugLog("[ctxce] Failed to connect memory MCP client: " + String(err));
      memoryClient = null;
    }
  }

  // Derive a simple session identifier for this bridge process. In the
  // future this can be made user-aware (e.g. from auth), but for now we
  // keep it deterministic per workspace to help the indexer reuse
  // session-scoped defaults.
  const sessionId =
    process.env.CTXCE_SESSION_ID || `ctxce-${Buffer.from(workspace).toString("hex").slice(0, 24)}`;

  // Best-effort: inform the indexer of default collection and session.
  // If this fails we still proceed, falling back to per-call injection.
  const defaultsPayload = { session: sessionId };
  if (defaultCollection) {
    defaultsPayload.collection = defaultCollection;
  }
  if (defaultMode) {
    defaultsPayload.mode = defaultMode;
  }
  if (defaultUnder) {
    defaultsPayload.under = defaultUnder;
  }

  if (Object.keys(defaultsPayload).length > 1) {
    await sendSessionDefaults(indexerClient, defaultsPayload, "indexer");
    if (memoryClient) {
      await sendSessionDefaults(memoryClient, defaultsPayload, "memory");
    }
  }

  const server = new Server( // TODO: marked as depreciated
    {
      name: "ctx-context-engine-bridge",
      version: "0.0.1",
    },
    {
      capabilities: {
        tools: {},
      },
    },
  );

  // tools/list → fetch tools from remote indexer
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    let remote;
    try {
      debugLog("[ctxce] tools/list: fetching tools from indexer");
      remote = await withTimeout(
        indexerClient.listTools(),
        10000,
        "indexer tools/list",
      );
    } catch (err) {
      debugLog("[ctxce] Error calling remote tools/list: " + String(err));
      const memoryToolsFallback = await listMemoryTools(memoryClient);
      const toolsFallback = dedupeTools([...memoryToolsFallback]);
      return { tools: toolsFallback };
    }

    try {
      const toolNames =
        remote && Array.isArray(remote.tools)
          ? remote.tools.map((t) => (t && typeof t.name === "string" ? t.name : "<unnamed>"))
          : [];
      debugLog("[ctxce] tools/list remote result tools: " + JSON.stringify(toolNames));
    } catch (err) {
      debugLog("[ctxce] tools/list remote result: <unserializable> " + String(err));
    }

    const indexerTools = Array.isArray(remote?.tools) ? remote.tools.slice() : [];
    const memoryTools = await listMemoryTools(memoryClient);
    const tools = dedupeTools([...indexerTools, ...memoryTools]);
    debugLog(`[ctxce] tools/list: returning ${tools.length} tools`);
    return { tools };
  });

  // tools/call → proxied to indexer or memory server
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const params = request.params || {};
    const name = params.name;
    let args = params.arguments;

    debugLog(`[ctxce] tools/call: ${name || "<no-name>"}`);

    // Attach session id so the target server can apply per-session defaults.
    if (sessionId && (args === undefined || args === null || typeof args === "object")) {
      const obj = args && typeof args === "object" ? { ...args } : {};
      if (!Object.prototype.hasOwnProperty.call(obj, "session")) {
        obj.session = sessionId;
      }
      args = obj;
    }

    if (name === "set_session_defaults") {
      const indexerResult = await indexerClient.callTool({ name, arguments: args });
      if (memoryClient) {
        try {
          await memoryClient.callTool({ name, arguments: args });
        } catch (err) {
          debugLog("[ctxce] Memory set_session_defaults failed: " + String(err));
        }
      }
      return indexerResult;
    }

    const targetClient = selectClientForTool(name, indexerClient, memoryClient);
    if (!targetClient) {
      throw new Error(`Tool ${name} not available on any configured MCP server`);
    }

    const result = await targetClient.callTool({
      name,
      arguments: args,
    });
    return result;
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);
}

function loadConfig(startDir) {
  try {
    let dir = startDir;
    for (let i = 0; i < 5; i += 1) {
      const cfgPath = path.join(dir, "ctx_config.json");
      if (fs.existsSync(cfgPath)) {
        try {
          const raw = fs.readFileSync(cfgPath, "utf8");
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object") {
            return parsed;
          }
        } catch (err) {
          // eslint-disable-next-line no-console
          console.error("[ctxce] Failed to parse ctx_config.json:", err);
          return null;
        }
      }
      const parent = path.dirname(dir);
      if (!parent || parent === dir) {
        break;
      }
      dir = parent;
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("[ctxce] Error while loading ctx_config.json:", err);
  }
  return null;
}

function detectGitBranch(workspace) {
  try {
    const out = execSync("git rev-parse --abbrev-ref HEAD", {
      cwd: workspace,
      stdio: ["ignore", "pipe", "ignore"],
    });
    const name = out.toString("utf8").trim();
    return name || null;
  } catch {
    return null;
  }
}

