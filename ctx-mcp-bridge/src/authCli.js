import process from "node:process";
import { loadAuthEntry, saveAuthEntry, deleteAuthEntry } from "./authConfig.js";

function parseAuthArgs(args) {
  let backendUrl = process.env.CTXCE_AUTH_BACKEND_URL || "";
  let token = process.env.CTXCE_AUTH_TOKEN || "";
  let username = process.env.CTXCE_AUTH_USERNAME || "";
  let password = process.env.CTXCE_AUTH_PASSWORD || "";
  let outputJson = false;
  for (let i = 0; i < args.length; i += 1) {
    const a = args[i];
    if ((a === "--backend-url" || a === "--auth-url") && i + 1 < args.length) {
      backendUrl = args[i + 1];
      i += 1;
      continue;
    }
    if ((a === "--token" || a === "--api-key") && i + 1 < args.length) {
      token = args[i + 1];
      i += 1;
      continue;
    }
    if ((a === "--username" || a === "--user") && i + 1 < args.length) {
      username = args[i + 1];
      i += 1;
      continue;
    }
    if ((a === "--password" || a === "--pass") && i + 1 < args.length) {
      password = args[i + 1];
      i += 1;
      continue;
    }
    if (a === "--json" || a === "-j") {
      outputJson = true;
      continue;
    }
  }
  return { backendUrl, token, username, password, outputJson };
}

function getBackendUrl(backendUrl) {
  return (backendUrl || process.env.CTXCE_AUTH_BACKEND_URL || "").trim();
}

function requireBackendUrl(backendUrl) {
  const url = getBackendUrl(backendUrl);
  if (!url) {
    console.error("[ctxce] Auth backend URL not configured. Set CTXCE_AUTH_BACKEND_URL or use --backend-url.");
    process.exit(1);
  }
  return url;
}

function outputJsonStatus(url, state, entry, rawExpires) {
  const expiresAt = typeof rawExpires === "number"
    ? rawExpires
    : entry && typeof entry.expiresAt === "number"
      ? entry.expiresAt
      : null;
  console.log(JSON.stringify({
    backendUrl: url,
    state,
    sessionId: entry && entry.sessionId ? entry.sessionId : null,
    userId: entry && entry.userId ? entry.userId : null,
    expiresAt,
  }));
}

async function doLogin(args) {
  const { backendUrl, token, username, password } = parseAuthArgs(args);
  const url = requireBackendUrl(backendUrl);
  const trimmedUser = (username || "").trim();
  const usePassword = trimmedUser && (password || "").length > 0;

  let body;
  let target;
  if (usePassword) {
    body = {
      username: trimmedUser,
      password,
      workspace: process.cwd(),
    };
    target = url.replace(/\/+$/, "") + "/auth/login/password";
  } else {
    body = {
      client: "ctxce",
      workspace: process.cwd(),
    };
    if (token) {
      body.token = token;
    }
    target = url.replace(/\/+$/, "") + "/auth/login";
  }
  let resp;
  try {
    resp = await fetch(target, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    console.error("[ctxce] Auth login request failed:", String(err));
    process.exit(1);
  }
  if (!resp || !resp.ok) {
    console.error("[ctxce] Auth login failed with status", resp ? resp.status : "<no-response>");
    process.exit(1);
  }
  let data;
  try {
    data = await resp.json();
  } catch (err) {
    data = {};
  }
  const sessionId = data.session_id || data.sessionId || null;
  const userId = data.user_id || data.userId || null;
  const expiresAt = data.expires_at || data.expiresAt || null;
  if (!sessionId) {
    console.error("[ctxce] Auth login response missing session id.");
    process.exit(1);
  }
  saveAuthEntry(url, { sessionId, userId, expiresAt });
  console.error("[ctxce] Auth login successful for", url);
}

async function doStatus(args) {
  const { backendUrl, outputJson } = parseAuthArgs(args);
  const url = getBackendUrl(backendUrl);
  if (!url) {
    if (outputJson) {
      outputJsonStatus("", "missing_backend", null, null);
      process.exit(1);
    }
    console.error("[ctxce] Auth backend URL not configured. Set CTXCE_AUTH_BACKEND_URL or use --backend-url.");
    process.exit(1);
  }
  let entry;
  try {
    entry = loadAuthEntry(url);
  } catch (err) {
    entry = null;
  }
  const nowSecs = Math.floor(Date.now() / 1000);
  const rawExpires = entry && typeof entry.expiresAt === "number" ? entry.expiresAt : null;
  const hasSession = !!(entry && typeof entry.sessionId === "string" && entry.sessionId);
  const expired = !!(rawExpires && rawExpires > 0 && rawExpires < nowSecs);

  if (!entry || !hasSession) {
    if (outputJson) {
      outputJsonStatus(url, "missing", null, rawExpires);
      process.exit(1);
    }
    console.error("[ctxce] Not logged in for", url);
    process.exit(1);
  }

  if (expired) {
    if (outputJson) {
      outputJsonStatus(url, "expired", entry, rawExpires);
      process.exit(2);
    }
    console.error("[ctxce] Stored auth session appears expired for", url);
    if (rawExpires) {
      console.error("[ctxce] Session expired at", rawExpires);
    }
    process.exit(2);
  }

  if (outputJson) {
    outputJsonStatus(url, "ok", entry, rawExpires);
    return;
  }
  console.error("[ctxce] Logged in to", url, "as", entry.userId || "<unknown>");
  if (rawExpires) {
    console.error("[ctxce] Session expires at", rawExpires);
  }
}

async function doLogout(args) {
  const { backendUrl } = parseAuthArgs(args);
  const url = requireBackendUrl(backendUrl);
  const entry = loadAuthEntry(url);
  if (!entry) {
    console.error("[ctxce] No stored auth session for", url);
    return;
  }
  deleteAuthEntry(url);
  console.error("[ctxce] Logged out from", url);
}

export async function runAuthCommand(subcommand, args) {
  const sub = (subcommand || "").toLowerCase();
  if (sub === "login") {
    await doLogin(args || []);
    return;
  }
  if (sub === "status") {
    await doStatus(args || []);
    return;
  }
  if (sub === "logout") {
    await doLogout(args || []);
    return;
  }
  console.error("Usage: ctxce auth <login|status|logout> [--backend-url <url>] [--token <token>]");
  process.exit(1);
}
