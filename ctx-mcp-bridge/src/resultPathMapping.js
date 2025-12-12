import process from "node:process";
import fs from "node:fs";
import path from "node:path";

function envTruthy(value, defaultVal = false) {
  try {
    if (value === undefined || value === null) {
      return defaultVal;
    }
    const s = String(value).trim().toLowerCase();
    if (!s) {
      return defaultVal;
    }
    return s === "1" || s === "true" || s === "yes" || s === "on";
  } catch {
    return defaultVal;
  }
}

function _posixToNative(rel) {
  try {
    if (!rel) {
      return "";
    }
    return String(rel).split("/").join(path.sep);
  } catch {
    return rel;
  }
}

function computeWorkspaceRelativePath(containerPath, hostPath) {
  try {
    const cont = typeof containerPath === "string" ? containerPath.trim() : "";
    if (cont.startsWith("/work/")) {
      const rest = cont.slice("/work/".length);
      const parts = rest.split("/").filter(Boolean);
      if (parts.length >= 2) {
        return parts.slice(1).join("/");
      }
      if (parts.length === 1) {
        return parts[0];
      }
    }
  } catch {
  }
  try {
    const hp = typeof hostPath === "string" ? hostPath.trim() : "";
    if (!hp) {
      return "";
    }
    // If we don't have a container path, at least try to return a basename.
    return path.posix.basename(hp.replace(/\\/g, "/"));
  } catch {
    return "";
  }
}

function remapHitPaths(hit, workspaceRoot) {
  if (!hit || typeof hit !== "object") {
    return hit;
  }
  const hostPath = typeof hit.host_path === "string" ? hit.host_path : "";
  const containerPath = typeof hit.container_path === "string" ? hit.container_path : "";
  const relPath = computeWorkspaceRelativePath(containerPath, hostPath);
  const out = { ...hit };
  if (relPath) {
    out.rel_path = relPath;
  }
  if (workspaceRoot && relPath) {
    try {
      const relNative = _posixToNative(relPath);
      const candidate = path.join(workspaceRoot, relNative);
      const diagnostics = envTruthy(process.env.CTXCE_BRIDGE_PATH_DIAGNOSTICS, false);
      const strictClientPath = envTruthy(process.env.CTXCE_BRIDGE_CLIENT_PATH_STRICT, false);
      if (strictClientPath) {
        out.client_path = candidate;
        if (diagnostics) {
          out.client_path_joined = candidate;
          out.client_path_source = "workspace_join";
        }
      } else {
        // Prefer a host_path that is within the current bridge workspace.
        // This keeps provenance (host_path) intact while providing a user-local
        // absolute path even when the bridge workspace is a parent directory.
        const hp = typeof hostPath === "string" ? hostPath : "";
        const hpNorm = hp ? hp.replace(/\\/g, path.sep) : "";
        if (
          hpNorm &&
          hpNorm.startsWith(workspaceRoot) &&
          (!fs.existsSync(candidate) || fs.existsSync(hpNorm))
        ) {
          out.client_path = hpNorm;
          if (diagnostics) {
            out.client_path_joined = candidate;
            out.client_path_source = "host_path";
          }
        } else {
          out.client_path = candidate;
          if (diagnostics) {
            out.client_path_joined = candidate;
            out.client_path_source = "workspace_join";
          }
        }
      }
    } catch {
      // ignore
    }
  }
  const overridePath = envTruthy(process.env.CTXCE_BRIDGE_OVERRIDE_PATH, true);
  if (overridePath && relPath) {
    out.path = relPath;
  }
  return out;
}

function remapStringPath(p) {
  try {
    const s = typeof p === "string" ? p : "";
    if (!s) {
      return p;
    }
    if (s.startsWith("/work/")) {
      const rest = s.slice("/work/".length);
      const parts = rest.split("/").filter(Boolean);
      if (parts.length >= 2) {
        const rel = parts.slice(1).join("/");
        const override = envTruthy(process.env.CTXCE_BRIDGE_OVERRIDE_PATH, true);
        if (override) {
          return rel;
        }
        return p;
      }
    }
    return p;
  } catch {
    return p;
  }
}

function maybeParseToolJson(result) {
  try {
    if (
      result &&
      typeof result === "object" &&
      result.structuredContent &&
      typeof result.structuredContent === "object"
    ) {
      return { mode: "structured", value: result.structuredContent };
    }
  } catch {
  }
  try {
    const content = result && result.content;
    if (!Array.isArray(content)) {
      return null;
    }
    const first = content.find(
      (c) => c && c.type === "text" && typeof c.text === "string",
    );
    if (!first) {
      return null;
    }
    const txt = String(first.text || "").trim();
    if (!txt || !(txt.startsWith("{") || txt.startsWith("["))) {
      return null;
    }
    return { mode: "text", value: JSON.parse(txt) };
  } catch {
    return null;
  }
}

function applyPathMappingToPayload(payload, workspaceRoot) {
  if (!payload || typeof payload !== "object") {
    return payload;
  }
  const out = Array.isArray(payload) ? payload.slice() : { ...payload };

  const mapHitsArray = (arr) => {
    if (!Array.isArray(arr)) {
      return arr;
    }
    return arr.map((h) => remapHitPaths(h, workspaceRoot));
  };

  // Common result shapes across tools
  if (Array.isArray(out.results)) {
    out.results = mapHitsArray(out.results);
  }
  if (Array.isArray(out.citations)) {
    out.citations = mapHitsArray(out.citations);
  }
  if (Array.isArray(out.related_paths)) {
    out.related_paths = out.related_paths.map((p) => remapStringPath(p));
  }

  // context_search: {results:[{source:"code"|"memory", ...}]}
  if (Array.isArray(out.results)) {
    out.results = out.results.map((r) => {
      if (!r || typeof r !== "object") {
        return r;
      }
      // Only code results have path-like fields
      return remapHitPaths(r, workspaceRoot);
    });
  }

  // Some tools nest under {result:{...}}
  if (out.result && typeof out.result === "object") {
    out.result = applyPathMappingToPayload(out.result, workspaceRoot);
  }

  return out;
}

export function maybeRemapToolResult(name, result, workspaceRoot) {
  try {
    if (!name || !result || !workspaceRoot) {
      return result;
    }
    const enabled = envTruthy(process.env.CTXCE_BRIDGE_MAP_PATHS, true);
    if (!enabled) {
      return result;
    }
    const lower = String(name).toLowerCase();
    const shouldMap = (
      lower === "repo_search" ||
      lower === "context_search" ||
      lower === "context_answer" ||
      lower.endsWith("search_tests_for") ||
      lower.endsWith("search_config_for") ||
      lower.endsWith("search_callers_for") ||
      lower.endsWith("search_importers_for")
    );
    if (!shouldMap) {
      return result;
    }

    const parsed = maybeParseToolJson(result);
    if (!parsed) {
      return result;
    }

    const mapped = applyPathMappingToPayload(parsed.value, workspaceRoot);
    if (parsed.mode === "structured") {
      return { ...result, structuredContent: mapped };
    }

    // Replace text payload for clients that only read `content[].text`
    try {
      const content = Array.isArray(result.content) ? result.content.slice() : [];
      const idx = content.findIndex(
        (c) => c && c.type === "text" && typeof c.text === "string",
      );
      if (idx >= 0) {
        content[idx] = { ...content[idx], text: JSON.stringify(mapped) };
        return { ...result, content };
      }
    } catch {
      // ignore
    }
    return result;
  } catch {
    return result;
  }
}
