#!/usr/bin/env python3
"""
Minimal MCP Router: intent → tool selection → optional execution via RMCP/HTTP.

- Decides which MCP tool to call (qdrant-indexer server) based on the user query.
- Optionally executes the tool over HTTP JSON-RPC (streamable-http) without extra deps.
- Defaults are conservative and avoid sending null/empty params.

Usage:
  python scripts/mcp_router.py --plan "How do I ...?"
  python scripts/mcp_router.py --run  "What is hybrid search?"

Env:
  MCP_INDEXER_HTTP_URL  (default: http://localhost:8003/mcp)
  ROUTER_MAX_TOKENS     (default: "" -> omit, let server use DECODER_MAX_TOKENS)
  ROUTER_SEARCH_LIMIT   (default: 8)
  ROUTER_INCLUDE_SNIPPET(default: 1 -> include_snippet: True)

Notes:
- We only pass arguments that have concrete values; None/empty omitted to satisfy the server’s param rules.
- context_answer will respect DECODER_MAX_TOKENS from the server env; you can override via ROUTER_MAX_TOKENS.
"""
from __future__ import annotations

import json
import os
import re
import sys
import argparse
from typing import Dict, Any, List, Tuple
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urlparse

import time

HTTP_URL_INDEXER = os.environ.get("MCP_INDEXER_HTTP_URL", "http://localhost:8003/mcp").rstrip("/")
HTTP_URL_MEMORY = os.environ.get("MCP_MEMORY_HTTP_URL", "http://localhost:8002/mcp").rstrip("/")
DEFAULT_HTTP_URL = HTTP_URL_INDEXER

_LAST_INTENT_DEBUG: Dict[str, Any] = {}

# -----------------------------
# Intent classification
# -----------------------------
INTENT_ANSWER = "answer"         # use context_answer
INTENT_SEARCH = "search"         # use repo_search
INTENT_SEARCH_TESTS = "search_tests"
INTENT_SEARCH_CONFIG = "search_config"
INTENT_SEARCH_CALLERS = "search_callers"
INTENT_SEARCH_IMPORTERS = "search_importers"
INTENT_MEMORY_STORE = "memory_store"   # use memory.store
INTENT_MEMORY_FIND = "memory_find"     # use memory.find

INTENT_INDEX = "index"           # use qdrant_index_root
INTENT_PRUNE = "prune"           # use qdrant_prune
INTENT_STATUS = "status"         # use qdrant_status
INTENT_LIST = "list"             # use qdrant_list


def _classify_intent_rules(q: str) -> str | None:
    s = q.lower()
    # Admin / maintenance first
    if any(w in s for w in ["reindex", "reset", "recreate", "index now", "fresh index"]):
        return INTENT_INDEX
    if any(w in s for w in ["prune", "pruning", "cleanup", "clean up"]):
        return INTENT_PRUNE
    if any(w in s for w in ["status", "health", "points", "stats"]):
        return INTENT_STATUS
    if any(w in s for w in ["list collections", "collections", "list qdrant"]):
        return INTENT_LIST

    # Intent wrappers
    if any(w in s for w in ["tests", "pytest", "unit test", "test file", "where are tests"]):
        return INTENT_SEARCH_TESTS
    # Memory intents
    if any(w in s for w in [
        "remember this", "save memory", "store memory", "remember that", "save preference", "remember preference"
    ]):
        return INTENT_MEMORY_STORE
    if any(w in s for w in [
        "find memory", "recall", "retrieve memory", "memory search", "what did we save"
    ]):
        return INTENT_MEMORY_FIND

    if any(w in s for w in ["config", "yaml", "toml", "ini", "settings file", "configuration"]):
        return INTENT_SEARCH_CONFIG
    if any(w in s for w in ["who calls", "callers", "used by", "usage sites", "references this function"]):
        return INTENT_SEARCH_CALLERS
    if any(w in s for w in ["importers", "who imports", "imports this", "importing modules"]):
        return INTENT_SEARCH_IMPORTERS

    # Q&A-like prompts
    if re.match(r"^(what|how|why|explain|describe|summarize)(\b|\s)", s):
        return INTENT_ANSWER
    if any(w in s for w in ["recap", "design doc", "architecture", "adr", "retrospective", "postmortem", "summary of", "summarize the design"]):
        return INTENT_ANSWER
    return None


def _intent_prototypes() -> Dict[str, List[str]]:
    return {
        INTENT_ANSWER: [
            "explain, describe, summarize, recap, design, architecture, ADR, why/how",
            "summarize design decisions and architecture rationale",
        ],
        INTENT_SEARCH: [
            "find code references, search repository, locate files",
            "code search in repo, general lookup",
        ],
        INTENT_MEMORY_STORE: [
            "remember this, save preference, store memory",
        ],
        INTENT_MEMORY_FIND: [
            "what did we save, recall saved notes, retrieve memory",
        ],
        INTENT_SEARCH_TESTS: [
            "find unit tests, test files, pytest",
        ],
        INTENT_SEARCH_CONFIG: [
            "config files, configuration changes, yaml toml ini settings",
        ],
        INTENT_SEARCH_CALLERS: [
            "who calls this function, callers, usage sites",
        ],
        INTENT_SEARCH_IMPORTERS: [
            "who imports this module, importers, importing modules",
        ],
    }


def _classify_intent_ml(q: str) -> str:
    global _LAST_INTENT_DEBUG
    protos = _intent_prototypes()
    labels = list(protos.keys())
    texts = [q] + ["\n".join(protos[l]) for l in labels]
    vecs = _embed_texts(texts)
    if not vecs or len(vecs) < len(texts):
        _LAST_INTENT_DEBUG = {
            "strategy": "ml",
            "intent": INTENT_SEARCH,
            "confidence": 0.0,
            "query": q,
            "top_candidate": INTENT_SEARCH,
            "top_score": 0.0,
            "threshold": 0.25,
            "candidates": [],
            "reason": "embed_failed",
        }
        return INTENT_SEARCH
    qv = vecs[0]
    sims = []
    for i, lab in enumerate(labels):
        sims.append((lab, _cosine(qv, vecs[1 + i])))
    sims.sort(key=lambda x: x[1], reverse=True)
    top, score = sims[0]
    # light threshold: if nothing is clearly better, default to search
    picked = top if score >= 0.25 else INTENT_SEARCH
    _LAST_INTENT_DEBUG = {
        "strategy": "ml",
        "intent": picked,
        "confidence": float(score),
        "query": q,
        "top_candidate": top,
        "top_score": float(score),
        "threshold": 0.25,
        "candidates": [(name, float(val)) for name, val in sims[:5]],
        "fallback": picked == INTENT_SEARCH and top != INTENT_SEARCH,
    }
    return picked


def classify_intent(q: str) -> str:
    global _LAST_INTENT_DEBUG
    # Prefer high-precision rules; fall back to embedding classifier
    ruled = _classify_intent_rules(q)
    if ruled is not None:
        _LAST_INTENT_DEBUG = {
            "strategy": "rules",
            "intent": ruled,
            "confidence": 1.0,
            "query": q,
        }
        return ruled
    return _classify_intent_ml(q)


# -----------------------------
# Memory helper
# -----------------------------
_MEMORY_TRIGGER_RE = re.compile(
    r"^(?:remember(?:\s+(?:this|that|me|to))?|save\s+memory|store\s+memory)\s*[:,\-]?\s*",
    re.IGNORECASE,
)
_MEMORY_INTENT_SPLIT_RE = re.compile(
    r"\b(?:then|and|also)\s+(?:reindex|index|recreate|prune|clean\s+up)\b",
    re.IGNORECASE,
)
_MEMORY_META_KEYS = {"priority", "tag", "tags", "topic", "category", "owner"}


def _parse_memory_store_payload(q: str) -> Tuple[str, Dict[str, Any]]:
    raw = str(q or "").strip()
    if not raw:
        return "", {}
    cleaned = _MEMORY_TRIGGER_RE.sub("", raw, count=1).lstrip()
    meta: Dict[str, Any] = {}

    def _assign_meta(key: str, value: str) -> None:
        k = key.lower()
        v = value.strip().strip(" \t\r\n,;.")
        if not v:
            return
        if k in {"tag", "tags"}:
            tags = [t.strip() for t in re.split(r"[,\s/]+", v) if t.strip()]
            if tags:
                meta["tags"] = tags
        else:
            meta[k] = v

    if cleaned.startswith("["):
        m = re.match(r"\[([^\]]+)\]\s*(.*)", cleaned, flags=re.S)
        if m:
            meta_block = m.group(1)
            cleaned = m.group(2)
            for key, val in re.findall(r"(\w+)\s*=\s*([^\s,;]+(?:,[^\s,;]+)*)", meta_block):
                if key.strip().lower() in _MEMORY_META_KEYS:
                    _assign_meta(key, val)

    while True:
        m = re.match(
            r"^(?P<key>(?:priority|tag|tags|topic|category|owner))\s*=\s*(?P<val>[^\s;:]+)\s*[,;:]?\s*(?P<rest>.*)$",
            cleaned,
            flags=re.IGNORECASE | re.S,
        )
        if not m:
            break
        _assign_meta(m.group("key"), m.group("val"))
        cleaned = m.group("rest")

    cleaned = cleaned.lstrip(":- ").lstrip()

    split = _MEMORY_INTENT_SPLIT_RE.search(cleaned)
    if split:
        cleaned = cleaned[: split.start()].rstrip(" ,;.")

    cleaned = cleaned.strip().strip('"').strip()
    if not cleaned:
        cleaned = raw
    return cleaned, meta


# -----------------------------
# Tool planning
# -----------------------------

def build_plan(q: str) -> List[Tuple[str, Dict[str, Any]]]:
    intent = classify_intent(q)
    include_snippet = str(os.environ.get("ROUTER_INCLUDE_SNIPPET", "1")).lower() in {"1", "true", "yes", "on"}
    search_limit = int(os.environ.get("ROUTER_SEARCH_LIMIT", "8") or 8)
    max_tokens_env = os.environ.get("ROUTER_MAX_TOKENS", "").strip()

    def _reuse_last_filters(args: Dict[str, Any]) -> None:
        """Optionally hydrate `args` with cached filters when user asks for reuse."""
        try:
            if _looks_like_same_filters(q):
                sp = _load_scratchpad()
                lf = sp.get("last_filters") if isinstance(sp, dict) else None
                if isinstance(lf, dict):
                    for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
                        if k not in args and lf.get(k) not in (None, ""):
                            args[k] = lf.get(k)
        except Exception:
            pass

    # Repeat/redo handling: reuse last plan if asked
    try:
        if _looks_like_repeat(q):
            sp = _load_scratchpad()
            lp = sp.get("last_plan")
            if isinstance(lp, list) and lp:
                # Normalize to list of (tool, args) tuples
                norm: list[tuple] = []
                for it in lp:
                    if isinstance(it, (list, tuple)) and len(it) == 2:
                        norm.append((it[0], it[1]))
                if norm:
                    return norm
    except Exception:
        pass

    # Multi-intent: memory store + reindex in a single prompt
    lowq = q.lower()
    if any(w in lowq for w in ["remember this", "store memory", "save memory", "remember that"]) and any(w in lowq for w in ["reindex", "index now", "recreate", "fresh index"]):
        idx_args: Dict[str, Any] = {}
        if any(w in lowq for w in ["recreate", "fresh", "from scratch", "fresh index"]):
            idx_args["recreate"] = True
        info, meta = _parse_memory_store_payload(q)
        store_args: Dict[str, Any] = {"information": info or q.strip()}
        if meta:
            store_args["metadata"] = meta
        return [("store", store_args), ("qdrant_index_root", idx_args)]

    if intent == INTENT_INDEX:
        # Zero-arg safe default; recreate only if user asked explicitly
        recreate = True if any(w in q.lower() for w in ["recreate", "fresh", "from scratch"]) else None
        args = {}
        if recreate is True:
            args["recreate"] = True
        return [("qdrant_index_root", args)]

    if intent == INTENT_PRUNE:
        return [("qdrant_prune", {})]


    if intent == INTENT_STATUS:
        return [("qdrant_status", {})]

    if intent == INTENT_LIST:
        return [("qdrant_list", {})]


    if intent == INTENT_SEARCH:
        # Parse lightweight repo hints and choose the best search tool via signature similarity
        hints = _parse_repo_hints(q)
        args = {"query": q}
        if search_limit:
            args["limit"] = search_limit
        if include_snippet:
            args["include_snippet"] = True
        # Attach safe filters if we inferred them
        # Optionally reuse last filters if requested
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v
        try:
            tool_servers = _discover_tool_endpoints(allow_network=False)
            picked = _select_best_search_tool_by_signature(q, tool_servers, allow_network=False) or "repo_search"
        except Exception:
            picked = "repo_search"
        return [(picked, args)]

    if intent == INTENT_SEARCH_TESTS:
        hints = _parse_repo_hints(q)
        args = {"query": q}
        if search_limit:
            args["limit"] = search_limit
        if include_snippet:
            args["include_snippet"] = True
        # Optionally reuse last filters if requested
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v
        return [("search_tests_for", args)]

    if intent == INTENT_SEARCH_CONFIG:
        hints = _parse_repo_hints(q)
        args = {"query": q}
        if search_limit:
            args["limit"] = search_limit
        if include_snippet:
            args["include_snippet"] = True
        # Optionally reuse last filters if requested
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v
        return [("search_config_for", args)]

    if intent == INTENT_MEMORY_STORE:
        info, meta = _parse_memory_store_payload(q)
        payload: Dict[str, Any] = {"information": info or q.strip()}
        if meta:
            payload["metadata"] = meta
        return [("store", payload)]

    if intent == INTENT_MEMORY_FIND:
        args = {"query": q}
        if search_limit:
            args["limit"] = max(5, search_limit)
        return [("find", args)]

    if intent == INTENT_SEARCH_CALLERS:
        hints = _parse_repo_hints(q)
        args = {"query": q}
        if search_limit:
            args["limit"] = search_limit
        # Optionally reuse last filters if requested
        _reuse_last_filters(args)

        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v
        return [("search_callers_for", args)]

    if intent == INTENT_SEARCH_IMPORTERS:
        hints = _parse_repo_hints(q)
        args = {"query": q}
        if search_limit:
            args["limit"] = search_limit
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v
        return [("search_importers_for", args)]

    if intent == INTENT_ANSWER:
        # Primary: context_answer_compat, then context_answer; Fallback: repo_search
        # If the prompt looks like a design/recap, preface plan with a memory find step
        def _looks_like_design_recap(s: str) -> bool:
            low = s.lower()
            return any(w in low for w in ["recap", "design doc", "architecture", "adr", "retrospective", "postmortem"]) and any(w in low for w in ["design", "summary", "recap", "explain"])

        # Start with base args and carry through parsed repo hints to answer tools
        args: Dict[str, Any] = {"query": q}
        if max_tokens_env:
            try:
                mt = int(max_tokens_env)
                if mt > 0:
                    args["max_tokens"] = mt
            except Exception:
                pass

        # Parse lightweight hints and tighten when the query mentions "router"
        hints = _parse_repo_hints(q)
        lowq = q.lower()
        if "router" in lowq:
            # Bias strongly toward router implementation
            router_globs = ["**/mcp_router.py", "**/*router*.py"]
            if not hints.get("path_glob"):
                hints["path_glob"] = router_globs
            # Default language bias to python unless explicitly provided
            if not hints.get("language"):
                hints["language"] = "python"
        # Carry hints into args so context_answer and compat use them
        for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
            v = hints.get(k)
            if v not in (None, ""):
                args[k] = v

        plan: List[Tuple[str, Dict[str, Any]]] = []
        if _looks_like_design_recap(q):
            plan.append(("find", {"query": q, "limit": 3}))
        # Answer first (compat then native), then fall back to a targeted search with the same hints
        plan.extend([
            ("context_answer_compat", dict(args)),
            ("context_answer", dict(args)),
            ("repo_search", {**{k: v for k, v in args.items() if k != "max_tokens"}, "limit": max(5, search_limit)}),
        ])
        return plan

    # Fallback
    return [("repo_search", {"query": q, "limit": search_limit})]


# -----------------------------
# HTTP client helpers
# -----------------------------

def _post_raw(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float = 60.0) -> Tuple[Dict[str, str], bytes]:
    req = request.Request(url, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    data = json.dumps(payload).encode("utf-8")
    with request.urlopen(req, data=data, timeout=timeout) as resp:
        body = resp.read()
        # Lowercase header keys for ease
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
    return hdrs, body


def _parse_stream_or_json(body: bytes) -> Dict[str, Any]:
    txt = body.decode("utf-8", errors="ignore")
    # If it looks like SSE/streamable-http (data: ...), pick the last data: line
    if "data:" in txt and ("event:" in txt or txt.strip().startswith("data:")):
        last = None
        for line in txt.splitlines():
            if line.startswith("data:"):
                last = line[len("data:"):].strip()
        if last:
            try:
                return json.loads(last)
            except Exception:
                pass
    # Fallback: parse as JSON
    return json.loads(txt)


def _filter_args(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v not in (None, "")}  # omit null/empty


def _mcp_handshake(base_url: str, timeout: float = 30.0) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "router", "version": "0.1.0"},
        },
        "id": 1,
    }
    hdrs, body = _post_raw_retry(base_url, init_payload, headers, timeout=timeout)
    sid = hdrs.get("mcp-session-id") or hdrs.get("Mcp-Session-Id")
    if not sid:
        # Some servers may return JSON with sessionId; best-effort parse
        try:
            j = _parse_stream_or_json(body)
            sid = j.get("sessionId")
        except Exception:
            sid = None
    if not sid:
        raise RuntimeError("MCP handshake failed: no session id")
    headers["Mcp-Session-Id"] = sid
    # Send initialized notification (no id required)
    try:
        _post_raw_retry(base_url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, headers, timeout=timeout)
    except Exception:
        pass
    return headers


def _extract_iserror_text(resp: Dict[str, Any]) -> str | None:
    try:
        r = resp.get("result") or {}
        if isinstance(r, dict) and r.get("isError"):
            content = r.get("content")
            if isinstance(content, list) and content and isinstance(content[0], dict):
                if content[0].get("type") == "text":
                    return content[0].get("text")
    except Exception:
        pass
    return None


def call_tool_http(base_url: str, tool_name: str, args: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    # Handshake per fastmcp streamable-http
    headers = _mcp_handshake(base_url, timeout=min(timeout, 30.0))

    def _do_call(arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": "router-1",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        _, body = _post_raw_retry(base_url, payload, headers, timeout=timeout)
        return _parse_stream_or_json(body)

    # First attempt: choose wrapper based on tool shape
    args1 = _filter_args(args)
    if tool_name.endswith("_compat"):
        # Compat tools expose a single 'arguments' parameter; call with nested wrapper first
        resp = _do_call({"arguments": args1})
    else:
        resp = _do_call(args1)

    def _get_structured_error(r: Dict[str, Any]) -> str | None:
        try:
            rr = r.get("result") or {}
            sc = rr.get("structuredContent") or {}
            rs = sc.get("result") or {}
            err = rs.get("error")
            if isinstance(err, str):
                return err
        except Exception:
            pass
        return None

    msg = _extract_iserror_text(resp)
    serr = _get_structured_error(resp)
    if msg:
        low = msg.lower()
        if ("kwargs" in low) and ("field required" in low or "missing" in low):
            # Retry with kwargs wrapper for servers that expose **kwargs in schema
            resp2 = _do_call({"kwargs": args1})
            return resp2
        if ("arguments" in low) and ("field required" in low or "missing" in low):
            # Retry with single-blob arguments wrapper for **arguments schemas
            resp3 = _do_call({"arguments": args1})
            return resp3
    # Heuristic: some tools accept the call but return {"error": "query required"}
    # when kwargs weren't expanded. If we provided a query, retry with kwargs wrapper.
    if (serr and serr.strip().lower() == "query required") and ("query" in args1 or "queries" in args1):
        # Attempt 1: top-level kwargs
        resp4 = _do_call({"kwargs": args1})
        serr2 = _get_structured_error(resp4)
        if not (serr2 and serr2.strip().lower() == "query required"):
            return resp4
        # Attempt 2: nested kwargs under arguments (for **arguments wrappers)
        resp5 = _do_call({"arguments": {"kwargs": args1}})
        serr3 = _get_structured_error(resp5)
        if not (serr3 and serr3.strip().lower() == "query required"):
            return resp5
        # Attempt 3: nested direct args under arguments
        resp6 = _do_call({"arguments": args1})
        return resp6
    return resp

def _is_failure_response(resp: Dict[str, Any]) -> bool:
    try:
        r = resp.get("result") or {}
        if r.get("isError") is True:
            return True
        sc = r.get("structuredContent") or {}
        rs = sc.get("result") or {}
        if isinstance(rs, dict) and isinstance(rs.get("error"), str):
            return True
    except Exception:
        return False
    return False

# -----------------------------
# Multi-server tool discovery and result validation
# Prefer live /tools registry from the servers' health ports when available
try:
    _HEALTH_PORT_INDEXER = int(os.environ.get("FASTMCP_INDEXER_HTTP_HEALTH_PORT", "18003") or 18003)
except Exception:
    _HEALTH_PORT_INDEXER = 18003
try:
    _HEALTH_PORT_MEMORY = int(os.environ.get("FASTMCP_HTTP_HEALTH_PORT", "18002") or 18002)
except Exception:
    _HEALTH_PORT_MEMORY = 18002


def _tools_describe_from_health(base_url: str, timeout: float = 3.0) -> list[dict]:
    """Best-effort: fetch tool descriptors from health /tools endpoint.
    Only attempts for known default HTTP_URL_INDEXER/HTTP_URL_MEMORY.
    """
    try:
        import urllib.request
        if base_url == HTTP_URL_INDEXER:
            url = f"http://localhost:{_HEALTH_PORT_INDEXER}/tools"
        elif base_url == HTTP_URL_MEMORY:
            url = f"http://localhost:{_HEALTH_PORT_MEMORY}/tools"
        else:
            return []
        with urllib.request.urlopen(url, timeout=timeout) as r:
            if getattr(r, "status", 200) != 200:
                return []
            body = r.read()
            j = _parse_stream_or_json(body)
            tools = (j.get("tools") if isinstance(j, dict) else None) or []
            out = []
            for t in tools:
                if not isinstance(t, dict):
                    continue
                nm = t.get("name")
                if not nm:
                    continue
                out.append({"name": nm, "description": (t.get("description") or "").strip()})
            return out
    except Exception:
        return []

# -----------------------------

def _post_raw_retry(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: float = 60.0, retries: int = 2, backoff: float = 0.5) -> Tuple[Dict[str, str], bytes]:
    last_exc: Exception | None = None
    for i in range(max(0, retries) + 1):
        try:
            return _post_raw(url, payload, headers, timeout=timeout)
        except Exception as e:
            last_exc = e
            if i < retries:
                try:
                    time.sleep(backoff * (2 ** i))
                except Exception:
                    pass
            else:
                raise last_exc


def _mcp_tools_list(base_url: str, timeout: float = 30.0) -> List[str]:
    try:
        headers = _mcp_handshake(base_url, timeout=min(timeout, 15.0))
        payload = {"jsonrpc": "2.0", "id": "router-list", "method": "tools/list"}
        _, body = _post_raw_retry(base_url, payload, headers, timeout=timeout)
        j = _parse_stream_or_json(body)
        tools = ((j.get("result") or {}).get("tools") or [])
        names: List[str] = []
        for t in tools:
            try:
                n = t.get("name") if isinstance(t, dict) else None
                if isinstance(n, str) and n:
                    names.append(n)
            except Exception:
                continue
        return names
    except Exception:
        return []

# -----------------------------
# Signature embedding + hint parsing
# -----------------------------
try:
    from fastembed import TextEmbedding as _FE_Embedding  # optional
except Exception:  # pragma: no cover
    _FE_Embedding = None  # type: ignore

# Lightweight cosine for both dense and lexical vectors
def _cosine(a: list[float], b: list[float]) -> float:
    try:
        s = 0.0
        na = 0.0
        nb = 0.0
        for i in range(min(len(a), len(b))):
            va = float(a[i])
            vb = float(b[i])
            s += va * vb
            na += va * va
            nb += vb * vb
        na = (na or 1.0) ** 0.5
        nb = (nb or 1.0) ** 0.5
        return s / (na * nb)
    except Exception:
        return 0.0


def _mcp_tools_describe(base_url: str, timeout: float = 20.0) -> list[dict]:
    """Return tool dicts from tools/list (includes name/description/schema if provided)."""
    try:
        headers = _mcp_handshake(base_url, timeout=min(timeout, 10.0))
        payload = {"jsonrpc": "2.0", "id": "router-list2", "method": "tools/list"}
        _, body = _post_raw_retry(base_url, payload, headers, timeout=timeout)
        j = _parse_stream_or_json(body)
        tools = ((j.get("result") or {}).get("tools") or [])
        out = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            name = (t.get("name") or "").strip()
            if not name:
                continue
            out.append(t)
        return out
    except Exception:
        return []

# -----------------------------
# Router discovery caches
# -----------------------------
_TOOL_ENDPOINTS_CACHE_MAP: Dict[str, str] = {}
_TOOL_ENDPOINTS_CACHE_TS: float = 0.0
_TOOLS_DESCR_CACHE: Dict[str, list] = {}
_TOOLS_DESCR_TS: Dict[str, float] = {}


def _cache_ttl_sec() -> int:
    try:
        return int(os.environ.get("ROUTER_TOOLS_CACHE_TTL_SEC", "60") or 60)
    except Exception:
        return 60

# -----------------------------
# Persistent scratchpad (global scope)
# -----------------------------

def _scratchpad_path() -> str:
    base = os.path.join(os.getcwd(), ".codebase")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        pass
    return os.path.join(base, "router_scratchpad.json")


def _load_scratchpad() -> Dict[str, Any]:
    p = _scratchpad_path()
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
            if isinstance(j, dict):
                try:
                    ts = float(j.get("timestamp") or 0.0)
                except Exception:
                    ts = 0.0
                if ts and (time.time() - ts) > _scratchpad_ttl_sec():
                    for stale_key in (
                        "last_plan",
                        "last_filters",
                        "mem_snippets",
                        "last_answer",
                        "last_citations",
                        "last_paths",
                        "last_metrics",
                    ):
                        j.pop(stale_key, None)
                    j["timestamp"] = 0.0
                return j
    except Exception:
        pass
    return {}


def _save_scratchpad(d: Dict[str, Any]) -> None:
    p = _scratchpad_path()
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f)
    except Exception:
        pass


def _looks_like_repeat(q: str) -> bool:
    s = q.strip().lower()
    pats = [
        "repeat", "again", "same thing", "do that again", "rerun", "run it again", "same as before",
    ]
    return any(p in s for p in pats)


def _looks_like_same_filters(q: str) -> bool:
    s = q.strip().lower()
    return any(p in s for p in ["same filters", "reuse filters", "previous filters"])
def _tools_describe_cached(base_url: str, allow_network: bool = True, timeout: float = 20.0) -> list[dict]:
    now = time.time()
    ts = _TOOLS_DESCR_TS.get(base_url, 0.0)
    if base_url in _TOOLS_DESCR_CACHE and (now - ts) <= _cache_ttl_sec():
        return _TOOLS_DESCR_CACHE[base_url]
    if not allow_network:
        return _TOOLS_DESCR_CACHE.get(base_url, [])
    # Prefer /tools on health port; fallback to MCP tools/list
    desc = _tools_describe_from_health(base_url, timeout=min(timeout, 3.0)) or _mcp_tools_describe(base_url, timeout=timeout)
    _TOOLS_DESCR_CACHE[base_url] = desc
    _TOOLS_DESCR_TS[base_url] = now
    return desc


def _scratchpad_ttl_sec() -> int:
    try:
        return int(os.environ.get("ROUTER_SCRATCHPAD_TTL_SEC", "300") or 300)
    except Exception:
        return 300


def _looks_like_expand(q: str) -> bool:
    s = q.strip().lower()
    pats = [
        "expand on", "expand that", "expand the summary", "elaborate",
        "more detail", "more details", "go deeper", "add details",
    ]
    return any(p in s for p in pats)


def _default_tool_endpoints() -> Dict[str, str]:
    idx = HTTP_URL_INDEXER
    mem = HTTP_URL_MEMORY
    mapping: Dict[str, str] = {}
    for n in [
        "repo_search","context_answer","context_answer_compat","expand_query",
        "search_tests_for","search_config_for","search_callers_for","search_importers_for",
        "qdrant_index_root","qdrant_prune","qdrant_status","qdrant_list",
        "workspace_info","list_workspaces","change_history_for_path","code_search","context_search",
    ]:
        mapping[n] = idx
    mapping["store"] = mem
    mapping["find"] = mem
    return mapping



def _signature_text(t: dict) -> str:
    name = (t.get("name") or "").strip()
    desc = (t.get("description") or "").strip()
    # Include simple param names if present
    params = []
    try:
        schema = t.get("inputSchema") or {}
        props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        params = [k for k in props.keys()]
    except Exception:
        params = []
    ptxt = (" params:" + ",".join(params)) if params else ""
    return (name + "\n" + desc + ptxt).strip()


# Lazy embedder (dense if fastembed available; else lexical hashing fallback)
_embedder_singleton = {"model": None}


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    # Try dense embedding (fastembed) first
    if _FE_Embedding is not None:
        try:
            model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            em = _embedder_singleton.get("model")
            if em is None or getattr(em, "_name", "") != model_name:
                em = _FE_Embedding(model_name=model_name)
                em._name = model_name  # type: ignore
                _embedder_singleton["model"] = em
            vecs = [list(next(em.embed([txt]))) for txt in texts]
            return vecs
        except Exception:
            pass
    # Fallback to lexical vectors
    try:
        from scripts.utils import lex_hash_vector_text  # type: ignore
    except Exception:
        # absolute minimal fallback: 1D length proxy
        return [[float(len(t))] for t in texts]
    return [lex_hash_vector_text(t, dim=4096) for t in texts]


def _select_best_search_tool_by_signature(q: str, tool_dict: dict[str, str], allow_network: bool = True) -> str | None:
    """Among available search tools, pick the best match by signature similarity.
    tool_dict maps tool_name -> base_url; we try to fetch descriptions and score.
    Falls back to repo_search unless a specific search_* clearly wins by a small margin.
    """
    # Consider only search_* and repo_search
    candidates = [n for n in tool_dict.keys() if n == "repo_search" or n.startswith("search_")]
    if not candidates:
        return None
    # Fetch descriptions per server (avoid duplicate calls) with cache and offline support
    per_server: dict[str, list[dict]] = {}
    for base in set(tool_dict[t] for t in candidates):
        try:
            per_server[base] = _tools_describe_cached(base, allow_network=allow_network)
        except Exception:
            per_server[base] = []
    sig_map: dict[str, str] = {}
    for tname in candidates:
        base = tool_dict.get(tname)
        descs = per_server.get(base, [])
        # find matching tool object
        obj = None
        for td in descs:
            if (td.get("name") or "").strip() == tname:
                obj = td
                break
        sig_map[tname] = _signature_text(obj or {"name": tname, "description": ""})
    texts = [q] + [sig_map[n] for n in candidates]
    vecs = _embed_texts(texts)
    if not vecs or len(vecs) < 1 + len(candidates):
        return None
    qv = vecs[0]
    scores: list[tuple[str, float]] = []
    for i, name in enumerate(candidates):
        sv = vecs[1 + i]
        scores.append((name, _cosine(qv, sv)))
    scores.sort(key=lambda x: x[1], reverse=True)
    best, best_s = scores[0]
    # Prefer repo_search unless a specific search_* beats it by margin
    repo_s = next((s for n, s in scores if n == "repo_search"), None)
    margin = 0.02
    if best == "repo_search" or repo_s is None:
        return best
    if best != "repo_search" and best_s >= (repo_s + margin):
        return best
    return "repo_search"


_LANGS = {"python","typescript","javascript","go","java","rust","kotlin","c++","cpp","csharp","c#","ruby","php","scala","swift","bash","shell"}


def _parse_repo_hints(q: str) -> Dict[str, Any]:
    """Extract light filters from the query: language, under (path), symbol, ext, path_glob, not_glob."""
    s = q.strip()
    low = s.lower()
    out: Dict[str, Any] = {}
    # language
    for lang in sorted(_LANGS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(lang)}\b", low):
            # normalize a couple common aliases
            out["language"] = {"javascript":"js","typescript":"ts","c++":"cpp","c#":"csharp"}.get(lang, lang)
            break
    # under / in folder
    # Prefer explicit "under <path>"; else fallback to "in/inside <path>"
    m_under = re.search(r"\bunder\s+([\w./-]+)", low)
    m_in = re.search(r"\b(?:in|inside)\s+([\w./-]+)", low)
    m = m_under or m_in
    if m:
        cand = m.group(1)
        # avoid treating language mention as a path (e.g., "in python")
        if len(cand) >= 2 and cand not in _LANGS:
            out["under"] = cand
    # symbol-like tokens: foo(), Foo.bar, pkg::Sym
    m2 = re.search(r"([A-Za-z_][A-Za-z0-9_]*\s*\(\))|([A-Za-z_][\w]*\.[A-Za-z_][\w]*)|([A-Za-z_][\w]*::[A-Za-z_][\w]*)", s)
    if m2:
        sym = m2.group(0)
        sym = re.sub(r"\s*\(\)\s*$", "", sym)
        out["symbol"] = sym
    # file extension mention
    m3 = re.search(r"\.(py|ts|tsx|js|jsx|go|java|rs|kt|rb|php|scala|swift)$", s)
    if m3:
        out["ext"] = m3.group(1)
    # glob inclusions: "only *.py" or "only py files"
    globs: List[str] = []
    if re.search(r"\bonly\b", low):
        # *.ext
        m_glob = re.search(r"\*\.[A-Za-z0-9]+", s)
        if m_glob:
            globs.append("**/" + m_glob.group(0))
        # 'python files' style
        if "python" in low and "*.py" not in " ".join(globs):
            globs.append("**/*.py")
    if globs:
        out["path_glob"] = globs
    # exclusions: "exclude vendor", "exclude tests"
    not_glob: List[str] = []
    for ex in ["vendor", "node_modules", "dist", "build", "tests", "__pycache__"]:
        if re.search(rf"\bexclude\s+{re.escape(ex)}\b", low):
            not_glob.append(f"**/{ex}/**")
    if not_glob:
        out["not_glob"] = not_glob
    return out



def _discover_tool_endpoints(force: bool = False, allow_network: bool = True) -> Dict[str, str]:
    """Return mapping of tool_name -> base_url.
    Uses a small TTL cache; optionally avoids network (offline planning).
    Indexer HTTP has priority if both expose the same tool name.
    """
    global _TOOL_ENDPOINTS_CACHE_TS, _TOOL_ENDPOINTS_CACHE_MAP
    now = time.time()
    ttl = _cache_ttl_sec()
    if not force and _TOOL_ENDPOINTS_CACHE_MAP and (now - _TOOL_ENDPOINTS_CACHE_TS) <= ttl:
        return _TOOL_ENDPOINTS_CACHE_MAP
    if not allow_network:
        # Use cached if present; otherwise a conservative default
        return _TOOL_ENDPOINTS_CACHE_MAP or _default_tool_endpoints()
    mapping: Dict[str, str] = {}
    # Indexer first (priority) — prefer health /tools registry when available
    idx_desc = _tools_describe_cached(HTTP_URL_INDEXER, allow_network=allow_network)
    for t in idx_desc:
        n = t.get("name") if isinstance(t, dict) else None
        if n:
            mapping[n] = HTTP_URL_INDEXER
    # Memory server next
    mem_desc = _tools_describe_cached(HTTP_URL_MEMORY, allow_network=allow_network)
    for t in mem_desc:
        n = t.get("name") if isinstance(t, dict) else None
        if n and n not in mapping:
            mapping[n] = HTTP_URL_MEMORY
    if mapping:
        _TOOL_ENDPOINTS_CACHE_MAP.clear()
        _TOOL_ENDPOINTS_CACHE_MAP.update(mapping)
        _TOOL_ENDPOINTS_CACHE_TS = now
    return mapping or (_TOOL_ENDPOINTS_CACHE_MAP or _default_tool_endpoints())


def _is_result_good(tool: str, resp: Dict[str, Any]) -> bool:
    """Lightweight validation to decide if we should stop or try fallbacks."""
    try:
        r = resp.get("result") or {}
        sc = r.get("structuredContent") or {}
        rs = sc.get("result") or {}
        # Answer tools: require a non-empty answer and avoid obvious "no context" replies
        if tool in {"context_answer", "context_answer_compat"}:
            ans = rs.get("answer") if isinstance(rs, dict) else None
            if isinstance(ans, str):
                s = ans.strip()
                if s and not any(p in s.lower() for p in [
                    "insufficient context", "not enough context", "no relevant", "don't know", "cannot answer"
                ]):
                    return True
            # If empty or unclear text, accept if we have citations
            cites = rs.get("citations") if isinstance(rs, dict) else None
            if isinstance(cites, list) and len(cites) > 0:
                return True
            return False
        # Search tools: require at least one hit
        if tool.startswith("search_") or tool == "repo_search":
            total = rs.get("total") if isinstance(rs, dict) else None
            if isinstance(total, int) and total > 0:
                return True
            results = rs.get("results") if isinstance(rs, dict) else None
            if isinstance(results, list) and len(results) > 0:
                return True
            return False
        # Admin/status tools: treat non-error responses as success
        return not _is_failure_response(resp)
    except Exception:
        # Be conservative: if we can't parse, require non-error
        return not _is_failure_response(resp)



# -----------------------------
# Divergence detection helpers (scratchpad v3)
# -----------------------------

def _extract_metric_from_resp(tool: str, resp: Dict[str, Any]) -> tuple[str, float] | None:
    try:
        r = resp.get("result") or {}
        sc = r.get("structuredContent") or {}
        rs = sc.get("result") or {}
        # Search-like tools
        if tool in {"repo_search", "code_search", "context_search", "search_tests_for", "search_config_for", "search_callers_for", "search_importers_for"}:
            tot = rs.get("total")
            if isinstance(tot, (int, float)):
                return ("total_results", float(tot))
            results = rs.get("results")
            if isinstance(results, list):
                return ("total_results", float(len(results)))
            return None
        # Answer tools: track citations count
        if tool in {"context_answer", "context_answer_compat"}:
            cites = rs.get("citations")
            if isinstance(cites, list):
                return ("citations", float(len(cites)))
            return ("citations", 0.0)
        # Admin/status: qdrant_status → point count
        if tool == "qdrant_status":
            cnt = rs.get("count")
            if isinstance(cnt, (int, float)):
                return ("points", float(cnt))
            return None
    except Exception:
        return None
    return None


def _divergence_thresholds() -> tuple[float, int]:
    try:
        drop_frac = float(os.environ.get("ROUTER_DIVERGENCE_DROP_FRAC", "0.5") or 0.5)
    except Exception:
        drop_frac = 0.5
    try:
        min_base = int(os.environ.get("ROUTER_DIVERGENCE_MIN_BASE", "3") or 3)
    except Exception:
        min_base = 3
    return drop_frac, min_base


def _material_drop(prev: float | None, curr: float, drop_frac: float, min_base: int) -> bool:
    try:
        if prev is None:
            return False
        if prev < float(min_base):
            return False
        return curr < (float(prev) * float(drop_frac))
    except Exception:
        return False


# Determine if divergence should be fatal for a given tool (env-driven)
# ROUTER_DIVERGENCE_FATAL_TOOLS can be a comma-separated list (case-insensitive), or '*'/'all' to apply to all tools.
def _divergence_is_fatal_for(tool: str) -> bool:
    try:
        s = (os.environ.get("ROUTER_DIVERGENCE_FATAL_TOOLS", "") or "").strip()
        if not s:
            return False
        low = s.lower()
        if low in {"*", "all", "1", "true"}:
            return True
        names = {t.strip().lower() for t in s.split(",") if t.strip()}
        return tool.strip().lower() in names
    except Exception:
        return False

# -----------------------------
# CLI
# -----------------------------

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query to route")
    ap.add_argument("--plan", action="store_true", help="Only print routing plan (no execution)")
    ap.add_argument("--run", action="store_true", help="Execute the routed tool(s) over HTTP")
    ap.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout for tool calls")
    args = ap.parse_args(argv)

    plan = build_plan(args.query)
    print(json.dumps({"router": {"url": HTTP_URL_INDEXER, "plan": plan}}, indent=2))

    if args.plan and not args.run:
        return 0

    # Load scratchpad for prior context and TTL freshness
    sp = {}
    fresh = False
    prior_answer = None
    prior_citations = None
    prior_paths = None
    try:
        sp = _load_scratchpad()
        ts = float(sp.get("timestamp") or 0.0)
        fresh = bool(ts and (time.time() - ts) <= _scratchpad_ttl_sec())
        if fresh:
            prior_answer = sp.get("last_answer")
            prior_citations = sp.get("last_citations")
            prior_paths = sp.get("last_paths")
    except Exception:
        pass

    # Execute sequentially until one succeeds
    last_err = None
    last = None
    tool_servers = _discover_tool_endpoints()
    # Simple scratchpad for memory→answer workflow
    mem_snippets: list[str] = list(sp.get("mem_snippets") or []) if fresh else []
    for idx, (tool, targs) in enumerate(plan):
        base_url = tool_servers.get(tool, HTTP_URL_INDEXER)
        # Skip memory.find if we already have fresh snippets and this is a repeat/expand
        if (tool.lower().endswith("find") or tool.lower() in {"find", "memory.find"}) and mem_snippets and fresh and (_looks_like_repeat(args.query) or _looks_like_expand(args.query)):
            try:
                print(json.dumps({"tool": tool, "skipped": "scratchpad_fresh"}))
            except Exception:
                pass
            continue

        # If we have memory/prior summary and are about to answer, augment the query text
        if tool in {"context_answer", "context_answer_compat"} and (mem_snippets or (fresh and (prior_answer or prior_citations or prior_paths))):
            try:
                tq = str((targs or {}).get("query") or args.query)
                sections = [tq]
                # Memory snippets → bullets
                if mem_snippets:
                    bullets = []
                    for s in mem_snippets[:3]:
                        ss = re.sub(r"\s+", " ", str(s)).strip()
                        if len(ss) > 200:
                            ss = ss[:197] + "..."
                        bullets.append(f"- {ss}")
                    sections.append("Memory context:\n" + "\n".join(bullets))
                # Prior summary/citations if asked to expand or repeat and still fresh
                if fresh and (_looks_like_expand(args.query) or _looks_like_repeat(args.query)):
                    if isinstance(prior_answer, str) and prior_answer.strip():
                        pa = re.sub(r"\s+", " ", prior_answer).strip()
                        if len(pa) > 400:
                            pa = pa[:397] + "..."
                        sections.append("Prior summary:\n" + pa)
                    paths_list = []
                    if isinstance(prior_paths, list) and prior_paths:
                        paths_list = [str(p) for p in prior_paths[:5]]
                    elif isinstance(prior_citations, list) and prior_citations:
                        uniq = []
                        for c in prior_citations:
                            if isinstance(c, dict) and c.get("path") and c["path"] not in uniq:
                                uniq.append(c["path"])
                        paths_list = uniq[:5]
                    if paths_list:
                        sections.append("Citations context:\n" + "\n".join(f"- {p}" for p in paths_list))
                aug = "\n\n".join(sections)
                targs = {**(targs or {}), "query": aug}
            except Exception:
                pass
        try:
            res = call_tool_http(base_url, tool, targs, timeout=args.timeout)
            print(json.dumps({"tool": tool, "result": res}, indent=2))
            last = res
            # If this was a memory.find step, capture snippets for later augmentation
            try:
                if tool.lower().endswith("find") or tool.lower() in {"find", "memory.find"}:
                    r = res.get("result") or {}
                    items = []
                    sc = r.get("structuredContent")
                    if isinstance(sc, dict):
                        rs0 = sc.get("result") or sc
                        if isinstance(rs0, dict):
                            items = rs0.get("results") or rs0.get("hits") or []
                    if not items:
                        content = r.get("content")
                        if isinstance(content, list):
                            for c in content:
                                if not isinstance(c, dict):
                                    continue
                                # Prefer native JSON payloads if present
                                if "json" in c:
                                    j = c.get("json")
                                    if isinstance(j, (dict, list)):
                                        container = j.get("result") if isinstance(j, dict) and "result" in j else j
                                        if isinstance(container, dict):
                                            items = container.get("results") or container.get("hits") or []
                                            if items:
                                                break
                                # Fallback: text field containing JSON
                                if c.get("type") == "text":
                                    ttxt = c.get("text")
                                    if isinstance(ttxt, str) and ttxt.strip():
                                        try:
                                            j = json.loads(ttxt)
                                        except Exception:
                                            continue
                                        container = j.get("result") if isinstance(j, dict) and "result" in j else j
                                        if isinstance(container, dict):
                                            items = container.get("results") or container.get("hits") or []
                                            if items:
                                                break
                    for it in items:
                        if isinstance(it, dict):
                            txt = it.get("information") or it.get("content") or it.get("text")
                            if isinstance(txt, str) and txt.strip():
                                mem_snippets.append(txt.strip())
            except Exception:
                pass
            # Determine if we should treat this step as terminal
            has_future_answer = any(tn in {"context_answer", "context_answer_compat"} for (tn, _) in plan[idx + 1 :])
            if (not _is_failure_response(res)) and _is_result_good(tool, res):
                if tool.lower() in {"find", "memory.find"} and has_future_answer:
                    # Don't stop; proceed to answer step with augmented query
                    continue
                # Persist scratchpad: filters, memory, prior answer/citations, and success criteria
                try:
                    last_filters: Dict[str, Any] = {}
                    for (tn, ta) in plan:
                        if tn == "repo_search" or tn.startswith("search_"):
                            if isinstance(ta, dict):
                                for k in ("language", "under", "symbol", "ext", "path_glob", "not_glob"):
                                    if ta.get(k) not in (None, ""):
                                        last_filters[k] = ta.get(k)
                            break
                    # Extract prior answer and citations if this was an answer tool
                    last_answer_text = None
                    last_citations_list = None
                    last_paths_list: list[str] | None = None
                    if tool in {"context_answer", "context_answer_compat"}:
                        try:
                            r0 = res.get("result") or {}
                            sc0 = r0.get("structuredContent") or {}
                            rs0 = sc0.get("result") or sc0
                            if isinstance(rs0, dict):
                                ans0 = rs0.get("answer")
                                if isinstance(ans0, str):
                                    last_answer_text = ans0
                                cites0 = rs0.get("citations")
                                if isinstance(cites0, list):
                                    last_citations_list = cites0
                                    uniqp: list[str] = []
                                    for c in cites0:
                                        if isinstance(c, dict) and c.get("path") and c["path"] not in uniqp:
                                            uniqp.append(c["path"])
                                    last_paths_list = uniqp
                        except Exception:
                            pass
                    # Divergence detection against last-known-good metrics
                    divergence_should_abort = False
                    last_metrics_prev = {}
                    try:
                        last_metrics_prev = sp.get("last_metrics") or {}
                        if not isinstance(last_metrics_prev, dict):
                            last_metrics_prev = {}
                    except Exception:
                        last_metrics_prev = {}
                    metric = _extract_metric_from_resp(tool, res)
                    last_metrics_map = dict(last_metrics_prev)
                    if metric is not None:
                        mname, mval = metric
                        # Compare to previous metric for this tool
                        prev_val = None
                        try:
                            prev_val = last_metrics_prev.get(tool, {}).get(mname)
                            if prev_val is not None:
                                prev_val = float(prev_val)
                        except Exception:
                            prev_val = None
                        drop_frac, min_base = _divergence_thresholds()
                        if _material_drop(prev_val, float(mval), drop_frac, min_base):
                            fatal = _divergence_is_fatal_for(tool)
                            try:
                                print(json.dumps({
                                    "divergence": {
                                        "tool": tool,
                                        "metric": mname,
                                        "previous": prev_val,
                                        "current": float(mval),
                                        "drop_frac": drop_frac,
                                        "fatal": fatal,
                                    }
                                }))
                            except Exception:
                                pass
                            if fatal:
                                divergence_should_abort = True
                        # Update metrics for persistence
                        try:
                            last_metrics_map.setdefault(tool, {})[mname] = float(mval)
                        except Exception:
                            pass
                    else:
                        last_metrics_map = last_metrics_prev

                    success_criteria = {
                        "context_answer": {"expected_fields": ["answer"], "min_citations": 0},
                        "context_answer_compat": {"expected_fields": ["answer"], "min_citations": 0},
                        "repo_search": {"min_results": 1},
                        "search_config_for": {"min_results": 1},
                        "search_tests_for": {"min_results": 1},
                        "search_callers_for": {"min_results": 1},
                        "search_importers_for": {"min_results": 1},
                        "find": {"min_results": 1},
                    }
                    sp = {
                        "last_query": args.query,
                        "last_plan": plan,
                        "last_filters": last_filters or None,
                        "mem_snippets": mem_snippets[:5],
                        "last_answer": last_answer_text,
                        "last_citations": last_citations_list,
                        "last_paths": last_paths_list,
                        "success_criteria": success_criteria,
                        "last_metrics": last_metrics_map,
                        "timestamp": time.time(),
                    }
                    _save_scratchpad(sp)
                except Exception:
                    pass

                if divergence_should_abort:
                    # Treat as failure for this tool and try next in plan
                    continue

                return 0
            # else try next tool in plan
        except Exception as e:
            last_err = e
            # Print a concise hint about which server failed
            try:
                print(json.dumps({"tool": tool, "server": base_url, "error": str(e)}), file=sys.stderr)
            except Exception:
                pass
            continue
    if last_err:
        print(f"Router: all attempts failed: {last_err}", file=sys.stderr)
    # If we have a last response but it was a failure, return non-zero
    return 1 if last is not None else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
