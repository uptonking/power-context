#!/usr/bin/env python3
"""
ingest/metadata.py - Metadata extraction for indexed files.

This module provides functions for extracting git metadata, imports, calls,
and other file-level information for the indexing pipeline.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import List, Tuple, Optional


def _git_metadata(file_path: Path) -> Tuple[int, int, int]:
    """Return (last_modified_at, churn_count, author_count) using git when available.
    
    Falls back to fs mtime and zeros when not in a repo.
    """
    try:
        import subprocess

        fp = str(file_path)
        # last commit unix timestamp (%ct)
        ts = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout.strip()
        last_ts = int(ts) if ts.isdigit() else int(file_path.stat().st_mtime)
        # churn: number of commits touching this file (bounded)
        churn_s = subprocess.run(
            ["git", "rev-list", "--count", "HEAD", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout.strip()
        churn = int(churn_s) if churn_s.isdigit() else 0
        # author count
        authors = subprocess.run(
            ["git", "shortlog", "-s", "--", fp],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        ).stdout
        author_count = len([ln for ln in authors.splitlines() if ln.strip()])
        return last_ts, churn, author_count
    except Exception:
        try:
            return int(file_path.stat().st_mtime), 0, 0
        except Exception:
            return int(time.time()), 0, 0


def _extract_imports(language: str, text: str) -> List[str]:
    """Lightweight import extraction per language (best-effort)."""
    lines = text.splitlines()
    imps: List[str] = []
    if language == "python":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.]+)", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*from\s+([\w\.]+)\s+import\s+", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language in ("javascript", "typescript"):
        for ln in lines:
            m = re.match(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*require\(\s*['\"]([^'\"]+)['\"]\s*\)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "go":
        block = False
        for ln in lines:
            if re.match(r"^\s*import\s*\(", ln):
                block = True
                continue
            if block:
                if ")" in ln:
                    block = False
                    continue
                m = re.match(r"^\s*\"([^\"]+)\"", ln)
                if m:
                    imps.append(m.group(1))
                    continue
            m = re.match(r"^\s*import\s+\"([^\"]+)\"", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "java":
        for ln in lines:
            m = re.match(r"^\s*import\s+([\w\.\*]+);", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "csharp":
        for ln in lines:
            m = re.match(r"^\s*using\s+(?:static\s+)?([A-Za-z_][\w\._]*)(?:\s*;|\s*=)", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "php":
        for ln in lines:
            m = re.match(r"^\s*use\s+(?:function\s+|const\s+)?([A-Za-z_][A-Za-z0-9_\\\\]*)\s*;", ln)
            if m:
                imps.append(m.group(1).replace("\\\\", "\\"))
                continue
        for ln in lines:
            m = re.match(r"^\s*(?:include|include_once|require|require_once)\s*\(?\s*['\"]([^'\"]+)['\"]\s*\)?\s*;", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "rust":
        for ln in lines:
            m = re.match(r"^\s*use\s+([^;]+);", ln)
            if m:
                imps.append(m.group(1).strip())
                continue
    elif language == "terraform":
        for ln in lines:
            m = re.match(r"^\s*source\s*=\s*['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*provider\s*=\s*['\"]([^'\"]+)['\"]", ln)
            if m:
                imps.append(m.group(1))
                continue
    elif language == "powershell":
        for ln in lines:
            m = re.match(
                r"^\s*Import-Module\s+([A-Za-z0-9_.\-]+)", ln, flags=re.IGNORECASE
            )
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*using\s+module\s+([^\s;]+)", ln, flags=re.IGNORECASE)
            if m:
                imps.append(m.group(1))
                continue
            m = re.match(r"^\s*using\s+namespace\s+([^\s;]+)", ln, flags=re.IGNORECASE)
            if m:
                imps.append(m.group(1))
                continue
    return imps[:200]


def _extract_calls(language: str, text: str) -> List[str]:
    """Lightweight call-site extraction (best-effort, language-agnostic heuristics)."""
    names: List[str] = []
    # Simple heuristic: word followed by '(' that isn't a keyword
    kw = set([
        "if", "for", "while", "switch", "return", "new",
        "catch", "func", "def", "class", "match",
    ])
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
        name = m.group(1)
        if name not in kw:
            names.append(name)
    # Deduplicate preserving order
    out: List[str] = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out[:200]


def _get_imports_calls(language: str, text: str) -> Tuple[List[str], List[str]]:
    """Get imports and calls for a file, using tree-sitter when available."""
    from scripts.ingest.tree_sitter import _use_tree_sitter, _ts_parser
    
    if _use_tree_sitter() and language == "python":
        return _ts_extract_imports_calls_python(text)
    return _extract_imports(language, text), _extract_calls(language, text)


def _ts_extract_imports_calls_python(text: str) -> Tuple[List[str], List[str]]:
    """Extract imports and calls from Python using tree-sitter."""
    from scripts.ingest.tree_sitter import _ts_parser
    
    parser = _ts_parser("python")
    if not parser:
        return [], []
    data = text.encode("utf-8")
    try:
        tree = parser.parse(data)
        if tree is None:
            return [], []
        root = tree.root_node
    except (ValueError, Exception):
        return [], []

    def node_text(n):
        return data[n.start_byte : n.end_byte].decode("utf-8", errors="ignore")

    imports: List[str] = []
    calls: List[str] = []

    def walk(n):
        t = n.type
        if t == "import_statement":
            s = node_text(n)
            m = re.search(r"\bimport\s+([\w\.]+)", s)
            if m:
                imports.append(m.group(1))
        elif t == "import_from_statement":
            s = node_text(n)
            m = re.search(r"\bfrom\s+([\w\.]+)\s+import\b", s)
            if m:
                imports.append(m.group(1))
        elif t == "call":
            func = n.child_by_field_name("function")
            if func:
                name = node_text(func)
                base = re.split(r"[\.:]", name)[-1]
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", base):
                    calls.append(base)
        for c in n.children:
            walk(c)

    walk(root)
    # Deduplicate preserving order
    seen = set()
    calls_dedup = []
    for x in calls:
        if x not in seen:
            calls_dedup.append(x)
            seen.add(x)
    return imports[:200], calls_dedup[:200]


def _get_host_path_from_origin(workspace_path: str, repo_name: str = None) -> Optional[str]:
    """Get client host_path from origin source_path in workspace state."""
    try:
        from scripts.workspace_state import get_workspace_state
        state = get_workspace_state(workspace_path, repo_name)
        if state and state.get("origin", {}).get("source_path"):
            return state["origin"]["source_path"]
    except Exception:
        pass
    return None


def _compute_host_and_container_paths(cur_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Compute host_path and container_path for a given absolute path."""
    _host_root = str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
    if ":" in _host_root:
        _host_root = ""
    _host_path: Optional[str] = None
    _container_path: Optional[str] = None
    _origin_client_path: Optional[str] = None

    try:
        if cur_path.startswith("/work/"):
            _parts = cur_path[6:].split("/")
            if len(_parts) >= 2:
                _repo_name = _parts[0]
                _workspace_path = f"/work/{_repo_name}"
                _origin_client_path = _get_host_path_from_origin(
                    _workspace_path, _repo_name
                )
    except Exception:
        _origin_client_path = None

    try:
        if cur_path.startswith("/work/") and (_host_root or _origin_client_path):
            _rel = cur_path[len("/work/"):]
            if _origin_client_path:
                _parts = _rel.split("/", 1)
                _tail = _parts[1] if len(_parts) > 1 else ""
                _base = _origin_client_path.rstrip("/")
                _host_path = (
                    os.path.realpath(os.path.join(_base, _tail)) if _tail else _base
                )
            else:
                _host_path = os.path.realpath(os.path.join(_host_root, _rel))
            _container_path = cur_path
        else:
            _host_path = cur_path
            if (
                (_host_root or _origin_client_path)
                and cur_path.startswith(((_origin_client_path or _host_root) + "/"))
            ):
                _rel = cur_path[len((_origin_client_path or _host_root)) + 1:]
                _container_path = "/work/" + _rel
    except Exception:
        _host_path = cur_path
        _container_path = cur_path if cur_path.startswith("/work/") else None

    return _host_path, _container_path
