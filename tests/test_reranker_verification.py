import importlib
import os
import sys
import types
import pytest

# Provide a minimal stub for mcp.server.fastmcp.FastMCP so importing the server doesn't exit
mcp_pkg = types.ModuleType("mcp")
server_pkg = types.ModuleType("mcp.server")
fastmcp_pkg = types.ModuleType("mcp.server.fastmcp")

class _FastMCP:
    def __init__(self, *args, **kwargs):
        pass
    def tool(self, *args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

setattr(fastmcp_pkg, "FastMCP", _FastMCP)
sys.modules.setdefault("mcp", mcp_pkg)
sys.modules.setdefault("mcp.server", server_pkg)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp_pkg)

srv = importlib.import_module("scripts.mcp_indexer_server")


@pytest.mark.service
@pytest.mark.anyio
async def test_rerank_inproc_changes_order(monkeypatch):
    # Force in-process hybrid + in-process rerank paths
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")
    monkeypatch.setenv("RERANK_IN_PROCESS", "1")

    # Baseline hybrid results (JSON structured items); A before B
    def fake_run_hybrid_search(**kwargs):
        return [
            {
                "score": 0.6,
                "path": "/work/a.py",
                "symbol": "",
                "start_line": 1,
                "end_line": 3,
            },
            {
                "score": 0.5,
                "path": "/work/b.py",
                "symbol": "",
                "start_line": 10,
                "end_line": 12,
            },
        ]

    # Reranker returns higher score for B than A to force reordering
    def fake_rerank_local(pairs):
        # pairs order corresponds to hybrid results; return scores [A,B] -> [0.10, 0.90]
        return [0.10, 0.90]

    # Patch hybrid and rerank
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    monkeypatch.setattr(
        importlib.import_module("scripts.hybrid_search"), "run_hybrid_search", fake_run_hybrid_search
    )
    monkeypatch.setattr(importlib.import_module("scripts.rerank_local"), "rerank_local", fake_rerank_local)

    # Baseline (rerank disabled) preserves hybrid order A then B
    base = await srv.repo_search(query="q", limit=2, per_path=2, rerank_enabled=False, compact=True)
    assert [r["path"] for r in base["results"]] == ["/work/a.py", "/work/b.py"]

    # With rerank enabled, order should flip to B then A; counters should show inproc_hybrid
    rr = await srv.repo_search(query="q", limit=2, per_path=2, rerank_enabled=True, compact=True)
    assert rr.get("used_rerank") is True
    assert rr.get("rerank_counters", {}).get("inproc_hybrid", 0) >= 1
    assert [r["path"] for r in rr["results"]] == ["/work/b.py", "/work/a.py"]


@pytest.mark.service
@pytest.mark.anyio
async def test_rerank_subprocess_timeout_fallback(monkeypatch):
    # Force hybrid via subprocess output (doesn't matter which) and disable inproc rerank
    monkeypatch.setenv("HYBRID_IN_PROCESS", "1")
    monkeypatch.setenv("RERANK_IN_PROCESS", "0")

    def fake_run_hybrid_search(**kwargs):
        return [
            {"score": 0.6, "path": "/work/a.py", "symbol": "", "start_line": 1, "end_line": 3},
            {"score": 0.5, "path": "/work/b.py", "symbol": "", "start_line": 10, "end_line": 12},
        ]

    async def fake_run_async(cmd, env=None, timeout=None):
        # Simulate subprocess reranker timing out
        return {"ok": False, "code": -1, "stdout": "", "stderr": f"Command timed out after {timeout}s"}

    monkeypatch.setattr(
        importlib.import_module("scripts.hybrid_search"), "run_hybrid_search", fake_run_hybrid_search
    )
    monkeypatch.setattr(srv, "_run_async", fake_run_async)

    rr = await srv.repo_search(query="q", limit=2, per_path=2, rerank_enabled=True, compact=True)
    # Fallback should keep original order from hybrid; timeout counter incremented
    assert rr.get("used_rerank") is False
    assert rr.get("rerank_counters", {}).get("timeout", 0) >= 1
    assert [r["path"] for r in rr["results"]] == ["/work/a.py", "/work/b.py"]

