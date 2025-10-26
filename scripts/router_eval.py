import json, os, threading, time, sys, re
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Tuple

# Simple Mock MCP server for evals
class MockMCPHandler(BaseHTTPRequestHandler):
    server_version = "MockMCP/0.1"

    def _send_json(self, obj: Dict[str, Any], session: str | None = None, code: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if session:
            self.send_header("Mcp-Session-Id", session)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        try:
            j = json.loads(raw.decode("utf-8", errors="ignore"))
        except Exception:
            return self._send_json({"jsonrpc": "2.0", "error": {"message": "bad json"}}, code=400)
        method = j.get("method")
        if method == "initialize":
            # Return session via header; some clients also parse body
            return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"ok": True, "server": self.server.server_name}}, session="mock-session")
        if method == "notifications/initialized":
            return self._send_json({"jsonrpc": "2.0", "result": {"ok": True}})
        if method == "tools/list":
            tools = getattr(self.server, "tools", [])
            return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"tools": tools}})
        if method == "tools/call":
            params = j.get("params") or {}
            name = (params.get("name") or "").strip()
            args = params.get("arguments") or {}
            # Indexer tools
            if name in {"repo_search", "search_config_for", "search_tests_for", "search_callers_for", "search_importers_for"}:
                res = {
                    "result": {
                        "args": {
                            "queries": [str(args.get("query") or "")],
                            "limit": int(args.get("limit") or 8),
                            "include_snippet": bool(args.get("include_snippet") or False),
                            "language": str(args.get("language") or ""),
                            "under": str(args.get("under") or ""),
                            "symbol": str(args.get("symbol") or ""),
                            "ext": str(args.get("ext") or ""),
                            "compact": False,
                        },
                        "total": 1,
                        "results": [
                            {"score": 0.9, "path": "/work/README.md", "start_line": 1, "end_line": 2, "snippet": "demo"}
                        ],
                        "ok": True,
                        "code": 0,
                        "stdout": "",
                        "stderr": "",
                    }
                }
                return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}})
            if name == "context_answer_compat":
                # Require nested arguments wrapper
                if not isinstance(args, dict) or "arguments" not in args:
                    return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps({"error": "compat requires nested arguments"})}], "structuredContent": {"error": "compat requires nested arguments"}, "isError": True}})
                inner = args.get("arguments") or {}
                q = str(inner.get("query") or "")
                ans = {
                    "answer": "Short ok." if len(q) < 80 else "Longer answer",
                    "citations": [{"id": 1, "path": "/work/file.py", "start_line": 1, "end_line": 2}],
                    "query": [q],
                    "used": {"gate_first": True, "refrag": True},
                }
                res = {"result": ans}
                return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}})
            if name == "context_answer":
                q = str(args.get("query") or "")
                ans = {"answer": "Ok.", "citations": []}
                res = {"result": ans}
                return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}})
            if name in {"qdrant_status", "qdrant_list"}:
                res = {"result": {"ok": True}}
                return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}})
            # Memory tools
            if name == "store":
                res = {"result": {"ok": True}}
                return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}})
            if name == "find":
                q = str(args.get("query") or "")
                res = {"result": {"ok": True, "results": [{"information": "The MCP indexer uses hybrid search combining dense embeddings and lexical matching with optional reranking", "metadata": {"category": "architecture"}}], "count": 1}}
                return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(res)}], "structuredContent": res, "isError": False}})
            return self._send_json({"jsonrpc": "2.0", "id": j.get("id"), "result": {"content": [{"type": "text", "text": json.dumps({"error": f"unknown tool {name}"})}], "structuredContent": {"error": f"unknown tool {name}"}, "isError": True}})
        return self._send_json({"jsonrpc": "2.0", "error": {"message": f"unknown method {method}"}}, code=400)


def start_mock_server(port: int, tools: List[Dict[str, Any]]) -> Tuple[HTTPServer, threading.Thread]:
    httpd = HTTPServer(("localhost", port), MockMCPHandler)
    httpd.tools = tools  # type: ignore
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    # Warmup
    time.sleep(0.05)
    return httpd, t


def tool(name: str, description: str, params: List[str] = None) -> Dict[str, Any]:
    schema = {"type": "object", "properties": {p: {"type": "string"} for p in (params or [])}}
    return {"name": name, "description": description, "inputSchema": schema}


def run_eval_suite() -> int:
    # Two mock servers: indexer and memory
    indexer_tools = [
        tool("repo_search", "General code search", ["query", "limit", "include_snippet", "language", "under", "symbol", "ext"]),
        tool("search_config_for", "Intent-specific search for configuration files", ["query", "limit", "include_snippet"]),
        tool("search_importers_for", "Find files importing a module or symbol", ["query", "limit", "language", "under"]),
        tool("context_answer_compat", "Answer a question using code context (compat)", ["arguments"]),
        tool("context_answer", "Answer a question using code context", ["query", "limit"]),
        tool("qdrant_status", "Qdrant status"),
        tool("qdrant_list", "Qdrant list"),
    ]
    memory_tools = [tool("store", "Store memory", ["information"]), tool("find", "Find memory", ["query", "limit"])]
    idx, _ = start_mock_server(18031, indexer_tools)
    mem, _ = start_mock_server(18032, memory_tools)

    try:
        os.environ["MCP_INDEXER_HTTP_URL"] = "http://localhost:18031/mcp"
        os.environ["MCP_MEMORY_HTTP_URL"] = "http://localhost:18032/mcp"
        os.environ["ROUTER_SEARCH_LIMIT"] = "8"
        os.environ["ROUTER_INCLUDE_SNIPPET"] = "1"

        # Import router after env set so its defaults bind to mock URLs
        import importlib.util as _ilu
        _p = os.path.join(os.path.dirname(__file__), "mcp_router.py")
        _spec = _ilu.spec_from_file_location("mcp_router", _p)
        router = _ilu.module_from_spec(_spec)
        assert _spec and _spec.loader
        _spec.loader.exec_module(router)  # type: ignore

        failures = []

        def run_plan(q: str) -> List[Tuple[str, Dict[str, Any]]]:
            plan = router.build_plan(q)
            return plan

        # 1) Signature selection: prefer search_config_for for config changes
        p1 = run_plan("compare callers to config changes")
        if not p1 or p1[0][0] != "search_config_for":
            failures.append("signature selection: expected search_config_for")

        # 2) Repo hints: language+under parsed
        p2 = run_plan("who imports foo in python under src/lib")
        if not p2 or p2[0][0] != "search_importers_for":
            failures.append("repo hints: expected search_importers_for")
        else:
            args2 = p2[0][1]
            if args2.get("language") != "python":
                failures.append("repo hints: language not parsed")
            if args2.get("under") != "src/lib":
                failures.append("repo hints: under not parsed")

        # 3) Design recap: memory find precedes answer
        p3 = run_plan("recap our architecture decisions for the indexer")
        expect_order = ["find", "context_answer_compat"]
        if not p3 or [p3[0][0], p3[1][0]] != expect_order:
            failures.append("design recap plan: expected find -> context_answer_compat")

        # 4) Multi-intent: store + reindex
        p4 = run_plan("remember this: prefer concise answers; then reindex fresh")
        if not p4 or [p4[0][0], p4[1][0]] != ["store", "qdrant_index_root"]:
            failures.append("multi-intent: expected store then index")
        else:
            if not p4[1][1].get("recreate"):
                failures.append("multi-intent: expected recreate true")

        # 5) Run end-to-end for recap and ensure compat accepted and short answers not rejected
        # Capture stdout of router.main
        def run_router(args: List[str]) -> str:
            from io import StringIO
            old = sys.stdout
            try:
                buf = StringIO()
                sys.stdout = buf
                router.main(args)
                return buf.getvalue()
            finally:
                sys.stdout = old
        out = run_router(["--run", "recap our architecture decisions for the indexer"])
        if "compat requires nested arguments" in out:
            failures.append("compat: still sending flattened args")
        if "Memory context:" not in out:
            failures.append("memory→answer: query was not augmented with memory context")
            print("--- router stdout ---\n" + out + "\n--- end stdout ---")

        if failures:
            print("Router eval: FAIL\n- " + "\n- ".join(failures))
            return 1
        print("Router eval: PASS (all checks)")
        return 0
    finally:
        idx.shutdown(); mem.shutdown()


if __name__ == "__main__":
    raise SystemExit(run_eval_suite())

