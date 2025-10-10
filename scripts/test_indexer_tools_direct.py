#!/usr/bin/env python3
"""Direct test of indexer MCP tools (bypassing MCP protocol)"""
import asyncio
import json
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the server module
from scripts import mcp_indexer_server as server

async def main():
    print("=== Testing qdrant_list ===")
    result = await server.qdrant_list()
    print(json.dumps(result, indent=2))
    
    print("\n=== Testing repo_search (simple query) ===")
    result = await server.repo_search(
        query="hybrid search python",
        limit=5,
        include_snippet=False,
    )
    # Print compact version
    compact = {
        "ok": result.get("ok"),
        "code": result.get("code"),
        "args": result.get("args"),
        "result_count": len(result.get("results", [])),
        "top_3": [
            {
                "score": round(r.get("score", 0), 4),
                "path": r.get("path", ""),
                "symbol": r.get("symbol", ""),
            }
            for r in result.get("results", [])[:3]
        ],
    }
    print(json.dumps(compact, indent=2))
    
    print("\n=== Testing repo_search (DSL query with file filter) ===")
    result = await server.repo_search(
        query="lang:python file:scripts/ watch_index",
        limit=5,
        include_snippet=True,
        context_lines=2,
        highlight_snippet=True,
    )
    compact = {
        "ok": result.get("ok"),
        "code": result.get("code"),
        "args": result.get("args"),
        "result_count": len(result.get("results", [])),
        "top_result": result.get("results", [])[0] if result.get("results") else None,
    }
    print(json.dumps(compact, indent=2))

if __name__ == "__main__":
    asyncio.run(main())

