#!/usr/bin/env python3
import os, json, asyncio, importlib, sys
from pathlib import Path
# Ensure /work (repo root) is on sys.path even when running from /work/scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

mod = importlib.import_module('scripts.mcp_indexer_server')

async def run_case(title, **kwargs):
    print(f"\n=== {title} ===")
    res = await mod.repo_search(**kwargs)
    results = (res or {}).get('results', [])[:5]
    print(json.dumps({
        'used_rerank': (res or {}).get('used_rerank'),
        'count': len(results),
        'items': [
            {
                'score': round(float(i.get('score', 0.0)), 4),
                'path': i.get('path'),
                'symbol': i.get('symbol'),
                'why': i.get('why', [])[:5],
            } for i in results
        ]
    }, indent=2))

async def main():
    await run_case(
        'DSL: python scripts with watch_index',
        query='lang:python file:scripts/ watch_index',
        limit=8,
        include_snippet=True,
        highlight_snippet=True,
        context_lines=2,
    )
    await run_case(
        'Rerank enabled: hybrid + cross-encoder',
        query='lang:python file:scripts/ hybrid search',
        limit=8,
        rerank_enabled=True,
        rerank_top_n=40,
        rerank_return_m=8,
        rerank_timeout_ms=10000,
        include_snippet=True,
    )
    await run_case(
        'Symbol exact: symbol:watch_index',
        query='symbol:watch_index lang:python',
        limit=5,
        include_snippet=True,
    )

if __name__ == '__main__':
    asyncio.run(main())

