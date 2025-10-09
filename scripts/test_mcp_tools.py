#!/usr/bin/env python3
import os, json, asyncio, importlib, sys
from pathlib import Path

# Ensure repo root is on sys.path when executing from /work/scripts
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

mod = importlib.import_module('scripts.mcp_indexer_server')

async def run():
    print("Env:", {
        'QDRANT_URL': os.environ.get('QDRANT_URL'),
        'COLLECTION_NAME': os.environ.get('COLLECTION_NAME'),
        'REPO_NAME': os.environ.get('REPO_NAME'),
    })

    print("\n=== qdrant_list ===")
    cols = await mod.qdrant_list()
    print(json.dumps(cols, indent=2))

    print("\n=== qdrant_index (subdir=scripts) ===")
    idx_res = await mod.qdrant_index(subdir='scripts', recreate=False)
    print(json.dumps({k: idx_res.get(k) for k in ('ok','code')}, indent=2))

    print("\n=== repo_search (DSL: lang:python file:scripts/) ===")
    res = await mod.repo_search(
        query='lang:python file:scripts/ hybrid search',
        limit=8,
        include_snippet=True,
        highlight_snippet=True,
        context_lines=2,
    )
    # Print compact
    items = (res or {}).get('results', [])[:5]
    out = {
        'used_rerank': (res or {}).get('used_rerank'),
        'count': len(items),
        'items': [
            {
                'score': round(float(i.get('score', 0.0)), 4),
                'path': i.get('path'),
                'symbol': i.get('symbol'),
                'why': i.get('why', [])[:4],
            } for i in items
        ]
    }
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    asyncio.run(run())

