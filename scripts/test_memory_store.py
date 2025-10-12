import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure module path includes /work
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure env mirrors compose defaults
os.environ.setdefault("QDRANT_URL", "http://qdrant:6333")
os.environ.setdefault("COLLECTION_NAME", os.environ.get("COLLECTION_NAME", "my-collection"))

from scripts import mcp_indexer_server as mod


async def main():
    print("=== memory_store smoke ===", flush=True)
    res = await mod.memory_store("Test memory entry from CI", {"kind": "preference", "source": "memory"})
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

