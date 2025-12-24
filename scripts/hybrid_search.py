#!/usr/bin/env python3
"""
Backward-compatible shim for hybrid search.

All functionality has moved to scripts/hybrid/ package.
This shim re-exports everything for existing imports like:
    from scripts.hybrid_search import run_hybrid_search
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path for CLI execution
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.hybrid import *  # noqa: F401, F403

if __name__ == "__main__":
    main()
