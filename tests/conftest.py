import os
import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path so `import scripts...` works locally
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _ensure_mcp_imported():
    """Ensure mcp package is properly imported before any tests run.

    This prevents import conflicts when scripts.mcp_indexer_server imports
    from mcp.server.fastmcp and later tests try to import fastmcp.
    """
    try:
        import mcp.types  # noqa: F401
    except ImportError:
        pass  # mcp package not available, tests will skip if needed
    yield


@pytest.fixture(scope="session", autouse=True)
def _disable_tokenizers_parallelism():
    """Force tokenizers to stay single-threaded to avoid fork warnings during tests."""
    prev = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = prev
