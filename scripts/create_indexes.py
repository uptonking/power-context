#!/usr/bin/env python3
import os
from qdrant_client import QdrantClient, models

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
from datetime import datetime
# Import critical functions first
try:
    from scripts.workspace_state import get_collection_name, is_multi_repo_mode
except Exception:
    get_collection_name = None  # type: ignore
    is_multi_repo_mode = None  # type: ignore

# Import other optional functions
try:
    from scripts.workspace_state import log_activity
except Exception:
    log_activity = None  # type: ignore

COLLECTION = os.environ.get("COLLECTION_NAME", "codebase")
# Discover workspace path for state updates (allows subdir indexing)
WS_PATH = os.environ.get("INDEX_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"

# Skip creating root collection in multi-repo mode when indexing entire /work tree
if is_multi_repo_mode and is_multi_repo_mode() and WS_PATH == "/work":
    print("Multi-repo mode enabled - skipping root collection creation for /work")
    exit(0)

# Prefer workspace-derived collection names when env value is a placeholder
if 'get_collection_name' in globals() and get_collection_name:
    try:
        resolved = get_collection_name(None)
        if resolved:
            placeholders = {"", "default-collection", "my-collection", "codebase"}
            if COLLECTION in placeholders:
                COLLECTION = resolved
    except Exception:
        pass


cli = QdrantClient(url=QDRANT_URL)

# Create keyword indexes for metadata fields
cli.create_payload_index(
    collection_name=COLLECTION,
    field_name="metadata.language",
    field_schema=models.PayloadSchemaType.KEYWORD,
)
cli.create_payload_index(
    collection_name=COLLECTION,
    field_name="metadata.path_prefix",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

# Log activity using cleaned workspace_state function
try:
    if log_activity:
        log_activity(
            repo_name=None,
            action="initialized",
            file_path="",
            details={"created_indexes": ["metadata.language", "metadata.path_prefix"]},
        )
except Exception:
    pass

info = cli.get_collection(COLLECTION)
print(info.payload_schema)
