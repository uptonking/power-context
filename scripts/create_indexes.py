#!/usr/bin/env python3
import os
from qdrant_client import QdrantClient, models

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
from datetime import datetime
try:
    from scripts.workspace_state import update_workspace_state, update_last_activity, get_collection_name
except Exception:
    update_workspace_state = None  # type: ignore
    update_last_activity = None  # type: ignore
    get_collection_name = None  # type: ignore

COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
# Discover workspace path for state updates (allows subdir indexing)
WS_PATH = os.environ.get("INDEX_ROOT") or os.environ.get("WORKSPACE_PATH") or "/work"

# Prefer per-workspace unique collection if none provided
if (COLLECTION == "my-collection") and ('get_collection_name' in globals()) and get_collection_name:
    try:
        COLLECTION = get_collection_name(WS_PATH)
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

# Update workspace state to record collection and activity
try:
    if update_workspace_state:
        update_workspace_state(WS_PATH, {"qdrant_collection": COLLECTION})
    if update_last_activity:
        update_last_activity(
            WS_PATH,
            {
                "timestamp": datetime.now().isoformat(),
                "action": "initialized",
                "file_path": "",
                "details": {"created_indexes": ["metadata.language", "metadata.path_prefix"]},
            },
        )
except Exception:
    pass

info = cli.get_collection(COLLECTION)
print(info.payload_schema)
