#!/usr/bin/env python3
"""
Workspace state management for .codebase/state.json files.

This module provides functionality to track workspace-specific state including:
- Collection information and indexing status
- Progress tracking during indexing operations
- Activity logging with structured metadata
- Multi-project support with per-workspace state files

Based on the codebase-index-cli workspace state pattern but adapted for our Python ecosystem.
"""
import json
import os
import uuid
import re
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, TypedDict
import threading

# Type definitions matching codebase-index-cli patterns
IndexingState = Literal['idle', 'initializing', 'scanning', 'indexing', 'watching', 'error']
ActivityAction = Literal['indexed', 'deleted', 'skipped', 'scan-completed', 'initialized', 'moved']

class IndexingProgress(TypedDict, total=False):
    files_processed: int
    total_files: Optional[int]
    current_file: Optional[str]

class IndexingStatus(TypedDict, total=False):
    state: IndexingState
    started_at: Optional[str]
    progress: Optional[IndexingProgress]
    error: Optional[str]

class ActivityDetails(TypedDict, total=False):
    block_count: Optional[int]
    reason: Optional[str]
    files_processed: Optional[int]
    total_blocks: Optional[int]
    git_commit: Optional[str]
    git_branch: Optional[str]
    chunk_count: Optional[int]
    file_size: Optional[int]

class LastActivity(TypedDict, total=False):
    timestamp: str
    action: ActivityAction
    file_path: Optional[str]
    details: Optional[ActivityDetails]

class QdrantStats(TypedDict, total=False):
    total_vectors: int
    unique_files: int
    vector_dimension: int
    last_updated: str
    collection_name: str

class WorkspaceState(TypedDict, total=False):
    workspace_path: str
    created_at: str
    updated_at: str
    qdrant_collection: str
    indexing_status: Optional[IndexingStatus]
    last_activity: Optional[LastActivity]
    qdrant_stats: Optional[QdrantStats]

# Constants
STATE_DIRNAME = ".codebase"
STATE_FILENAME = "state.json"

# Thread-safe state management
_state_locks: Dict[str, threading.Lock] = {}
_state_lock = threading.Lock()

def _get_state_lock(workspace_path: str) -> threading.Lock:
    """Get or create a thread-safe lock for a specific workspace."""
    with _state_lock:
        if workspace_path not in _state_locks:
            _state_locks[workspace_path] = threading.Lock()
        return _state_locks[workspace_path]

def _get_state_path(workspace_path: str) -> Path:
    """Get the path to the state.json file for a workspace."""
    workspace = Path(workspace_path).resolve()
    state_dir = workspace / STATE_DIRNAME
    return state_dir / STATE_FILENAME

def _ensure_state_dir(workspace_path: str) -> Path:
    """Ensure the .codebase directory exists and return the state file path."""
    workspace = Path(workspace_path).resolve()
    state_dir = workspace / STATE_DIRNAME
    state_dir.mkdir(exist_ok=True)
    return state_dir / STATE_FILENAME

def _sanitize_name(s: str, max_len: int = 64) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9_.-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "workspace"
    return s[:max_len]


def _detect_repo_name_from_path(path: Path) -> str:
    try:
        base = path if path.is_dir() else path.parent
        r = subprocess.run(["git", "-C", str(base), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        top = (r.stdout or "").strip()
        if r.returncode == 0 and top:
            return Path(top).name
    except Exception:
        pass
    try:
        # Walk up to find .git
        cur = path if path.is_dir() else path.parent
        for p in [cur] + list(cur.parents):
            try:
                if (p / ".git").exists():
                    return p.name
            except Exception:
                continue
    except Exception:
        pass
    return (path if path.is_dir() else path.parent).name or "workspace"


def _generate_collection_name(workspace_path: str) -> str:
    ws = Path(workspace_path).resolve()
    repo = _sanitize_name(_detect_repo_name_from_path(ws))
    # stable suffix from absolute path
    h = hashlib.sha1(str(ws).encode("utf-8", errors="ignore")).hexdigest()[:6]
    return _sanitize_name(f"{repo}-{h}")

def _atomic_write_state(state_path: Path, state: WorkspaceState) -> None:
    """Atomically write state to prevent corruption during concurrent access."""
    # Write to temp file first, then rename (atomic on most filesystems)
    temp_path = state_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        temp_path.replace(state_path)
    except Exception:
        # Clean up temp file if something went wrong
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def get_workspace_state(workspace_path: str) -> WorkspaceState:
    """Get the current workspace state, creating it if it doesn't exist."""
    lock = _get_state_lock(workspace_path)
    with lock:
        state_path = _get_state_path(workspace_path)

        if state_path.exists():
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    # Ensure required fields exist
                    if not isinstance(state, dict):
                        raise ValueError("Invalid state format")
                    return state
            except (json.JSONDecodeError, ValueError, OSError):
                # Corrupted or invalid state file, recreate
                pass

        # Create new state
        now = datetime.now().isoformat()
        env_coll = os.environ.get("COLLECTION_NAME")
        if isinstance(env_coll, str) and env_coll.strip() and env_coll.strip() != "my-collection":
            collection_name = env_coll.strip()
        else:
            collection_name = _generate_collection_name(workspace_path)

        state: WorkspaceState = {
            "workspace_path": str(Path(workspace_path).resolve()),
            "created_at": now,
            "updated_at": now,
            "qdrant_collection": collection_name,
            "indexing_status": {
                "state": "idle"
            }
        }

        # Ensure directory exists and write state
        state_path = _ensure_state_dir(workspace_path)
        _atomic_write_state(state_path, state)
        return state

def update_workspace_state(workspace_path: str, updates: Dict[str, Any]) -> WorkspaceState:
    """Update workspace state with the given changes."""
    lock = _get_state_lock(workspace_path)
    with lock:
        state = get_workspace_state(workspace_path)

        # Apply updates
        for key, value in updates.items():
            if key in state or key in WorkspaceState.__annotations__:
                state[key] = value

        # Always update timestamp
        state["updated_at"] = datetime.now().isoformat()

        # Write back to file
        state_path = _ensure_state_dir(workspace_path)
        _atomic_write_state(state_path, state)
        return state

def update_indexing_status(workspace_path: str, status: IndexingStatus) -> WorkspaceState:
    """Update the indexing status in workspace state."""
    return update_workspace_state(workspace_path, {"indexing_status": status})

def update_last_activity(workspace_path: str, activity: LastActivity) -> WorkspaceState:
    """Update the last activity in workspace state."""
    return update_workspace_state(workspace_path, {"last_activity": activity})

def update_qdrant_stats(workspace_path: str, stats: QdrantStats) -> WorkspaceState:
    """Update Qdrant statistics in workspace state."""
    stats["last_updated"] = datetime.now().isoformat()
    return update_workspace_state(workspace_path, {"qdrant_stats": stats})

def get_collection_name(workspace_path: str) -> str:
    """Get the Qdrant collection name for a workspace.
    If none is present in state, persist either COLLECTION_NAME from env or a generated
    repoName-<shortHash> based on the workspace path, and return it.
    """
    state = get_workspace_state(workspace_path)
    coll = state.get("qdrant_collection") if isinstance(state, dict) else None
    if isinstance(coll, str) and coll.strip():
        return coll.strip()
    # Prefer explicit env override if provided
    env_coll = os.environ.get("COLLECTION_NAME")
    if env_coll and env_coll.strip():
        coll = env_coll.strip()
    else:
        coll = _generate_collection_name(workspace_path)
    update_workspace_state(workspace_path, {"qdrant_collection": coll})
    return coll

def list_workspaces(search_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find all workspaces with .codebase/state.json files."""
    if search_root is None:
        search_root = os.getcwd()

    workspaces = []
    search_path = Path(search_root).resolve()

    # Search for .codebase directories
    for state_dir in search_path.rglob(STATE_DIRNAME):
        state_file = state_dir / STATE_FILENAME
        if state_file.exists():
            try:
                workspace_path = str(state_dir.parent)
                state = get_workspace_state(workspace_path)
                workspaces.append({
                    "workspace_path": workspace_path,
                    "collection_name": state.get("qdrant_collection"),
                    "last_updated": state.get("updated_at"),
                    "indexing_state": state.get("indexing_status", {}).get("state", "unknown")
                })
            except Exception:
                # Skip corrupted state files
                continue

    return sorted(workspaces, key=lambda x: x.get("last_updated", ""), reverse=True)

def cleanup_old_state_locks():
    """Clean up unused state locks to prevent memory leaks."""
    with _state_lock:
        # Keep only locks for recently accessed workspaces
        # In practice, this would need more sophisticated cleanup logic
        pass

if __name__ == "__main__":
    # Simple CLI for testing
    import sys
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
        state = get_workspace_state(workspace)
        print(json.dumps(state, indent=2))
    else:
        workspaces = list_workspaces()
        for ws in workspaces:
            print(f"{ws['workspace_path']}: {ws['collection_name']} ({ws['indexing_state']})")
