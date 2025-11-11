#!/usr/bin/env python3
"""
Workspace state management for .codebase/state.json files.

This module provides functionality to track workspace-specific state including:
- Collection information and indexing status
- Progress tracking during indexing operations
- Activity logging with structured metadata
- Multi-repo support with per-repo state files
"""
import json
import os
import uuid
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, TypedDict
import threading

# Type definitions
IndexingState = Literal['idle', 'initializing', 'scanning', 'indexing', 'watching', 'error']
ActivityAction = Literal['indexed', 'deleted', 'skipped', 'scan-completed', 'initialized', 'moved']

# Constants
STATE_DIRNAME = ".codebase"
STATE_FILENAME = "state.json"
CACHE_FILENAME = "cache.json"
PLACEHOLDER_COLLECTION_NAMES = {"", "default-collection", "my-collection"}

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

class OriginInfo(TypedDict, total=False):
    repo_name: Optional[str]
    container_path: Optional[str]
    source_path: Optional[str]
    collection_name: Optional[str]
    updated_at: Optional[str]


class WorkspaceState(TypedDict, total=False):
    created_at: str
    updated_at: str
    qdrant_collection: str
    indexing_status: Optional[IndexingStatus]
    last_activity: Optional[LastActivity]
    qdrant_stats: Optional[Dict[str, Any]]
    origin: Optional[OriginInfo]

def is_multi_repo_mode() -> bool:
    """Check if multi-repo mode is enabled."""
    return os.environ.get("MULTI_REPO_MODE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

# Simple locking for concurrent access
_state_locks: Dict[str, threading.RLock] = {}

def _resolve_workspace_root() -> str:
    """Determine the default workspace root path."""
    return os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work"

def _resolve_repo_context(
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Normalize workspace/repo context, ensuring multi-repo callers map to repo state."""
    resolved_workspace = workspace_path or _resolve_workspace_root()

    if is_multi_repo_mode():
        if repo_name:
            return resolved_workspace, repo_name

        if workspace_path:
            detected = _detect_repo_name_from_path(Path(workspace_path))
            if detected:
                return resolved_workspace, detected

        return resolved_workspace, None

    return resolved_workspace, repo_name

def _get_state_lock(workspace_path: Optional[str] = None, repo_name: Optional[str] = None) -> threading.RLock:
    """Get or create a lock for the workspace or repo state."""
    if workspace_path:
        key = str(Path(workspace_path).resolve())
    elif repo_name:
        key = f"repo::{repo_name}"
    else:
        key = str(Path(_resolve_workspace_root()).resolve())

    if key not in _state_locks:
        _state_locks[key] = threading.RLock()
    return _state_locks[key]

def _get_repo_state_dir(repo_name: str) -> Path:
    """Get the state directory for a repository."""
    # Use workspace root (typically /work in containers) not script directory
    base_dir = Path(os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work")
    if is_multi_repo_mode():
        return base_dir / STATE_DIRNAME / "repos" / repo_name
    return base_dir / STATE_DIRNAME

def _get_state_path(workspace_path: str) -> Path:
    """Get the path to the state.json file for a workspace."""
    workspace = Path(workspace_path).resolve()
    state_dir = workspace / STATE_DIRNAME
    return state_dir / STATE_FILENAME

def get_workspace_state(workspace_path: Optional[str] = None, repo_name: Optional[str] = None) -> WorkspaceState:
    """Get the current workspace state, creating it if it doesn't exist."""
    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping state read for workspace={workspace_path} without repo_name"
        )
        return {}

    lock = _get_state_lock(workspace_path, repo_name)
    with lock:
        # In multi-repo mode, use repo-based state path
        if is_multi_repo_mode() and repo_name:
            state_path = _get_repo_state_dir(repo_name) / STATE_FILENAME
        else:
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
        state = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "qdrant_collection": get_collection_name(repo_name),
            "indexing_status": {"state": "idle"},
        }

        # Write state
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        return state

def update_workspace_state(
    workspace_path: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
    repo_name: Optional[str] = None,
) -> WorkspaceState:
    """Update workspace state with new values."""
    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)
    updates = updates or {}

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping state update for workspace={workspace_path} without repo_name"
        )
        return {}

    lock = _get_state_lock(workspace_path, repo_name)
    with lock:
        state = get_workspace_state(workspace_path, repo_name)
        state.update(updates)
        state["updated_at"] = datetime.now().isoformat()

        # Write updated state using same path logic as get_workspace_state
        if is_multi_repo_mode() and repo_name:
            state_path = _get_repo_state_dir(repo_name) / STATE_FILENAME
        else:
            state_path = _get_state_path(workspace_path)

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        return state

def update_indexing_status(
    workspace_path: Optional[str] = None,
    status: Optional[IndexingStatus] = None,
    repo_name: Optional[str] = None,
) -> WorkspaceState:
    """Update indexing status in workspace state."""
    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping indexing status update for workspace={workspace_path} without repo_name"
        )
        return {}

    if status is None:
        status = {"state": "idle"}

    return update_workspace_state(
        workspace_path=workspace_path,
        updates={"indexing_status": status},
        repo_name=repo_name,
    )


def update_repo_origin(
    workspace_path: Optional[str] = None,
    repo_name: Optional[str] = None,
    *,
    container_path: Optional[str] = None,
    source_path: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> WorkspaceState:
    """Update origin metadata for a repository/workspace."""

    resolved_workspace, resolved_repo = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and resolved_repo is None:
        return {}

    state = get_workspace_state(resolved_workspace, resolved_repo)
    if not state:
        state = {}

    origin: OriginInfo = dict(state.get("origin", {}))  # type: ignore[arg-type]
    if resolved_repo:
        origin["repo_name"] = resolved_repo
    if container_path or workspace_path:
        origin["container_path"] = container_path or workspace_path
    if source_path:
        origin["source_path"] = source_path
    if collection_name:
        origin["collection_name"] = collection_name
    origin["updated_at"] = datetime.now().isoformat()

    updates: Dict[str, Any] = {"origin": origin}
    if collection_name:
        updates.setdefault("qdrant_collection", collection_name)

    return update_workspace_state(
        workspace_path=resolved_workspace,
        updates=updates,
        repo_name=resolved_repo,
    )

def log_activity(repo_name: Optional[str] = None, action: Optional[ActivityAction] = None,
               file_path: Optional[str] = None, details: Optional[ActivityDetails] = None) -> None:
    """Log activity to workspace state."""
    if not action:
        return

    activity = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "file_path": file_path,
        "details": details or {}
    }

    if is_multi_repo_mode() and repo_name:
        # Multi-repo mode: use repo-based state
        state_dir = _get_repo_state_dir(repo_name)
        state_path = state_dir / STATE_FILENAME

        state_path.parent.mkdir(parents=True, exist_ok=True)

        if state_path.exists():
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError):
                state = {"created_at": datetime.now().isoformat()}
        else:
            state = {"created_at": datetime.now().isoformat()}

        state["last_activity"] = activity
        state["updated_at"] = datetime.now().isoformat()

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    else:
        # Single-repo mode: use workspace-based state (not implemented here)
        pass

def _generate_collection_name_from_repo(repo_name: str) -> str:
    """Generate collection name with 8-char hash for local workspaces.

    Used by local indexer/watcher. Remote uploads use 16+8 char pattern
    for collision avoidance when folder names may be identical.
    """
    hash_obj = hashlib.sha256(repo_name.encode())
    short_hash = hash_obj.hexdigest()[:8]
    return f"{repo_name}-{short_hash}"

def get_collection_name(repo_name: Optional[str] = None) -> str:
    """Get collection name for repository or workspace."""
    # In multi-repo mode, prioritize repo-specific collection names
    if is_multi_repo_mode() and repo_name:
        return _generate_collection_name_from_repo(repo_name)

    # Check environment for single-repo mode or fallback
    env_coll = os.environ.get("COLLECTION_NAME", "").strip()
    if env_coll and env_coll not in PLACEHOLDER_COLLECTION_NAMES:
        return env_coll

    # Use repo name if provided (for single-repo mode with repo name)
    if repo_name:
        return _generate_collection_name_from_repo(repo_name)

    # Default fallback
    return "global-collection"

def _detect_repo_name_from_path(path: Path) -> str:
    """Detect repository name from path. Clean, robust implementation."""
    try:
        resolved_path = path.resolve()
    except Exception:
        return None

    candidate_roots: List[Path] = []
    for root_str in (
        os.environ.get("WATCH_ROOT"),
        os.environ.get("WORKSPACE_PATH"),
        "/work",
        os.environ.get("HOST_ROOT"),
        "/home/coder/project/Context-Engine/dev-workspace",
    ):
        if not root_str:
            continue
        try:
            root_path = Path(root_str).resolve()
        except Exception:
            continue
        if root_path not in candidate_roots:
            candidate_roots.append(root_path)

    for base in candidate_roots:
        try:
            rel_path = resolved_path.relative_to(base)
        except ValueError:
            continue

        if not rel_path.parts:
            continue

        repo_name = rel_path.parts[0]
        if repo_name in (".codebase", ".git", "__pycache__"):
            continue

        repo_path = base / repo_name
        if repo_path.exists() or str(resolved_path).startswith(str(repo_path) + os.sep):
            return repo_name

    return None

def _extract_repo_name_from_path(workspace_path: str) -> str:
    """Extract repository name from workspace path."""
    return _detect_repo_name_from_path(Path(workspace_path))

# Cache functions for file hash tracking
def _get_cache_path(workspace_path: str) -> Path:
    """Get the path to the cache.json file."""
    workspace = Path(workspace_path).resolve()
    return workspace / STATE_DIRNAME / CACHE_FILENAME

def _read_cache(workspace_path: str) -> Dict[str, Any]:
    """Read cache file, return empty dict if doesn't exist."""
    cache_path = _get_cache_path(workspace_path)
    if not cache_path.exists():
        return {"file_hashes": {}, "updated_at": datetime.now().isoformat()}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("file_hashes"), dict):
                return obj
            return {"file_hashes": {}, "updated_at": datetime.now().isoformat()}
    except Exception:
        return {"file_hashes": {}, "updated_at": datetime.now().isoformat()}

def _write_cache(workspace_path: str, cache: Dict[str, Any]) -> None:
    """Write cache file atomically."""
    cache_path = _get_cache_path(workspace_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        tmp.replace(cache_path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

def get_cached_file_hash(file_path: str, repo_name: Optional[str] = None) -> str:
    """Get cached file hash for tracking changes."""
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    file_hashes = cache.get("file_hashes", {})
                    return file_hashes.get(str(Path(file_path).resolve()), "")
            except Exception:
                pass

    return ""

def set_cached_file_hash(file_path: str, file_hash: str, repo_name: Optional[str] = None) -> None:
    """Set cached file hash for tracking changes."""
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            else:
                cache = {"file_hashes": {}, "created_at": datetime.now().isoformat()}

            cache.setdefault("file_hashes", {})[str(Path(file_path).resolve())] = file_hash
            cache["updated_at"] = datetime.now().isoformat()

            # Atomic write
            tmp = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            tmp.replace(cache_path)
        except Exception:
            pass

def remove_cached_file(file_path: str, repo_name: Optional[str] = None) -> None:
    """Remove file entry from cache."""
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    file_hashes = cache.get("file_hashes", {})

                fp = str(Path(file_path).resolve())
                if fp in file_hashes:
                    file_hashes.pop(fp, None)
                    cache["updated_at"] = datetime.now().isoformat()

                    tmp = cache_path.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)
                    tmp.replace(cache_path)
            except Exception:
                pass

# Additional functions needed by callers
def _state_file_path(workspace_path: Optional[str] = None, repo_name: Optional[str] = None) -> Path:
    """Get state file path for workspace or repo."""
    if repo_name and is_multi_repo_mode():
        state_dir = _get_repo_state_dir(repo_name)
        return state_dir / STATE_FILENAME

    if workspace_path:
        return _get_state_path(workspace_path)

    # Default to current directory
    return Path.cwd() / STATE_DIRNAME / STATE_FILENAME

def _get_global_state_dir() -> Path:
    """Get the global .codebase directory."""
    base_dir = Path.cwd()
    return base_dir / ".codebase"

def list_workspaces(search_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find all workspaces with .codebase/state.json files."""
    if search_root is None:
        # Use workspace root instead of current directory
        search_root = os.environ.get("WORKSPACE_PATH") or "/work"

    workspaces = []
    root_path = Path(search_root)

    # Look for state files
    for state_file in root_path.rglob(STATE_FILENAME):
        try:
            rel_path = state_file.relative_to(root_path)
            workspace_info = {
                "path": str(rel_path.parent),
                "state_file": str(state_file),
                "relative_path": str(rel_path.parent)
            }
            workspaces.append(workspace_info)
        except (ValueError, OSError):
            continue

    return workspaces


def get_collection_mappings(search_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Enumerate collection mappings with origin metadata."""

    root_path = Path(search_root or _resolve_workspace_root()).resolve()
    mappings: List[Dict[str, Any]] = []

    try:
        if is_multi_repo_mode():
            repos_root = root_path / STATE_DIRNAME / "repos"
            if repos_root.exists():
                for repo_dir in sorted(p for p in repos_root.iterdir() if p.is_dir()):
                    repo_name = repo_dir.name
                    state_path = repo_dir / STATE_FILENAME
                    if not state_path.exists():
                        continue
                    try:
                        with open(state_path, "r", encoding="utf-8") as f:
                            state = json.load(f) or {}
                    except Exception:
                        continue

                    origin = state.get("origin", {}) or {}
                    mappings.append(
                        {
                            "repo_name": repo_name,
                            "collection_name": state.get("qdrant_collection")
                            or get_collection_name(repo_name),
                            "container_path": origin.get("container_path")
                            or str((Path(_resolve_workspace_root()) / repo_name).resolve()),
                            "source_path": origin.get("source_path"),
                            "state_file": str(state_path),
                            "updated_at": state.get("updated_at"),
                        }
                    )
        else:
            state_path = root_path / STATE_DIRNAME / STATE_FILENAME
            if state_path.exists():
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f) or {}
                except Exception:
                    state = {}

                origin = state.get("origin", {}) or {}
                repo_name = origin.get("repo_name") or Path(root_path).name
                mappings.append(
                    {
                        "repo_name": repo_name,
                        "collection_name": state.get("qdrant_collection")
                        or get_collection_name(repo_name),
                        "container_path": origin.get("container_path")
                        or str(root_path),
                        "source_path": origin.get("source_path"),
                        "state_file": str(state_path),
                        "updated_at": state.get("updated_at"),
                    }
                )
    except Exception:
        return mappings

    return mappings

# Add missing functions that callers expect (already defined above)