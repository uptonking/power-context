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
import re
import uuid
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal, TypedDict
import threading
import time

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
    logical_repo_id: Optional[str]

def is_multi_repo_mode() -> bool:
    """Check if multi-repo mode is enabled."""
    return os.environ.get("MULTI_REPO_MODE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def logical_repo_reuse_enabled() -> bool:
    """Feature flag for logical-repo / collection reuse.

    Controlled by LOGICAL_REPO_REUSE env var: 1/true/yes/on => enabled.
    When disabled, behavior falls back to legacy per-repo collection logic
    and does not write logical_repo_id into workspace state.
    """
    return os.environ.get("LOGICAL_REPO_REUSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

_state_lock = threading.Lock()
# Track last-used timestamps for cleanup of idle workspace locks
_state_locks: Dict[str, threading.RLock] = {}
_state_lock_last_used: Dict[str, float] = {}

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
    """Get or create a lock for the workspace or repo state and track usage."""
    if repo_name and is_multi_repo_mode():
        key = f"repo::{repo_name}"
    else:
        key = str(Path(workspace_path or _resolve_workspace_root()).resolve())

    with _state_lock:
        if key not in _state_locks:
            _state_locks[key] = threading.RLock()
        _state_lock_last_used[key] = time.time()
        return _state_locks[key]

def _get_repo_state_dir(repo_name: str) -> Path:
    """Get the state directory for a repository."""
    base_dir = Path(os.environ.get("WORKSPACE_PATH") or os.environ.get("WATCH_ROOT") or "/work")
    if is_multi_repo_mode():
        return base_dir / STATE_DIRNAME / "repos" / repo_name
    return base_dir / STATE_DIRNAME

def _get_state_path(workspace_path: str) -> Path:
    """Get the path to the state.json file for a workspace."""
    workspace = Path(workspace_path).resolve()
    state_dir = workspace / STATE_DIRNAME
    return state_dir / STATE_FILENAME


def _get_global_state_dir(workspace_path: Optional[str] = None) -> Path:
    """Return the root .codebase directory used for workspace metadata."""

    base_dir = Path(workspace_path or _resolve_workspace_root()).resolve()
    return base_dir / STATE_DIRNAME

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


def _detect_git_common_dir(start: Path) -> Optional[Path]:
    try:
        base = start if start.is_dir() else start.parent
        r = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
        )
        raw = (r.stdout or "").strip()
        if r.returncode != 0 or not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        return p.resolve()
    except Exception:
        return None


def compute_logical_repo_id(workspace_path: str) -> str:
    try:
        p = Path(workspace_path).resolve()
    except Exception:
        p = Path(workspace_path)

    common = _detect_git_common_dir(p)
    if common is not None:
        key = str(common)
        prefix = "git:"
    else:
        key = str(p)
        prefix = "fs:"

    h = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}{h}"


def ensure_logical_repo_id(state: WorkspaceState, workspace_path: str) -> WorkspaceState:
    if not isinstance(state, dict):
        return state
    if not logical_repo_reuse_enabled():
        # Gate: when logical repo reuse is disabled, leave state untouched
        return state
    if state.get("logical_repo_id"):
        return state
    lrid = compute_logical_repo_id(workspace_path)
    state["logical_repo_id"] = lrid
    origin = dict(state.get("origin", {}) or {})
    origin.setdefault("logical_repo_id", lrid)
    state["origin"] = origin
    return state


# Cross-process file locking (POSIX fcntl), falls back to no-op if unavailable
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

from contextlib import contextmanager

@contextmanager
def _cross_process_lock(lock_path: Path):
    """Advisory cross-process exclusive lock using a companion .lock file.
    Safe across container/process boundaries; pairs with atomic rename writes.
    Ensures group-writable permissions so non-root indexers/watchers can operate.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    lock_file = None
    fd = None
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)
        lock_file = os.fdopen(fd, "a+")
    except PermissionError:
        # If we cannot create or open the requested lock, fall back to /tmp (permissive)
        tmp_path = Path("/tmp") / (lock_path.name)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(tmp_path, os.O_CREAT | os.O_RDWR, 0o664)
        lock_file = os.fdopen(fd, "a+")
        lock_path = tmp_path

    try:
        try:
            os.chmod(lock_path, 0o664)
        except PermissionError:
            pass

        if fcntl is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        yield
    finally:
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                lock_file.close()
            except Exception:
                pass

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
        # Ensure state/cache files are group-writable so multiple processes
        # (upload service, watcher, indexer) can update them.
        try:
            os.chmod(state_path, 0o664)
        except PermissionError:
            pass
    except Exception:
        # Clean up temp file if something went wrong
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def get_workspace_state(
    workspace_path: Optional[str] = None, repo_name: Optional[str] = None
) -> WorkspaceState:
    """Get the current workspace state, creating it if it doesn't exist."""

    workspace_path, repo_name = _resolve_repo_context(workspace_path, repo_name)

    if is_multi_repo_mode() and repo_name is None:
        print(
            f"[workspace_state] Multi-repo: Skipping state read for workspace={workspace_path} without repo_name"
        )
        return {}

    lock = _get_state_lock(workspace_path, repo_name)
    with lock:
        state_path: Path
        lock_scope_path: Path

        if is_multi_repo_mode() and repo_name:
            state_dir = _get_repo_state_dir(repo_name)
            state_dir.mkdir(parents=True, exist_ok=True)
            # Ensure repo state dir is group-writable so root upload service and
            # non-root watcher/indexer processes can both write state/cache files.
            try:
                os.chmod(state_dir, 0o775)
            except Exception:
                pass
            state_path = state_dir / STATE_FILENAME
            lock_scope_path = state_dir
        else:
            try:
                state_path = _ensure_state_dir(workspace_path)
                lock_scope_path = state_path.parent
            except PermissionError:
                lock_scope_path = _get_global_state_dir(workspace_path)
                lock_scope_path.mkdir(parents=True, exist_ok=True)
                state_path = lock_scope_path / STATE_FILENAME

        lock_path = lock_scope_path / (STATE_FILENAME + ".lock")
        with _cross_process_lock(lock_path):
            if state_path.exists():
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    if isinstance(state, dict):
                        if logical_repo_reuse_enabled():
                            workspace_real = str(Path(workspace_path or _resolve_workspace_root()).resolve())
                            state = ensure_logical_repo_id(state, workspace_real)
                            try:
                                _atomic_write_state(state_path, state)
                            except Exception as e:
                                print(f"[workspace_state] Failed to persist logical_repo_id to {state_path}: {e}")
                        return state
                except (json.JSONDecodeError, ValueError, OSError) as e:
                    print(f"[workspace_state] Failed to read state from {state_path}: {e}")

            now = datetime.now().isoformat()
            collection_name = get_collection_name(repo_name)

            state: WorkspaceState = {
                "workspace_path": str(Path(workspace_path or _resolve_workspace_root()).resolve()),
                "created_at": now,
                "updated_at": now,
                "qdrant_collection": collection_name,
                "indexing_status": {"state": "idle"},
            }

            if logical_repo_reuse_enabled():
                try:
                    state = ensure_logical_repo_id(state, state.get("workspace_path", workspace_path or _resolve_workspace_root()))
                except Exception as e:
                    print(f"[workspace_state] Failed to ensure logical_repo_id for {workspace_path}: {e}")

            _atomic_write_state(state_path, state)
            return state


def update_workspace_state(
    workspace_path: Optional[str] = None,
    updates: Optional[Dict[str, Any]] = None,
    repo_name: Optional[str] = None,
) -> WorkspaceState:
    """Update workspace state with the given changes."""

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
        for key, value in updates.items():
            if key in state or key in WorkspaceState.__annotations__:
                state[key] = value

        state["updated_at"] = datetime.now().isoformat()

        if is_multi_repo_mode() and repo_name:
            state_dir = _get_repo_state_dir(repo_name)
            state_dir.mkdir(parents=True, exist_ok=True)
            state_path = state_dir / STATE_FILENAME
        else:
            try:
                state_path = _ensure_state_dir(workspace_path)
            except PermissionError:
                state_dir = _get_global_state_dir(workspace_path)
                state_dir.mkdir(parents=True, exist_ok=True)
                state_path = state_dir / STATE_FILENAME

        _atomic_write_state(state_path, state)
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


def log_activity(
    repo_name: Optional[str] = None,
    action: Optional[ActivityAction] = None,
    file_path: Optional[str] = None,
    details: Optional[ActivityDetails] = None,
    workspace_path: Optional[str] = None,
) -> None:
    """Log activity to workspace state."""

    if not action:
        return

    activity = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "file_path": file_path,
        "details": details or {},
    }

    resolved_workspace = workspace_path or _resolve_workspace_root()

    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / STATE_FILENAME
        lock_path = state_path.with_suffix(".lock")

        with _cross_process_lock(lock_path):
            try:
                if state_path.exists():
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f)
                else:
                    state = {"created_at": datetime.now().isoformat()}
            except Exception:
                state = {"created_at": datetime.now().isoformat()}

            state["last_activity"] = activity
            state["updated_at"] = datetime.now().isoformat()
            _atomic_write_state(state_path, state)
    else:
        update_workspace_state(
            workspace_path=resolved_workspace,
            updates={"last_activity": activity},
            repo_name=repo_name,
        )


def _generate_collection_name_from_repo(repo_name: str) -> str:
    """Generate collection name with 8-char hash for local workspaces.

    Used by local indexer/watcher. Remote uploads use 16+8 char pattern
    for collision avoidance when folder names may be identical.
    """
    hash_obj = hashlib.sha256(repo_name.encode())
    short_hash = hash_obj.hexdigest()[:8]
    return f"{repo_name}-{short_hash}"

def _normalize_repo_name_for_collection(repo_name: str) -> str:
    """Normalize repo identifier to a stable base name for collection naming.

    In multi-repo remote mode, repo_name may be a slug like "name-<16hex>" used
    for folder collision avoidance. For Qdrant collections we always want the
    base repo directory name, so strip a trailing 16-hex segment when present.
    """
    try:
        m = re.match(r"^(.*)-([0-9a-f]{16})$", repo_name)
        if m:
            base = (m.group(1) or "").strip()
            if base:
                return base
    except Exception:
        pass
    return repo_name


def get_collection_name(repo_name: Optional[str] = None) -> str:
    """Get collection name for repository or workspace."""
    normalized = _normalize_repo_name_for_collection(repo_name) if repo_name else None

    # In multi-repo mode, prioritize repo-specific collection names
    if is_multi_repo_mode() and normalized:
        return _generate_collection_name_from_repo(normalized)

    # Check environment for single-repo mode or fallback
    env_coll = os.environ.get("COLLECTION_NAME", "").strip()
    if env_coll and env_coll not in PLACEHOLDER_COLLECTION_NAMES:
        return env_coll

    # Use repo name if provided (for single-repo mode with repo name)
    if normalized:
        return _generate_collection_name_from_repo(normalized)

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
    """Read cache file, return empty dict if it doesn't exist or is invalid."""

    cache_path = _get_cache_path(workspace_path)
    if not cache_path.exists():
        return {"file_hashes": {}, "updated_at": datetime.now().isoformat()}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("file_hashes"), dict):
                return obj
    except Exception:
        pass

    return {"file_hashes": {}, "updated_at": datetime.now().isoformat()}


def _write_cache(workspace_path: str, cache: Dict[str, Any]) -> None:
    """Atomic write of cache file with cross-process locking."""

    lock = _get_state_lock(workspace_path)
    with lock:
        cache_path = _get_cache_path(workspace_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with _cross_process_lock(lock_path):
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
                    fp = str(Path(file_path).resolve())
                    val = file_hashes.get(fp, "")
                    if isinstance(val, dict):
                        return str(val.get("hash") or "")
                    return str(val or "")
            except Exception:
                pass
    else:
        cache = _read_cache(_resolve_workspace_root())
        fp = str(Path(file_path).resolve())
        val = cache.get("file_hashes", {}).get(fp, "")
        if isinstance(val, dict):
            return str(val.get("hash") or "")
        return str(val or "")

    return ""


def set_cached_file_hash(file_path: str, file_hash: str, repo_name: Optional[str] = None) -> None:
    """Set cached file hash for tracking changes."""

    fp = str(Path(file_path).resolve())

    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME
        state_dir.mkdir(parents=True, exist_ok=True)

        try:
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache = json.load(f)
                except Exception:
                    # If the existing cache is corrupt/empty, recreate it
                    cache = {"file_hashes": {}, "created_at": datetime.now().isoformat()}
            else:
                cache = {"file_hashes": {}, "created_at": datetime.now().isoformat()}

            entry: Any = file_hash
            try:
                st = Path(file_path).stat()
                entry = {
                    "hash": file_hash,
                    "size": int(getattr(st, "st_size", 0)),
                    "mtime": int(getattr(st, "st_mtime", 0)),
                }
            except Exception:
                pass

            cache.setdefault("file_hashes", {})[fp] = entry
            cache["updated_at"] = datetime.now().isoformat()

            _atomic_write_state(cache_path, cache)  # reuse atomic writer for files
        except Exception:
            pass
        return

    cache = _read_cache(_resolve_workspace_root())
    entry: Any = file_hash
    try:
        st = Path(file_path).stat()
        entry = {
            "hash": file_hash,
            "size": int(getattr(st, "st_size", 0)),
            "mtime": int(getattr(st, "st_mtime", 0)),
        }
    except Exception:
        pass
    cache.setdefault("file_hashes", {})[fp] = entry
    cache["updated_at"] = datetime.now().isoformat()
    _write_cache(_resolve_workspace_root(), cache)


def get_cached_file_meta(file_path: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
    fp = str(Path(file_path).resolve())
    if is_multi_repo_mode() and repo_name:
        state_dir = _get_repo_state_dir(repo_name)
        cache_path = state_dir / CACHE_FILENAME

        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    file_hashes = cache.get("file_hashes", {})
                    val = file_hashes.get(fp)
            except Exception:
                val = None
        else:
            val = None
    else:
        cache = _read_cache(_resolve_workspace_root())
        val = cache.get("file_hashes", {}).get(fp)

    if isinstance(val, dict):
        return {
            "hash": str(val.get("hash") or ""),
            "size": val.get("size"),
            "mtime": val.get("mtime"),
        }
    if isinstance(val, str):
        return {"hash": val}
    return {}


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

                    _atomic_write_state(cache_path, cache)
            except Exception:
                pass
        return

    cache = _read_cache(_resolve_workspace_root())
    fp = str(Path(file_path).resolve())
    if fp in cache.get("file_hashes", {}):
        cache["file_hashes"].pop(fp, None)
        cache["updated_at"] = datetime.now().isoformat()
        _write_cache(_resolve_workspace_root(), cache)


def cleanup_old_cache_locks(max_idle_seconds: int = 900) -> int:
    """Best-effort cleanup of idle cache locks.

    Removes locks that have been idle (not requested via _get_state_lock) for longer than max_idle_seconds
    and whose lock can be acquired without blocking (i.e., not held).
    Returns the number of locks removed.
    """
    now = time.time()
    removed = 0
    with _state_lock:
        stale_keys = []
        for ws, lock in list(_state_locks.items()):
            last = _state_lock_last_used.get(ws, 0.0)
            # Prefer also pruning locks whose workspace no longer exists
            ws_exists = True
            try:
                ws_exists = Path(ws).exists()
            except Exception:
                ws_exists = False
            if (now - last) > max_idle_seconds or not ws_exists:
                acquired = False
                try:
                    acquired = lock.acquire(blocking=False)
                except Exception:
                    acquired = False
                if acquired:
                    try:
                        stale_keys.append(ws)
                    finally:
                        try:
                            lock.release()
                        except Exception:
                            pass
        for ws in stale_keys:
            _state_locks.pop(ws, None)
            _state_lock_last_used.pop(ws, None)
            removed += 1
    return removed


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
                    except Exception as e:
                        print(f"[workspace_state] Failed to read repo state from {state_path}: {e}")
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


def find_collection_for_logical_repo(logical_repo_id: str, search_root: Optional[str] = None) -> Optional[str]:
    if not logical_repo_reuse_enabled():
        return None

    root_path = Path(search_root or _resolve_workspace_root()).resolve()

    try:
        if is_multi_repo_mode():
            repos_root = root_path / STATE_DIRNAME / "repos"
            if repos_root.exists():
                for repo_dir in repos_root.iterdir():
                    if not repo_dir.is_dir():
                        continue
                    state_path = repo_dir / STATE_FILENAME
                    if not state_path.exists():
                        continue
                    try:
                        with open(state_path, "r", encoding="utf-8") as f:
                            state = json.load(f) or {}
                    except Exception:
                        continue

                    ws = state.get("workspace_path") or str(root_path)
                    state = ensure_logical_repo_id(state, ws)
                    if state.get("logical_repo_id") == logical_repo_id:
                        coll = state.get("qdrant_collection")
                        if coll:
                            try:
                                _atomic_write_state(state_path, state)
                            except Exception as e:
                                print(f"[workspace_state] Failed to persist logical_repo_id mapping to {state_path}: {e}")
                            return coll

        state_path = root_path / STATE_DIRNAME / STATE_FILENAME
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f) or {}
            except Exception as e:
                print(f"[workspace_state] Failed to read workspace state from {state_path}: {e}")
                state = {}

            ws = state.get("workspace_path") or str(root_path)
            state = ensure_logical_repo_id(state, ws)
            if state.get("logical_repo_id") == logical_repo_id:
                coll = state.get("qdrant_collection")
                if coll:
                    try:
                        _atomic_write_state(state_path, state)
                    except Exception as e:
                        print(f"[workspace_state] Failed to persist logical_repo_id mapping to {state_path}: {e}")
                    return coll
    except Exception as e:
        print(f"[workspace_state] Error while searching collections for logical_repo_id={logical_repo_id}: {e}")
        return None

    return None


def get_or_create_collection_for_logical_repo(
    workspace_path: str,
    preferred_repo_name: Optional[str] = None,
) -> str:
    # Gate entire logical-repo based resolution behind feature flag
    if not logical_repo_reuse_enabled():
        base_repo = preferred_repo_name
        try:
            coll = get_collection_name(base_repo)
        except Exception:
            coll = get_collection_name(None)
        try:
            update_workspace_state(
                workspace_path=workspace_path,
                updates={"qdrant_collection": coll},
                repo_name=preferred_repo_name,
            )
        except Exception as e:
            print(f"[workspace_state] Failed to persist legacy qdrant_collection for {workspace_path}: {e}")
        return coll
    try:
        ws = Path(workspace_path).resolve()
    except Exception:
        ws = Path(workspace_path)

    common = _detect_git_common_dir(ws)
    if common is not None:
        canonical_root = common.parent
    else:
        canonical_root = ws

    ws_path = str(canonical_root)

    try:
        state = get_workspace_state(workspace_path=ws_path, repo_name=preferred_repo_name)
    except Exception:
        state = {}

    if not isinstance(state, dict):
        state = {}

    try:
        state = ensure_logical_repo_id(state, ws_path)
    except Exception:
        pass

    lrid = state.get("logical_repo_id")
    if isinstance(lrid, str) and lrid:
        coll = find_collection_for_logical_repo(lrid, search_root=ws_path)
        if isinstance(coll, str) and coll:
            if state.get("qdrant_collection") != coll:
                try:
                    update_workspace_state(
                        workspace_path=ws_path,
                        updates={"qdrant_collection": coll, "logical_repo_id": lrid},
                        repo_name=preferred_repo_name,
                    )
                except Exception:
                    pass
            return coll

    coll = state.get("qdrant_collection")
    if not isinstance(coll, str) or not coll:
        base_repo = preferred_repo_name
        try:
            coll = get_collection_name(base_repo)
        except Exception:
            coll = get_collection_name(None)
        try:
            update_workspace_state(
                workspace_path=ws_path,
                updates={"qdrant_collection": coll},
                repo_name=preferred_repo_name,
            )
        except Exception:
            pass

    return coll


# ===== Symbol-Level Cache for Smart Reindexing =====

def _get_symbol_cache_path(file_path: str) -> Path:
    """Get symbol cache file path for a given file."""
    try:
        fp = str(Path(file_path).resolve())
        # Create symbol cache using file hash to handle renames
        file_hash = hashlib.md5(fp.encode('utf-8')).hexdigest()[:8]
        if is_multi_repo_mode():
            # Use the same repo-name detection as other state helpers so that
            # symbol caches live under the correct per-repo .codebase directory
            repo_name = _detect_repo_name_from_path(Path(file_path))
            state_dir = _get_repo_state_dir(repo_name)
            return state_dir / f"symbols_{file_hash}.json"
        else:
            cache_dir = _get_cache_path(_resolve_workspace_root()).parent
            return cache_dir / f"symbols_{file_hash}.json"
    except Exception:
        # Fallback to simple file-based path
        cache_dir = _get_cache_path(_resolve_workspace_root()).parent
        filename = Path(file_path).name.replace('.', '_').replace('/', '_')
        return cache_dir / f"symbols_{filename}.json"


def get_cached_symbols(file_path: str) -> dict:
    """Load cached symbol metadata for a file."""
    cache_path = _get_symbol_cache_path(file_path)

    if not cache_path.exists():
        return {}

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            return cache_data.get("symbols", {})
    except Exception:
        return {}


def set_cached_symbols(file_path: str, symbols: dict, file_hash: str) -> None:
    """Save symbol metadata for a file. Extends existing to include pseudo data."""
    cache_path = _get_symbol_cache_path(file_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cache_data = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "updated_at": datetime.now().isoformat(),
            "symbols": symbols
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

        # Ensure symbol cache files are group-writable so both indexer and
        # watcher processes (potentially different users sharing a group)
        # can update them on shared volumes.
        try:
            os.chmod(cache_path, 0o664)
        except PermissionError:
            pass
    except Exception as e:
        print(f"[SYMBOL_CACHE_WARNING] Failed to save symbol cache for {file_path}: {e}")


def get_cached_pseudo(file_path: str, symbol_id: str) -> tuple[str, list[str]]:
    """Load cached pseudo description and tags for a specific symbol.

    Returns:
        (pseudo, tags) tuple, or ("", []) if not found
    """
    cached_symbols = get_cached_symbols(file_path)

    if symbol_id in cached_symbols:
        symbol_info = cached_symbols[symbol_id]
        pseudo = symbol_info.get("pseudo", "")
        tags = symbol_info.get("tags", [])

        # Ensure correct types
        if isinstance(pseudo, str):
            pseudo = pseudo
        else:
            pseudo = ""

        if isinstance(tags, list):
            tags = [str(tag) for tag in tags]
        else:
            tags = []

        return pseudo, tags

    return "", []


def set_cached_pseudo(file_path: str, symbol_id: str, pseudo: str, tags: list[str], file_hash: str) -> None:
    """Update pseudo data for a specific symbol in the cache.

    This function updates only the pseudo data without recreating the entire symbol cache,
    making it efficient for incremental updates during indexing.
    """
    cached_symbols = get_cached_symbols(file_path)

    # Update the symbol with pseudo data
    if symbol_id in cached_symbols:
        cached_symbols[symbol_id]["pseudo"] = pseudo
        cached_symbols[symbol_id]["tags"] = tags

        # Save the updated cache only when we actually have symbol entries, to
        # avoid creating empty symbol cache files before the base symbol set
        # has been seeded by the indexer/smart reindex path.
        set_cached_symbols(file_path, cached_symbols, file_hash)


def update_symbols_with_pseudo(file_path: str, symbols_with_pseudo: dict, file_hash: str) -> None:
    """Update symbols cache with pseudo data for multiple symbols at once.

    Args:
        file_path: Path to the file
        symbols_with_pseudo: Dict mapping symbol_id to (symbol_info, pseudo, tags) tuples
        file_hash: Current file hash
    """
    cached_symbols = get_cached_symbols(file_path)

    # Update symbols with their new pseudo data
    for symbol_id, (symbol_info, pseudo, tags) in symbols_with_pseudo.items():
        if symbol_id in cached_symbols:
            # Update existing symbol with pseudo data
            cached_symbols[symbol_id]["pseudo"] = pseudo
            cached_symbols[symbol_id]["tags"] = tags

            # Update content hash from symbol_info if available
            if isinstance(symbol_info, dict):
                cached_symbols[symbol_id].update(symbol_info)

    # Save the updated cache
    set_cached_symbols(file_path, cached_symbols, file_hash)


def remove_cached_symbols(file_path: str) -> None:
    """Remove symbol cache for a file (when file is deleted)."""
    cache_path = _get_symbol_cache_path(file_path)
    try:
        if cache_path.exists():
            cache_path.unlink()
    except Exception:
        pass


def compare_symbol_changes(old_symbols: dict, new_symbols: dict) -> tuple[list, list]:
    """
    Compare old and new symbols to identify changes.

    Returns:
        (unchanged_symbols, changed_symbols)
    """
    unchanged = []
    changed = []

    for symbol_id, symbol_info in new_symbols.items():
        if symbol_id in old_symbols:
            old_info = old_symbols[symbol_id]
            # Compare content hash
            if old_info.get("content_hash") == symbol_info.get("content_hash"):
                unchanged.append(symbol_id)
            else:
                changed.append(symbol_id)
        else:
            # New symbol
            changed.append(symbol_id)

    return unchanged, changed


# Add missing functions that callers expect (already defined above)