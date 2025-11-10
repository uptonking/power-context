#!/usr/bin/env python3
import os
import time
import threading
from pathlib import Path
from typing import Set, Optional
from collections import OrderedDict

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# watcher
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Ensure project root is on sys.path when run as a script (so 'scripts' can be imported)
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import critical functions first to prevent cascading failures
try:
    from scripts.workspace_state import (
        _extract_repo_name_from_path,
        get_collection_name,
        _get_global_state_dir,
        _get_repo_state_dir,
        is_multi_repo_mode,
        get_cached_file_hash,
        set_cached_file_hash,
    )
except ImportError:
    # If critical imports fail, set None to prevent crashes
    _extract_repo_name_from_path = None  # type: ignore
    get_collection_name = None  # type: ignore
    _get_global_state_dir = None  # type: ignore
    _get_repo_state_dir = None  # type: ignore
    is_multi_repo_mode = None  # type: ignore
    get_cached_file_hash = None  # type: ignore
    set_cached_file_hash = None  # type: ignore

# Import optional functions that may not exist
try:
    from scripts.workspace_state import (
        get_workspace_state,
        update_indexing_status,
        update_workspace_state,
        remove_cached_file,
    )
except ImportError:
    # Optional functions - set to None if not available
    get_workspace_state = None  # type: ignore
    update_indexing_status = None  # type: ignore
    update_workspace_state = None  # type: ignore
    remove_cached_file = None  # type: ignore
import hashlib
from datetime import datetime

import scripts.ingest_code as idx

# Import remote upload client
try:
    from scripts.remote_upload_client import (
        RemoteUploadClient,
        is_remote_mode_enabled,
        get_remote_config
    )
    _REMOTE_UPLOAD_AVAILABLE = True
except ImportError:
    _REMOTE_UPLOAD_AVAILABLE = False

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
ROOT = Path(os.environ.get("WATCH_ROOT", "/work"))

# Debounce interval
DELAY_SECS = float(os.environ.get("WATCH_DEBOUNCE_SECS", "1.0"))

# Simple LRU cache implementation to prevent memory growth
class LRUCache:
    """Simple LRU cache with size limits."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self._hits += 1
            return self.cache[key]
        self._misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            # Update existing entry
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new entry, evict if necessary
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self._hits = 0
        self._misses = 0

    def get_hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def size(self):
        return len(self.cache)

# Multi-repo collection management with size-limited caches
_collection_cache = LRUCache(max_size=500)  # Cache for repo path -> collection name mapping
_repo_cache = LRUCache(max_size=2000)       # Cache for file path -> repo path mapping

# Optional cache statistics logging (disabled by default)
_ENABLE_CACHE_STATS = os.environ.get("ENABLE_CACHE_STATS", "false").lower() == "true"

def _log_cache_stats():
    """Log cache statistics for monitoring."""
    if _ENABLE_CACHE_STATS:
        print(f"[cache_stats] Collection cache: {_collection_cache.size()} items, "
              f"hit rate: {_collection_cache.get_hit_rate():.2%}")
        print(f"[cache_stats] Repo cache: {_repo_cache.size()} items, "
              f"hit rate: {_repo_cache.get_hit_rate():.2%}")


def _detect_repo_for_file(file_path: Path) -> Optional[Path]:
    """
    Detect which repository a file belongs to using the new workspace_state functions.
    Returns the repository root path or None if not under WATCH_ROOT.
    """
    try:
        # Normalize paths - get current WATCH_ROOT to handle env changes
        abs_file = file_path.resolve()
        watch_root = Path(os.environ.get("WATCH_ROOT", "/work")).resolve()
        abs_root = watch_root

        # File must be under WATCH_ROOT
        try:
            abs_file.relative_to(abs_root)
        except ValueError:
            return None

        # Check cache first
        file_key = str(abs_file)
        cached_result = _repo_cache.get(file_key)
        if cached_result is not None:
            return cached_result

        # Use new workspace_state function to extract repo name from file path
        repo_name = _extract_repo_name_from_path(str(abs_file))

        # Construct repo path from the detected repo name
        # Look for the repo directory under WATCH_ROOT
        repo_path = None
        rel_path = abs_file.relative_to(abs_root)
        path_parts = rel_path.parts

        if not path_parts:
            return None

        # Strategy 1: Look for repo with matching name in common locations
        # Check immediate directories under WATCH_ROOT
        if len(path_parts) >= 1:
            potential_repo_name = path_parts[0]
            if potential_repo_name and repo_name and (potential_repo_name == repo_name or potential_repo_name.lower() == repo_name.lower()):
                repo_path = abs_root / potential_repo_name
                if repo_path.exists():
                    _repo_cache.put(file_key, repo_path)
                    return repo_path

        # Strategy 2: Walk up the path hierarchy to find repo root
        current_path = abs_file.parent
        abs_root_resolved = abs_root.resolve()

        while True:
            # Check if current path name matches our detected repo name
            if current_path.name == repo_name or current_path.name.lower() == repo_name.lower():
                repo_path = current_path
                break

            # Check if current_path has .git
            if (current_path / ".git").exists():
                repo_path = current_path
                break

            # Stop if we've reached WATCH_ROOT or above it
            current_resolved = current_path.resolve()
            if current_resolved == abs_root_resolved or current_resolved == current_path.parent.resolve():
                break

            current_path = current_path.parent

        # Strategy 3: Fallback to first-level directory under WATCH_ROOT
        if repo_path is None:
            repo_path = abs_root / path_parts[0]
            if not repo_path.exists():
                # If the assumed repo path doesn't exist, fall back to WATCH_ROOT itself
                repo_path = abs_root

        # Cache the result
        _repo_cache.put(file_key, repo_path)
        return repo_path

    except (OSError, ValueError, RuntimeError) as e:
        # Log the specific error for debugging if needed
        print(f"[repo_detection] Error detecting repo for {file_path}: {e}")
        return None


def _get_collection_for_repo(repo_path: Path) -> str:
    """
    Get the collection name for a repository using new workspace_state functions.
    Uses caching to avoid repeated calls.
    """
    try:
        repo_key = str(repo_path)  # repo_path is already resolved

        # Check cache first
        cached_collection = _collection_cache.get(repo_key)
        if cached_collection is not None:
            return cached_collection

        # Extract repo name using new workspace_state function
        repo_name = _extract_repo_name_from_path(repo_key)

        # Use new workspace_state function to get collection name
        collection_name = get_collection_name(repo_name)

        # Cache the result
        _collection_cache.put(repo_key, collection_name)
        return collection_name

    except (OSError, ImportError, ValueError) as e:
        # Fallback to default collection name with logging
        print(f"[collection_detection] Error getting collection for {repo_path}: {e}")
        fallback = os.environ.get("COLLECTION_NAME", "my-collection")
        return fallback


def _get_collection_for_file(file_path: Path) -> str:
    """
    Get the collection name for a file by detecting its repository.
    """
    # In single-repo mode, always use the global collection
    if not is_multi_repo_mode():
        return os.environ.get("COLLECTION_NAME", "my-collection")

    # Multi-repo mode: detect repository for file
    repo_path = _detect_repo_for_file(file_path)

    if repo_path:
        collection = _get_collection_for_repo(repo_path)
        return collection

    # Fallback to default collection
    return os.environ.get("COLLECTION_NAME", "my-collection")


def _get_remote_client_for_repo(repo_path: Path, remote_clients: dict, remote_config: dict) -> Optional[RemoteUploadClient]:
    """
    Get or create a remote upload client for a specific repository.
    Uses the new repo-specific metadata structure for delta bundles.
    """
    repo_key = str(repo_path)  # repo_path is already resolved

    if repo_key in remote_clients:
        return remote_clients[repo_key]

    # Create new client for this repository
    try:
        collection_name = _get_collection_for_repo(repo_path)

        # Extract repo name and get the repo-specific metadata directory
        repo_name = _extract_repo_name_from_path(repo_key)
        repo_state_dir = _get_repo_state_dir(repo_name)

        # Use the actual repository path as workspace_path for file resolution
        # But use the repo-specific metadata directory for delta bundle storage
        workspace_path = repo_key  # This is the actual repo path where files are located
        metadata_path = str(repo_state_dir)  # This is where delta bundles are stored

        client = RemoteUploadClient(
            upload_endpoint=remote_config["upload_endpoint"],
            workspace_path=workspace_path,
            collection_name=collection_name,
            max_retries=remote_config["max_retries"],
            timeout=remote_config["timeout"],
            metadata_path=metadata_path
        )
        remote_clients[repo_key] = client
        print(f"[remote_upload] Created client for repo: {repo_path} -> {collection_name} (workspace: {workspace_path}, metadata: {metadata_path})")
        return client
    except (OSError, ValueError, ConnectionError, KeyError) as e:
        print(f"[remote_upload] Error creating client for {repo_path}: {e}")
        return None


class ChangeQueue:
    def __init__(self, process_cb, remote_clients: Optional[dict] = None, remote_config: Optional[dict] = None):
        self._lock = threading.Lock()
        self._paths: Set[Path] = set()
        self._timer: threading.Timer | None = None
        self._process_cb = process_cb
        self._remote_clients = remote_clients or {}
        self._remote_config = remote_config

    def add(self, p: Path):
        with self._lock:
            self._paths.add(p)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DELAY_SECS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self):
        with self._lock:
            paths = list(self._paths)
            self._paths.clear()
            self._timer = None

        # Handle remote upload if enabled
        if self._remote_clients and _REMOTE_UPLOAD_AVAILABLE and self._remote_config:
            try:
                # Group paths by repository for remote upload
                repo_groups = {}
                for path in paths:
                    repo_path = _detect_repo_for_file(path)
                    if repo_path:
                        repo_key = str(repo_path)  # repo_path is already resolved
                        if repo_key not in repo_groups:
                            repo_groups[repo_key] = []
                        repo_groups[repo_key].append(path)
                    else:
                        # Use default client for files not under any repo
                        if "default" not in repo_groups:
                            repo_groups["default"] = []
                        repo_groups["default"].append(path)

                # Process each repository with its own remote client
                all_successful = True
                for repo_key, repo_paths in repo_groups.items():
                    try:
                        # Get or create remote client for this repository
                        if repo_key == "default":
                            remote_client = self._remote_clients.get("default")
                        else:
                            remote_client = _get_remote_client_for_repo(
                                Path(repo_key), self._remote_clients, self._remote_config
                            )

                        if remote_client:
                            success = remote_client.process_and_upload_changes(repo_paths)
                            if not success:
                                all_successful = False
                                print(f"[remote_upload] Upload failed for repo {repo_key}, falling back to local processing")
                                self._process_cb(repo_paths)
                            else:
                                print(f"[remote_upload] Upload successful for repo {repo_key}")
                        else:
                            all_successful = False
                            print(f"[remote_upload] No remote client available for repo {repo_key}, falling back to local processing")
                            self._process_cb(repo_paths)
                    except Exception as e:
                        all_successful = False
                        print(f"[remote_upload] Error during delta upload for repo {repo_key}: {e}")
                        print("[remote_upload] Falling back to local processing")
                        self._process_cb(repo_paths)

                if all_successful:
                    print("[remote_upload] All repository uploads completed successfully")

            except Exception as e:
                print(f"[remote_upload] Error during multi-repo delta upload: {e}")
                print("[remote_upload] Falling back to local processing")
                self._process_cb(paths)
        else:
            self._process_cb(paths)


class IndexHandler(FileSystemEventHandler):
    def __init__(self, root: Path, queue: ChangeQueue, client: Optional[QdrantClient], default_collection: str = None):
        super().__init__()
        self.root = root
        self.queue = queue
        self.client = client
        self.default_collection = default_collection
        self.excl = idx._Excluder(root)
        # Track ignore file for live reloads
        try:
            ig_name = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
            self._ignore_path = (self.root / ig_name).resolve()
        except (OSError, ValueError) as e:
            print(f"[ignore_file] Could not resolve ignore file path: {e}")
            self._ignore_path = None
        self._ignore_mtime = (
            (self._ignore_path.stat().st_mtime if self._ignore_path and self._ignore_path.exists() else 0.0)
        )

    def _maybe_reload_excluder(self):
        try:
            if not self._ignore_path:
                return
            cur = self._ignore_path.stat().st_mtime if self._ignore_path.exists() else 0.0
            if cur != self._ignore_mtime:
                self.excl = idx._Excluder(self.root)
                self._ignore_mtime = cur
                try:
                    print(f"[ignore_reload] reloaded patterns from {self._ignore_path}")
                except (OSError, RuntimeError) as e:
                    print(f"[ignore_reload] Error printing reload message: {e}")
                    pass
        except (OSError, IOError) as e:
            print(f"[ignore_reload] Error reloading ignore patterns: {e}")
            pass

    def _maybe_enqueue(self, src_path: str):
        # Refresh ignore patterns if the file changed
        self._maybe_reload_excluder()
        p = Path(src_path)
        try:
            # normalize to absolute within root
            p = p.resolve()
        except (OSError, ValueError):
            return
        # skip directories
        if p.is_dir():
            return
        # ensure file is under root
        try:
            rel = p.relative_to(self.root.resolve())
        except ValueError:
            return

        # NEW: Exclude root-level metadata directory and its contents
        try:
            # Get the global state directory path and exclude it
            if _get_global_state_dir is not None:
                global_state_dir = _get_global_state_dir()
                if p.is_relative_to(global_state_dir):
                    return  # Skip files in /work/.codebase/
        except (OSError, ValueError):
            pass  # If we can't determine global state dir, continue processing

        # Skip all .codebase directories (including per-repo ones in multi-repo mode)
        if any(part == ".codebase" for part in p.parts):
            return

        # directory-level excludes (parent dir)
        rel_dir = "/" + str(rel.parent).replace(os.sep, "/")
        if rel_dir == "/.":
            rel_dir = "/"
        if self.excl.exclude_dir(rel_dir):
            return
        # only code files
        if p.suffix.lower() not in idx.CODE_EXTS:
            return
        # file-level excludes
        relf = (rel_dir.rstrip("/") + "/" + p.name).replace("//", "/")
        if self.excl.exclude_file(relf):
            return
        self.queue.add(p)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_enqueue(event.src_path)

    def on_deleted(self, event):
        if event.is_directory:
            return
        try:
            p = Path(event.src_path).resolve()
        except Exception:
            return
        # Only attempt deletion for code files we would have indexed
        if p.suffix.lower() not in idx.CODE_EXTS:
            return
        # Only attempt deletion if we have a local client
        if self.client is not None:
            try:
                # Get the correct collection for this file
                collection = _get_collection_for_file(p)
                idx.delete_points_by_path(self.client, collection, str(p))
                print(f"[deleted] {p} -> {collection}")
            except Exception:
                pass
        else:
            print(f"[remote_mode] File deletion detected: {p}")

        # Drop local cache entry (always do this)
        try:
            repo_path = _detect_repo_for_file(p)
            if repo_path:
                # Use new repo-based cache structure
                repo_name = _extract_repo_name_from_path(str(repo_path))
                remove_cached_file(str(p), repo_name)
            else:
                # Use root as fallback
                root_repo_name = _extract_repo_name_from_path(str(self.root))
                remove_cached_file(str(p), root_repo_name)
        except Exception:
            pass

        try:
            repo_path = _detect_repo_for_file(p) or self.root
            _log_activity(str(repo_path), "deleted", p)
        except Exception as e:
            try:
                print(f"[delete_error] {p}: {e}")
            except Exception:
                pass

    def on_moved(self, event):
        if event.is_directory:
            return
        # Attempt optimized rename when content unchanged; else fallback to reindex
        try:
            src = Path(event.src_path).resolve()
            dest = Path(event.dest_path).resolve()
        except Exception:
            return
        # Only react to code files
        if dest.suffix.lower() not in idx.CODE_EXTS and src.suffix.lower() not in idx.CODE_EXTS:
            return
        # If destination directory is ignored, treat as simple deletion
        try:
            rel_dir = "/" + str(dest.parent.relative_to(self.root.resolve())).replace(os.sep, "/")
            if rel_dir == "/.":
                rel_dir = "/"
            if self.excl.exclude_dir(rel_dir):
                if src.suffix.lower() in idx.CODE_EXTS:
                    if self.client is not None:
                        try:
                            src_collection = _get_collection_for_file(src)
                            idx.delete_points_by_path(self.client, src_collection, str(src))
                            print(f"[moved:ignored_dest_deleted_src] {src} -> {dest} (from {src_collection})")
                        except Exception:
                            pass
                    else:
                        print(f"[remote_mode] Move to ignored destination: {src} -> {dest}")
                return
        except Exception:
            pass
        # Try in-place rename (preserve vectors) - only if we have a local client
        moved_count = -1
        if self.client is not None:
            try:
                # Get collections for source and destination
                src_collection = _get_collection_for_file(src)
                dest_collection = _get_collection_for_file(dest)
                moved_count = _rename_in_store(self.client, src_collection, src, dest, dest_collection)
            except Exception:
                moved_count = -1
        if moved_count and moved_count > 0:
            try:
                src_collection = _get_collection_for_file(src)
                print(f"[moved] {src} -> {dest} ({moved_count} chunk(s) relinked from {src_collection})")
                # Update local cache: carry hash from src to dest if present
                prev_hash = None
                src_repo = _detect_repo_for_file(src)
                dest_repo = _detect_repo_for_file(dest)
                try:
                    # Use new repo-based cache structure
                    src_repo_name = _extract_repo_name_from_path(str(src_repo or self.root))
                    prev_hash = get_cached_file_hash(str(src), src_repo_name)
                except Exception:
                    prev_hash = None
                if prev_hash:
                    try:
                        # Use new repo-based cache structure
                        dest_repo_name = _extract_repo_name_from_path(str(dest_repo or self.root))
                        set_cached_file_hash(str(dest), prev_hash, dest_repo_name)
                    except Exception:
                        pass
                    try:
                        remove_cached_file(str(src), src_repo_name)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                repo_path = _detect_repo_for_file(dest) or self.root
                _log_activity(str(repo_path), "moved", dest, {"from": str(src), "chunks": int(moved_count)})
            except Exception:
                pass
            return
        # Fallback: delete old then index new destination
        if self.client is not None:
            try:
                if src.suffix.lower() in idx.CODE_EXTS:
                    src_collection = _get_collection_for_file(src)
                    idx.delete_points_by_path(self.client, src_collection, str(src))
                    print(f"[moved:deleted_src] {src} from {src_collection}")
            except Exception:
                pass
        else:
            print(f"[remote_mode] Move detected: {src} -> {dest}")
        try:
            self._maybe_enqueue(str(dest))
        except Exception:
            pass

# --- Workspace state helpers ---
def _set_status_indexing(workspace_path: str, total_files: int) -> None:
    try:
        # Extract repo name to use new structure
        repo_name = _extract_repo_name_from_path(workspace_path)
        if update_indexing_status is not None:
            update_indexing_status(
                repo_name=repo_name,
                status={
                    "state": "indexing",
                    "started_at": datetime.now().isoformat(),
                    "progress": {"files_processed": 0, "total_files": int(total_files)},
                },
            )
    except Exception:
        pass


def _update_progress(
    workspace_path: str, started_at: str, processed: int, total: int, current_file: Path | None
) -> None:
    try:
        # Extract repo name to use new structure
        repo_name = _extract_repo_name_from_path(workspace_path)
        if update_indexing_status is not None:
            update_indexing_status(
                repo_name=repo_name,
                status={
                    "state": "indexing",
                    "started_at": started_at,
                    "progress": {
                        "files_processed": int(processed),
                        "total_files": int(total),
                        "current_file": str(current_file) if current_file else None,
                    },
                },
            )
    except Exception:
        pass


def _log_activity(workspace_path: str, action: str, file_path: Path, details: dict | None = None) -> None:
    try:
        # Extract repo name from workspace path to use new structure
        repo_name = _extract_repo_name_from_path(workspace_path)

        # Import log_activity from workspace_state
        from scripts.workspace_state import log_activity

        # Convert action to match expected ActivityAction type
        valid_actions = {'indexed', 'deleted', 'skipped', 'scan-completed', 'initialized', 'moved'}
        if action not in valid_actions:
            action = 'indexed'  # Default fallback

        # Import ActivityAction for type checking
        from scripts.workspace_state import ActivityAction
        if isinstance(action, str):
            # Convert string to proper ActivityAction format
            action = action  # type: ignore  # The function will validate the action

        # Use new log_activity function with repo-based structure
        log_activity(
            repo_name=repo_name,
            action=action,  # type: ignore
            file_path=str(file_path),
            details=details
        )
    except Exception:
        pass


# --- Move/Rename optimization: reuse vectors when file content unchanged ---
def _rename_in_store(client: QdrantClient, src_collection: str, src: Path, dest: Path, dest_collection: str = None) -> int:
    """Best-effort: if dest content hash matches previously indexed src hash,
    update points in-place to the new path without re-embedding.

    Supports cross-collection moves when dest_collection is different from src_collection.

    Returns number of points moved, or -1 if not applicable/failure.
    """
    if dest_collection is None:
        dest_collection = src_collection
    try:
        if not dest.exists() or dest.is_dir():
            return -1
        try:
            text = dest.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return -1
        dest_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        prev = idx.get_indexed_file_hash(client, src_collection, str(src))
        if not prev or prev != dest_hash:
            return -1

        moved = 0
        next_offset = None
        while True:
            filt = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.path", match=models.MatchValue(value=str(src))
                    )
                ]
            )
            points, next_offset = client.scroll(
                collection_name=src_collection,
                scroll_filter=filt,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=next_offset,
            )
            if not points:
                break
            new_points = []
            for rec in points:
                payload = rec.payload or {}
                md = (payload.get("metadata") or {})
                code = md.get("code") or ""
                try:
                    start_line = int(md.get("start_line") or 1)
                    end_line = int(md.get("end_line") or start_line)
                except Exception:
                    start_line, end_line = 1, 1
                new_id = idx.hash_id(code, str(dest), start_line, end_line)

                # Update metadata path fields
                new_md = dict(md)
                new_md["path"] = str(dest)
                new_md["path_prefix"] = str(dest.parent)
                # Recompute dual-path hints
                cur_path = str(dest)
                host_root = str(os.environ.get("HOST_INDEX_PATH") or "").strip().rstrip("/")
                host_path = None
                container_path = None
                try:
                    if cur_path.startswith("/work/") and host_root:
                        rel = cur_path[len("/work/") :]
                        host_path = os.path.realpath(os.path.join(host_root, rel))
                        container_path = cur_path
                    else:
                        host_path = cur_path
                        if host_root and cur_path.startswith(host_root + "/"):
                            rel = cur_path[len(host_root) + 1 :]
                            container_path = "/work/" + rel
                except Exception:
                    host_path = cur_path
                    container_path = cur_path if cur_path.startswith("/work/") else None
                new_md["host_path"] = host_path
                new_md["container_path"] = container_path

                new_payload = dict(payload)
                new_payload["metadata"] = new_md

                vec = rec.vector  # Named or unnamed vector(s)
                try:
                    new_points.append(
                        models.PointStruct(id=new_id, vector=vec, payload=new_payload)
                    )
                except Exception:
                    continue
            if new_points:
                idx.upsert_points(client, dest_collection, new_points)
                moved += len(new_points)

        try:
            idx.delete_points_by_path(client, src_collection, str(src))
        except Exception:
            pass
        return moved
    except Exception:
        return -1


def main():
    # Check if remote mode is enabled
    remote_mode = False
    remote_clients = {}  # Map repo paths to remote clients

    if _REMOTE_UPLOAD_AVAILABLE and is_remote_mode_enabled():
        remote_mode = True
        try:
            remote_config = get_remote_config()

            # For multi-repo support, we'll create remote clients on-demand for each repository
            # The base configuration will be used, but collection names will be determined per-repo
            print(f"[remote_upload] Remote mode enabled: {remote_config['upload_endpoint']}")
            print("[remote_upload] Multi-repo remote support - will create clients per repository")

            # Create a default client for backward compatibility
            try:
                # For the default client, use the global metadata directory to avoid permission issues
                if _get_global_state_dir is not None:
                    global_state_dir = _get_global_state_dir()
                    default_workspace_path = str(global_state_dir)
                else:
                    # Fallback if function is not available
                    default_workspace_path = "/work"

                default_remote_client = RemoteUploadClient(
                    upload_endpoint=remote_config["upload_endpoint"],
                    workspace_path=default_workspace_path,
                    collection_name=remote_config["collection_name"],
                    max_retries=remote_config["max_retries"],
                    timeout=remote_config["timeout"]
                )

                # Check server status
                status = default_remote_client.get_server_status()
                if status.get("success", False):
                    print(f"[remote_upload] Server status: {status.get('status', 'unknown')}")
                else:
                    print(f"[remote_upload] Warning: Could not reach server - {status.get('error', {}).get('message', 'Unknown error')}")

                # Store as default client (will be used for single-repo scenarios)
                remote_clients["default"] = default_remote_client
                print(f"[remote_upload] Default client initialized with workspace: {default_workspace_path}")

            except Exception as e:
                print(f"[remote_upload] Error initializing default remote client: {e}")
                print("[remote_upload] Will create clients per-repository as needed")

        except Exception as e:
            print(f"[remote_upload] Error initializing remote mode: {e}")
            print("[remote_upload] Falling back to local mode")
            remote_mode = False
            remote_clients = {}

      # Determine collection and mode based on MULTI_REPO_MODE setting
    try:
        from scripts.workspace_state import get_collection_name as _get_coll
    except Exception:
        _get_coll = None

    multi_repo_enabled = is_multi_repo_mode() if is_multi_repo_mode else False

    if multi_repo_enabled:
        # Multi-repo mode: use per-repo collections
        default_collection = os.environ.get("COLLECTION_NAME", "my-collection")
        try:
            if _get_coll:
                default_collection = _get_coll(str(ROOT))
        except Exception:
            pass
        print("[multi_repo] Multi-repo mode enabled - files will be routed to per-repo collections")
    else:
        # Single-repo mode: use one collection for everything
        default_collection = os.environ.get("COLLECTION_NAME", "my-collection")
        print("[single_repo] Single-repo mode enabled - using single collection for all files")

    mode_str = "REMOTE" if remote_mode else "LOCAL"
    print(
        f"Watch mode: {mode_str} root={ROOT} qdrant={QDRANT_URL} collection={default_collection} model={MODEL}"
    )

    # Initialize Qdrant client for local mode (remote mode doesn't need it for basic operation)
    client = None
    model = None
    vector_name = None

    if not remote_mode:
        client = QdrantClient(
            url=QDRANT_URL, timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20)
        )

        # Compute embedding dimension first (for deterministic dense vector selection)
        model = TextEmbedding(model_name=MODEL)
        dim = len(next(model.embed(["dimension probe"])))

        # Determine dense vector name deterministically (use default collection as reference)
        try:
            info = client.get_collection(default_collection)
            cfg = info.config.params.vectors
            if isinstance(cfg, dict) and cfg:
                # Prefer vector whose size matches embedding dim
                vector_name = None
                for name, params in cfg.items():
                    psize = getattr(params, "size", None) or getattr(params, "dim", None)
                    if psize and int(psize) == int(dim):
                        vector_name = name
                        break
                # If LEX vector exists, pick a different name as dense
                if vector_name is None and getattr(idx, "LEX_VECTOR_NAME", None) in cfg:
                    for name in cfg.keys():
                        if name != idx.LEX_VECTOR_NAME:
                            vector_name = name
                            break
                if vector_name is None:
                    vector_name = idx._sanitize_vector_name(MODEL)
            else:
                vector_name = idx._sanitize_vector_name(MODEL)
        except Exception:
            vector_name = idx._sanitize_vector_name(MODEL)

        # Ensure default collection + payload indexes exist
        try:
            idx.ensure_collection(client, default_collection, dim, vector_name)
        except Exception:
            pass
        idx.ensure_payload_indexes(client, default_collection)

    # Ensure workspace state exists and set collection based on mode
    try:
        if multi_repo_enabled:
            # Multi-repo mode: use per-repo state structure
            root_repo_name = _extract_repo_name_from_path(str(ROOT))
            if not root_repo_name:
                print("[workspace_state] Multi-repo: Root path is not a repo; skipping root state initialization")
            else:
                root_collection = get_collection_name(root_repo_name)
                update_indexing_status(
                    repo_name=root_repo_name,
                    status={"state": "watching"},
                )
                print(
                    f"[workspace_state] Multi-repo: Initialized state for repo: {root_repo_name} -> {root_collection}"
                )
        else:
            # Single-repo mode: use original workspace state structure
            update_workspace_state(
                workspace_path=str(ROOT),
                updates={"qdrant_collection": default_collection},
            )
            update_indexing_status(status={"state": "watching"})
            print(f"[workspace_state] Single-repo: Initialized state for workspace: {str(ROOT)} -> {default_collection}")
    except Exception as e:
        print(f"[workspace_state] Error initializing workspace state: {e}")
        pass

    # Create change queue with remote clients if enabled
    if remote_mode:
        q = ChangeQueue(
            lambda paths: _process_paths(paths, client, model, vector_name, str(ROOT), remote_mode),
            remote_clients=remote_clients,
            remote_config=get_remote_config() if _REMOTE_UPLOAD_AVAILABLE else None
        )
    else:
        q = ChangeQueue(lambda paths: _process_paths(paths, client, model, vector_name, str(ROOT), remote_mode))

    handler = IndexHandler(ROOT, q, client, default_collection)

    obs = Observer()
    obs.schedule(handler, str(ROOT), recursive=True)
    obs.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        obs.stop()
        obs.join()


def _process_paths(paths, client, model, vector_name: str, workspace_path: str, remote_mode: bool = False):
    # In remote mode, actual processing is handled by the remote client
    # This function is called as a fallback when remote upload fails
    if remote_mode:
        print(f"[local_fallback] Processing {len(paths)} files locally due to remote upload failure")

    # Prepare progress
    unique_paths = sorted(set(Path(x) for x in paths))
    total = len(unique_paths)
    started_at = datetime.now().isoformat()

    # Group files by repository for progress tracking
    repo_groups = {}
    for p in unique_paths:
        repo_path = _detect_repo_for_file(p) or Path(workspace_path)
        if str(repo_path) not in repo_groups:
            repo_groups[str(repo_path)] = []
        repo_groups[str(repo_path)].append(p)

    # Initialize progress for all repositories
    for repo_path, repo_files in repo_groups.items():
        try:
            # Extract repo name to use new structure
            repo_name = _extract_repo_name_from_path(repo_path)
            update_indexing_status(
                repo_name=repo_name,
                status={
                    "state": "indexing",
                    "started_at": started_at,
                    "progress": {"files_processed": 0, "total_files": len(repo_files)},
                },
            )
        except Exception:
            pass

    processed = 0
    for p in unique_paths:
        current = p

        # Get collection for this file
        collection = _get_collection_for_file(p)
        repo_path = _detect_repo_for_file(p) or Path(workspace_path)


        if not p.exists():
            # File was removed; ensure its points are deleted
            if client is not None:  # Only process if we have a local client
                try:
                    idx.delete_points_by_path(client, collection, str(p))
                    print(f"[deleted] {p} -> {collection}")
                except Exception:
                    pass
            _log_activity(str(repo_path), "deleted", p)
            processed += 1
            # Update progress for the specific repository
            try:
                repo_files = repo_groups[str(repo_path)]
                repo_processed = len([f for f in repo_files[:processed] if not f.exists()])
                _update_progress(str(repo_path), started_at, repo_processed, len(repo_files), current)
            except Exception:
                pass
            continue

        # Only process files locally if we have a client and model
        if client is not None and model is not None:
            # Lazily instantiate model if needed
            if model is None:
                from fastembed import TextEmbedding
                mname = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
                model = TextEmbedding(model_name=mname)

            # Ensure collection exists for this repo
            try:
                idx.ensure_collection(client, collection, len(next(model.embed(["dimension probe"]))), vector_name)
                idx.ensure_payload_indexes(client, collection)
            except Exception:
                pass

            print(f"[DEBUG] Indexing file with path: {p}")
            ok = idx.index_single_file(
                client, model, collection, vector_name, p, dedupe=True, skip_unchanged=False
            )
            status = "indexed" if ok else "skipped"
            print(f"[{status}] {p} -> {collection}")
            if ok:
                try:
                    size = int(p.stat().st_size)
                except Exception:
                    size = None
                _log_activity(str(repo_path), "indexed", p, {"file_size": size})
            else:
                _log_activity(str(repo_path), "skipped", p, {"reason": "no-change-or-error"})
        else:
            # In remote mode without fallback, just log activity
            print(f"[remote_mode] Not processing locally: {p}")
            _log_activity(str(repo_path), "indexed", p, {"reason": "remote_processed"})

        processed += 1
        # Update progress for the specific repository
        try:
            repo_files = repo_groups[str(repo_path)]
            repo_processed = len([f for f in repo_files if f in unique_paths[:processed]])
            _update_progress(str(repo_path), started_at, repo_processed, len(repo_files), current)

            # Log cache stats periodically (every 50 files processed)
            if processed % 50 == 0:
                _log_cache_stats()
        except Exception:
            pass

    # Return to watching state for all repositories
    for repo_path in repo_groups.keys():
        try:
            # Extract repo name to use new structure
            repo_name = _extract_repo_name_from_path(repo_path)
            update_indexing_status(
                repo_name=repo_name,
                status={"state": "watching"},
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
