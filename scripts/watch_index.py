#!/usr/bin/env python3
import os
import time
import threading
from pathlib import Path
from typing import Set, Optional

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

from scripts.workspace_state import (
    _extract_repo_name_from_path,
    get_collection_name,
    _get_global_state_dir,
    is_multi_repo_mode,
    get_cached_file_hash,
    set_cached_file_hash,
    remove_cached_file,
    update_indexing_status,
)
import hashlib
from datetime import datetime

import scripts.ingest_code as idx


QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
ROOT = Path(os.environ.get("WATCH_ROOT", "/work"))

# Debounce interval
DELAY_SECS = float(os.environ.get("WATCH_DEBOUNCE_SECS", "1.0"))



def _detect_repo_for_file(file_path: Path) -> Optional[Path]:
    """
    Detect which repository a file belongs to.
    Returns the repository root path or None if not under WATCH_ROOT.
    """
    try:
        rel_path = file_path.relative_to(ROOT)
        if rel_path.parts:
            return ROOT / rel_path.parts[0]
    except ValueError:
        return None


def _get_collection_for_repo(repo_path: Path) -> str:
    """
    Get the collection name for a repository.
    """
    try:
        repo_name = _extract_repo_name_from_path(str(repo_path))
        return get_collection_name(repo_name)
    except Exception:
        return os.environ.get("COLLECTION_NAME", "my-collection")


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




class ChangeQueue:
    def __init__(self, process_cb):
        self._lock = threading.Lock()
        self._paths: Set[Path] = set()
        self._timer: threading.Timer | None = None
        self._process_cb = process_cb

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

        self._process_cb(paths)


class IndexHandler(FileSystemEventHandler):
    def __init__(self, root: Path, queue: ChangeQueue, client: Optional[QdrantClient], default_collection: str = None):
        super().__init__()
        self.root = root
        self.queue = queue
        self.client = client
        self.default_collection = default_collection
        self.collection = default_collection
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
            print(f"File deletion detected: {p}")

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
            # Move detected - proceed with rename logic
        except Exception as e:
            print(f"[move_error] {e}")
            return
        # Only react to code files
        if dest.suffix.lower() not in idx.CODE_EXTS and src.suffix.lower() not in idx.CODE_EXTS:
            return
        # If destination directory is ignored, treat as simple deletion
        try:
            rel_dir = "/" + str(dest.parent.resolve().relative_to(self.root.resolve())).replace(os.sep, "/")
            if rel_dir == "/.":
                rel_dir = "/"
            if self.excl.exclude_dir(rel_dir):
                if src.suffix.lower() in idx.CODE_EXTS:
                    if self.client is not None:
                        try:
                            # Try to delete from the file's current collection first
                            src_collection = _get_collection_for_file(src)
                            try:
                                idx.delete_points_by_path(self.client, src_collection, str(src))
                            except Exception:
                                # Fallback to original behavior if source collection doesn't exist
                                idx.delete_points_by_path(self.client, self.collection, str(src))
                            print(f"[moved:ignored_dest_deleted_src] {src} -> {dest}")
                        except Exception:
                            pass
                    else:
                        print(f"[remote_mode] Move to ignored destination: {src} -> {dest}")
                return
        except Exception:
            pass
        # Determine source and destination collections
        src_collection = _get_collection_for_file(src)
        dest_collection = _get_collection_for_file(dest)
        is_cross_collection = src_collection != dest_collection

        # For cross-collection moves, log the operation since it's a significant event
        if is_cross_collection:
            print(f"[cross_collection_move] {src} -> {dest}")
        # Fallback: delete old then index new destination
        # This handles all moves using reliable delete+reindex approach
        if self.client is not None:
            try:
                if src.suffix.lower() in idx.CODE_EXTS:
                    # Try to delete from the file's current collection first
                    src_collection = _get_collection_for_file(src)
                    try:
                        idx.delete_points_by_path(self.client, src_collection, str(src))
                    except Exception:
                        # Final fallback to original behavior if source collection doesn't exist
                        idx.delete_points_by_path(self.client, self.collection, str(src))
                    print(f"[moved:deleted_src] {src}")
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

    print(
        f"Watch mode: LOCAL root={ROOT} qdrant={QDRANT_URL} collection={default_collection} model={MODEL}"
    )

    # Initialize Qdrant client
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

    # Create change queue
    q = ChangeQueue(lambda paths: _process_paths(paths, client, model, vector_name, str(ROOT)))

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


def _process_paths(paths, client, model, vector_name: str, workspace_path: str):

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
            print(f"Not processing locally: {p}")
            _log_activity(str(repo_path), "indexed", p, {"reason": "skipped"})

        processed += 1
        # Update progress for the specific repository
        try:
            repo_files = repo_groups[str(repo_path)]
            repo_processed = len([f for f in repo_files if f in unique_paths[:processed]])
            _update_progress(str(repo_path), started_at, repo_processed, len(repo_files), current)
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
