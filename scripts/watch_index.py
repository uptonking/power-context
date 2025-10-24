#!/usr/bin/env python3
import os
import time
import threading
from pathlib import Path
from typing import Set

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
    get_workspace_state,
    update_indexing_status,
    update_last_activity,
    update_workspace_state,
    get_cached_file_hash,
    set_cached_file_hash,
    remove_cached_file,
)
import hashlib
from datetime import datetime

import scripts.ingest_code as idx

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
ROOT = Path(os.environ.get("WATCH_ROOT", "/work")).resolve()

# Debounce interval
DELAY_SECS = float(os.environ.get("WATCH_DEBOUNCE_SECS", "1.0"))


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
    def __init__(self, root: Path, queue: ChangeQueue, client: QdrantClient, collection: str):
        super().__init__()
        self.root = root
        self.queue = queue
        self.client = client
        self.collection = collection
        self.excl = idx._Excluder(root)
        # Track ignore file for live reloads
        try:
            ig_name = os.environ.get("QDRANT_IGNORE_FILE", ".qdrantignore")
            self._ignore_path = (self.root / ig_name).resolve()
        except Exception:
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
                except Exception:
                    pass
        except Exception:
            pass

    def _maybe_enqueue(self, src_path: str):
        # Refresh ignore patterns if the file changed
        self._maybe_reload_excluder()
        p = Path(src_path)
        try:
            # normalize to absolute within root
            p = p.resolve()
        except Exception:
            return
        # skip directories
        if p.is_dir():
            return
        # ensure file is under root
        try:
            rel = p.resolve().relative_to(self.root.resolve())
        except ValueError:
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
        try:
            idx.delete_points_by_path(self.client, self.collection, str(p))
            print(f"[deleted] {p}")
            # Drop local cache entry
            try:
                remove_cached_file(str(self.root), str(p))
            except Exception:
                pass

            try:
                _log_activity(str(self.root), "deleted", p)
            except Exception:
                pass
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
            rel_dir = "/" + str(dest.parent.resolve().relative_to(self.root.resolve())).replace(os.sep, "/")
            if rel_dir == "/.":
                rel_dir = "/"
            if self.excl.exclude_dir(rel_dir):
                if src.suffix.lower() in idx.CODE_EXTS:
                    try:
                        idx.delete_points_by_path(self.client, self.collection, str(src))
                        print(f"[moved:ignored_dest_deleted_src] {src} -> {dest}")
                    except Exception:
                        pass
                return
        except Exception:
            pass
        # Try in-place rename (preserve vectors)
        moved_count = -1
        try:
            moved_count = _rename_in_store(self.client, self.collection, src, dest)
        except Exception:
            moved_count = -1
        if moved_count and moved_count > 0:
            try:
                print(f"[moved] {src} -> {dest} ({moved_count} chunk(s) relinked)")
                # Update local cache: carry hash from src to dest if present
                prev_hash = None
                try:
                    prev_hash = get_cached_file_hash(str(self.root), str(src))
                except Exception:
                    prev_hash = None
                if prev_hash:
                    try:
                        set_cached_file_hash(str(self.root), str(dest), prev_hash)
                    except Exception:
                        pass
                    try:
                        remove_cached_file(str(self.root), str(src))
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                _log_activity(str(self.root), "moved", dest, {"from": str(src), "chunks": int(moved_count)})
            except Exception:
                pass
            return
        # Fallback: delete old then index new destination
        try:
            if src.suffix.lower() in idx.CODE_EXTS:
                idx.delete_points_by_path(self.client, self.collection, str(src))
                print(f"[moved:deleted_src] {src}")
        except Exception:
            pass
        try:
            self._maybe_enqueue(str(dest))
        except Exception:
            pass

# --- Workspace state helpers ---
def _set_status_indexing(workspace_path: str, total_files: int) -> None:
    try:
        update_indexing_status(
            workspace_path,
            {
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
        update_indexing_status(
            workspace_path,
            {
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
        update_last_activity(
            workspace_path,
            {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "file_path": str(file_path),
                "details": details or {},
            },
        )
    except Exception:
        pass


# --- Move/Rename optimization: reuse vectors when file content unchanged ---
def _rename_in_store(client: QdrantClient, collection: str, src: Path, dest: Path) -> int:
    """Best-effort: if dest content hash matches previously indexed src hash,
    update points in-place to the new path without re-embedding.

    Returns number of points moved, or -1 if not applicable/failure.
    """
    try:
        if not dest.exists() or dest.is_dir():
            return -1
        try:
            text = dest.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return -1
        dest_hash = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
        prev = idx.get_indexed_file_hash(client, collection, str(src))
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
                collection_name=collection,
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
                idx.upsert_points(client, collection, new_points)
                moved += len(new_points)

        try:
            idx.delete_points_by_path(client, collection, str(src))
        except Exception:
            pass
        return moved
    except Exception:
        return -1


def main():
    print(
        f"Watch mode: root={ROOT} qdrant={QDRANT_URL} collection={COLLECTION} model={MODEL}"
    )

    client = QdrantClient(
        url=QDRANT_URL, timeout=int(os.environ.get("QDRANT_TIMEOUT", "20") or 20)
    )

    # Compute embedding dimension first (for deterministic dense vector selection)
    model = TextEmbedding(model_name=MODEL)
    dim = len(next(model.embed(["dimension probe"])))

    # Determine dense vector name deterministically
    try:
        info = client.get_collection(COLLECTION)
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

    # Ensure collection + payload indexes exist
    try:
        idx.ensure_collection(client, COLLECTION, dim, vector_name)
    except Exception:
        pass
    idx.ensure_payload_indexes(client, COLLECTION)

    # Ensure workspace state exists and set collection
    try:
        update_workspace_state(str(ROOT), {"qdrant_collection": COLLECTION})
        update_indexing_status(str(ROOT), {"state": "watching"})
    except Exception:
        pass

    q = ChangeQueue(lambda paths: _process_paths(paths, client, model, vector_name, str(ROOT)))
    handler = IndexHandler(ROOT, q, client, COLLECTION)

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
    try:
        update_indexing_status(
            workspace_path,
            {
                "state": "indexing",
                "started_at": started_at,
                "progress": {"files_processed": 0, "total_files": total},
            },
        )
    except Exception:
        pass

    processed = 0
    for p in unique_paths:
        current = p
        if not p.exists():
            # File was removed; ensure its points are deleted
            try:
                idx.delete_points_by_path(client, COLLECTION, str(p))
                print(f"[deleted] {p}")
            except Exception:
                pass
            _log_activity(workspace_path, "deleted", p)
            processed += 1
            _update_progress(workspace_path, started_at, processed, total, current)
            continue
        # Lazily instantiate model if needed
        if model is None:
            from fastembed import TextEmbedding
            mname = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            model = TextEmbedding(model_name=mname)
        ok = idx.index_single_file(
            client, model, COLLECTION, vector_name, p, dedupe=True, skip_unchanged=False
        )
        status = "indexed" if ok else "skipped"
        print(f"[{status}] {p}")
        if ok:
            try:
                size = int(p.stat().st_size)
            except Exception:
                size = None
            _log_activity(workspace_path, "indexed", p, {"file_size": size})
        else:
            _log_activity(workspace_path, "skipped", p, {"reason": "no-change-or-error"})
        processed += 1
        _update_progress(workspace_path, started_at, processed, total, current)

    # Return to watching state
    try:
        update_indexing_status(workspace_path, {"state": "watching"})
    except Exception:
        pass


if __name__ == "__main__":
    main()
