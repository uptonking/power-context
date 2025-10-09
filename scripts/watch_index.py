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
    def __init__(self, root: Path, queue: ChangeQueue):
        super().__init__()
        self.root = root
        self.queue = queue
        self.excl = idx._Excluder(root)


    def _maybe_enqueue(self, src_path: str):
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



def main():
    print(f"Watch mode: root={ROOT} qdrant={QDRANT_URL} collection={COLLECTION} model={MODEL}")

    client = QdrantClient(url=QDRANT_URL)

    # Compute embedding dimension first (for deterministic dense vector selection)
    model = TextEmbedding(model_name=MODEL)
    dim = len(next(model.embed(["dimension probe"])) )

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
            if vector_name is None and getattr(idx, 'LEX_VECTOR_NAME', None) in cfg:
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

    q = ChangeQueue(lambda paths: _process_paths(paths, client, model, vector_name))
    handler = IndexHandler(ROOT, q)

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


def _process_paths(paths, client, model, vector_name: str):
    # De-duplicate and index each path
    for p in sorted(set(Path(x) for x in paths)):
        if not p.exists():
            continue
        # Lazily instantiate model if needed
        if model is None:
            from fastembed import TextEmbedding
            mname = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            model = TextEmbedding(model_name=mname)
        ok = idx.index_single_file(client, model, COLLECTION, vector_name, p, dedupe=True, skip_unchanged=False)
        status = "indexed" if ok else "skipped"
        print(f"[{status}] {p}")


if __name__ == "__main__":
    main()

