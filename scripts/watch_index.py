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
        # honor skip dirs by path parts
        for part in p.parts:
            if part in idx.SKIP_DIRS:
                return
        # only code files
        if p.suffix.lower() not in idx.CODE_EXTS:
            return
        # ensure file is under root
        try:
            p.relative_to(self.root)
        except ValueError:
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
    model = TextEmbedding(model_name=MODEL)

    # Determine vector name
    try:
        info = client.get_collection(COLLECTION)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict) and cfg:
            vector_name = list(cfg.keys())[0]
        else:
            vector_name = idx._sanitize_vector_name(MODEL)
    except Exception:
        vector_name = idx._sanitize_vector_name(MODEL)

    # Ensure collection + payload indexes exist
    dim = len(next(model.embed(["dimension probe"])) )
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
        ok = idx.index_single_file(client, model, COLLECTION, vector_name, p, dedupe=True, skip_unchanged=False)
        status = "indexed" if ok else "skipped"
        print(f"[{status}] {p}")


if __name__ == "__main__":
    main()

