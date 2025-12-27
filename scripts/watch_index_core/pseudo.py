"""Background pseudo backfill worker for the watcher."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import scripts.ingest_code as idx
from .utils import get_boolean_env
from scripts.workspace_state import (
    _cross_process_lock,
    _get_global_state_dir,
    _get_repo_state_dir,
    get_collection_mappings,
    is_multi_repo_mode,
)

from .config import ROOT


def _start_pseudo_backfill_worker(
    client,
    default_collection: str,
    model_dim: int,
    vector_name: str,
) -> None:
    """Start a daemon thread that periodically backfills pseudo/tags."""
    
    if not get_boolean_env("PSEUDO_BACKFILL_ENABLED"):
        return

    try:
        interval = float(os.environ.get("PSEUDO_BACKFILL_TICK_SECS", "60") or 60.0)
    except Exception:
        interval = 60.0
    if interval <= 0:
        return
    try:
        max_points = int(os.environ.get("PSEUDO_BACKFILL_MAX_POINTS", "256") or 256)
    except Exception:
        max_points = 256
    if max_points <= 0:
        max_points = 1

    def _worker() -> None:
        while True:
            try:
                try:
                    mappings = get_collection_mappings(search_root=str(ROOT))
                except Exception:
                    mappings = []
                if not mappings:
                    mappings = [
                        {"repo_name": None, "collection_name": default_collection},
                    ]
                for mapping in mappings:
                    coll = mapping.get("collection_name") or default_collection
                    repo_name: Optional[str] = mapping.get("repo_name")
                    if not coll:
                        continue
                    try:
                        if is_multi_repo_mode() and repo_name:
                            state_dir = _get_repo_state_dir(repo_name)
                        else:
                            state_dir = _get_global_state_dir(str(ROOT))
                        lock_path = state_dir / "pseudo.lock"
                        with _cross_process_lock(lock_path):
                            processed = idx.pseudo_backfill_tick(
                                client,
                                coll,
                                repo_name=repo_name,
                                max_points=max_points,
                                dim=model_dim,
                                vector_name=vector_name,
                            )
                            if processed:
                                try:
                                    print(
                                        f"[pseudo_backfill] repo={repo_name or 'default'} "
                                        f"collection={coll} processed={processed}"
                                    )
                                except Exception:
                                    pass
                    except Exception as exc:
                        try:
                            print(
                                f"[pseudo_backfill] error repo={repo_name or 'default'} "
                                f"collection={coll}: {exc}"
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(interval)

    thread = threading.Thread(target=_worker, name="pseudo-backfill", daemon=True)
    thread.start()


__all__ = ["_start_pseudo_backfill_worker"]
