"""Watch command: auto-reindex on file changes (daemon mode)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from cli.core import (
    MODEL_NAME,
    get_client,
    get_model,
    get_model_dim,
    resolve_collection,
)


def cmd_watch(args: argparse.Namespace) -> None:
    """Watch a directory for changes and auto-reindex."""
    root = Path(getattr(args, "path", ".")).resolve()
    # watch_index_core.config resolves WATCH_ROOT at import time.
    os.environ["WATCH_ROOT"] = str(root)
    os.environ.setdefault("WORKSPACE_PATH", str(root))

    collection_arg = getattr(args, "collection", None)

    from watchdog.observers import Observer
    from scripts.watch_index_core.handler import IndexHandler
    from scripts.watch_index_core.processor import _process_paths
    from scripts.watch_index_core.queue import ChangeQueue
    from scripts.watch_index_core.pseudo import _start_pseudo_backfill_worker
    from scripts.watch_index_core.utils import (
        create_observer,
        get_boolean_env,
        resolve_vector_name_config,
    )
    from scripts.workspace_state import initialize_watcher_state, is_multi_repo_mode
    import scripts.ingest_code as idx

    try:
        multi_repo_enabled = bool(is_multi_repo_mode())
    except Exception:
        multi_repo_enabled = False

    client = get_client()
    model = get_model()
    model_dim = get_model_dim()
    collection = resolve_collection(collection_arg, workspace_path=root)

    vector_name = resolve_vector_name_config(client, collection, model_dim, MODEL_NAME)
    ensure_default_collection = bool(collection_arg) or get_boolean_env(
        "WATCH_ENSURE_DEFAULT_COLLECTION",
        default=(False if multi_repo_enabled else True),
    )
    if ensure_default_collection:
        idx.ensure_collection_and_indexes_once(client, collection, model_dim, vector_name)
        try:
            _start_pseudo_backfill_worker(client, collection, model_dim, vector_name)
        except Exception:
            pass

    try:
        initialize_watcher_state(str(root), multi_repo_enabled, collection)
    except Exception:
        pass

    mode = "multi-repo" if multi_repo_enabled else "single-repo"
    print(f"Watching {root} → collection={collection} model={MODEL_NAME}", file=sys.stderr)
    print(f"Watch mode: {mode}", file=sys.stderr)

    collection_override = collection if collection_arg else None
    q = ChangeQueue(
        lambda paths: _process_paths(
            paths,
            client,
            model,
            vector_name,
            model_dim,
            str(root),
            collection_override=collection_override,
        )
    )
    handler = IndexHandler(
        root,
        q,
        client,
        collection,
        collection=collection_override,
    )

    use_polling = get_boolean_env("WATCH_USE_POLLING")
    obs = create_observer(use_polling, observer_cls=Observer)
    obs.schedule(handler, str(root), recursive=True)
    obs.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping watcher...", file=sys.stderr)
    finally:
        obs.stop()
        obs.join()
