"""Index management commands: index, prune, status, list-collections."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cli.core import (
    MODEL_NAME,
    QDRANT_URL,
    get_client,
    output_json,
    resolve_collection,
)


def cmd_index(args: argparse.Namespace) -> None:
    """Index a directory (or full workspace) into Qdrant.

    Calls index_repo() with the same signature the MCP server uses
    (via subprocess to ingest_code.py).
    """
    from scripts.ingest.pipeline import index_repo

    root = Path(getattr(args, "path", ".")).resolve()
    collection = resolve_collection(
        getattr(args, "collection", None),
        workspace_path=root,
    )
    subdir = getattr(args, "subdir", None)
    target = root / subdir if subdir else root
    recreate = getattr(args, "recreate", False)
    api_key = os.environ.get("QDRANT_API_KEY", "")

    print(f"Indexing {target} → collection={collection}", file=sys.stderr)
    index_repo(
        root=target,
        qdrant_url=QDRANT_URL,
        api_key=api_key,
        collection=collection,
        model_name=MODEL_NAME,
        recreate=recreate,
    )

    client = get_client()
    info = client.get_collection(collection)
    output_json({
        "ok": True,
        "collection": collection,
        "indexed_path": str(target),
        "points_count": info.points_count,
    })


def cmd_prune(args: argparse.Namespace) -> None:
    """Remove stale points (deleted/moved files) from the index.

    Mirrors scripts/prune.py logic: checks metadata.path for existence
    and metadata.file_hash for staleness, using the correct collection.
    """
    from scripts.prune import sha1_file
    from qdrant_client import models

    client = get_client()
    root = Path(getattr(args, "path", ".")).resolve()
    collection = resolve_collection(
        getattr(args, "collection", None),
        workspace_path=root,
    )

    seen: set[str] = set()
    removed_missing = 0
    removed_stale = 0
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=256, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for pt in points:
            md = (pt.payload or {}).get("metadata") or {}
            path_str = md.get("path", "")
            if not path_str or path_str in seen:
                continue
            seen.add(path_str)

            # Resolve absolute path (handle /work/ prefix from container)
            if path_str.startswith("/work/"):
                abs_path = root / Path(path_str).relative_to("/work")
            else:
                abs_path = root / path_str

            flt = models.Filter(must=[
                models.FieldCondition(
                    key="metadata.path",
                    match=models.MatchValue(value=path_str),
                )
            ])

            if not abs_path.exists():
                client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=flt),
                )
                removed_missing += 1
                print(f"[prune] removed missing: {path_str}", file=sys.stderr)
                continue

            # Check hash mismatch (stale content)
            file_hash = md.get("file_hash", "")
            if file_hash:
                current_hash = sha1_file(abs_path)
                if current_hash and current_hash != file_hash:
                    client.delete(
                        collection_name=collection,
                        points_selector=models.FilterSelector(filter=flt),
                    )
                    removed_stale += 1
                    print(f"[prune] removed stale: {path_str}", file=sys.stderr)

        if offset is None:
            break

    output_json({
        "ok": True,
        "collection": collection,
        "removed_missing": removed_missing,
        "removed_stale": removed_stale,
    })


def cmd_status(args: argparse.Namespace) -> None:
    """Show collection status and health info."""
    client = get_client()
    collection = resolve_collection(getattr(args, "collection", None))
    try:
        info = client.get_collection(collection)
        output_json({
            "ok": True,
            "collection": collection,
            "status": str(info.status),
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
        })
    except Exception as e:
        output_json({"ok": False, "error": str(e)})


def cmd_list_collections(args: argparse.Namespace) -> None:
    """List all Qdrant collections."""
    client = get_client()
    collections = client.get_collections().collections
    output_json({
        "ok": True,
        "collections": [{"name": c.name} for c in collections],
    })
