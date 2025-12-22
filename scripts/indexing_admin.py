import os
import re
import sys
import time
import subprocess
import shutil
import traceback
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None  # type: ignore

from scripts.collection_admin import copy_collection_qdrant

try:
    from scripts.ingest_code import (
        ensure_collection_and_indexes_once,
        ensure_payload_indexes,
        _sanitize_vector_name,
    )
except Exception:
    ensure_collection_and_indexes_once = None  # type: ignore
    ensure_payload_indexes = None  # type: ignore
    _sanitize_vector_name = None  # type: ignore

try:
    from scripts.workspace_state import (
        get_collection_mappings,
        get_workspace_state,
        update_workspace_state,
        update_indexing_status,
        get_indexing_config_snapshot,
        compute_indexing_config_hash,
        is_staging_enabled,
        set_staging_state,
        update_staging_status,
        clear_staging_collection,
        activate_staging_collection,
    )
except Exception:
    get_collection_mappings = None  # type: ignore
    get_workspace_state = None  # type: ignore
    update_workspace_state = None  # type: ignore
    update_indexing_status = None  # type: ignore
    get_indexing_config_snapshot = None  # type: ignore
    compute_indexing_config_hash = None  # type: ignore
    is_staging_enabled = None  # type: ignore
    set_staging_state = None  # type: ignore
    update_staging_status = None  # type: ignore
    clear_staging_collection = None  # type: ignore
    activate_staging_collection = None  # type: ignore


def _staging_enabled() -> bool:
    return bool(is_staging_enabled() if callable(is_staging_enabled) else False)

try:
    from scripts.embedder import get_model_dimension
except Exception:
    get_model_dimension = None  # type: ignore


def _resolve_codebase_root(work_root: Path) -> Path:
    env_root = (
        os.environ.get("CTXCE_CODEBASE_ROOT")
        or os.environ.get("CODEBASE_ROOT")
        or ""
    ).strip()
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(work_root)
    candidates.append(work_root.parent)
    for candidate in candidates:
        try:
            base = candidate.resolve()
        except Exception:
            base = candidate
        try:
            if (base / ".codebase" / "repos").exists():
                return base
        except Exception:
            continue
    return work_root


def _delete_path_tree(p: Path) -> bool:
    try:
        if not p.exists():
            return False
        if p.is_dir():
            try:
                shutil.rmtree(p)
                return True
            except Exception:
                # Best-effort permission fixup for shared volumes / root-owned files.
                try:
                    for sub in p.rglob("*"):
                        try:
                            if sub.is_dir():
                                os.chmod(sub, 0o777)
                            else:
                                os.chmod(sub, 0o666)
                        except Exception:
                            pass
                    try:
                        os.chmod(p, 0o777)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    shutil.rmtree(p)
                    return True
                except Exception:
                    return False
        p.unlink()
        return True
    except Exception:
        return False


def _cleanup_old_clone(
    *,
    collection: str,
    repo_name: Optional[str],
    delete_collection: bool = True,
    work_dir: Optional[str] = None,
    workspace_root: Optional[str] = None,
) -> None:
    """Best-effort cleanup of cloned *_old collection and workspace/meta."""

    old_collection = f"{collection}_old"
    if delete_collection:
        try:
            delete_collection_qdrant(
                qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
                api_key=os.environ.get("QDRANT_API_KEY") or None,
                collection=old_collection,
            )
        except Exception:
            pass

    if not repo_name:
        return

    old_slug = f"{repo_name}_old"

    work_root: Optional[Path] = None
    if workspace_root:
        try:
            work_root = Path(workspace_root).resolve().parent
        except Exception:
            try:
                work_root = Path(workspace_root).parent
            except Exception:
                work_root = None
    if work_root is None:
        try:
            if work_dir:
                work_root = Path(work_dir).resolve()
            else:
                work_root = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work").resolve()
        except Exception:
            work_root = Path("/work")

    codebase_root = _resolve_codebase_root(work_root)

    # Workspace clone dir
    try:
        _delete_path_tree((work_root / old_slug).resolve())
    except Exception:
        pass

    # Repo metadata dir
    try:
        _delete_path_tree((codebase_root / ".codebase" / "repos" / old_slug).resolve())
    except Exception:
        pass

    # If codebase_root differs from work_root, also try under work_root for safety.
    try:
        if str(codebase_root.resolve()) != str(work_root.resolve()):
            _delete_path_tree((work_root / ".codebase" / "repos" / old_slug).resolve())
    except Exception:
        pass


def current_env_indexing_hash() -> str:
    try:
        if get_indexing_config_snapshot and compute_indexing_config_hash:
            return compute_indexing_config_hash(get_indexing_config_snapshot())
    except Exception:
        return ""
    return ""


def collection_mapping_index(*, work_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    if not get_collection_mappings:
        return {}
    try:
        mappings = get_collection_mappings(search_root=work_dir) or []
    except Exception:
        mappings = []
    out: Dict[str, List[Dict[str, Any]]] = {}
    for m in mappings:
        try:
            name = str(m.get("collection_name") or "").strip()
        except Exception:
            name = ""
        if not name:
            continue
        out.setdefault(name, []).append(m)
    return out


def resolve_collection_root(*, collection: str, work_dir: str) -> Tuple[Optional[str], Optional[str]]:
    name = (collection or "").strip()
    if not name:
        return None, None
    idx = collection_mapping_index(work_dir=work_dir)
    candidates = idx.get(name) or []
    if not candidates:
        return None, None
    chosen = candidates[0]
    repo_name = chosen.get("repo_name")
    container_path = chosen.get("container_path")
    try:
        repo_name = str(repo_name) if repo_name is not None else None
    except Exception:
        repo_name = None
    try:
        container_path = str(container_path) if container_path is not None else None
    except Exception:
        container_path = None
    if container_path:
        return container_path, repo_name
    return work_dir, repo_name


def _get_workspace_state_safe(workspace_path: str, repo_name: Optional[str]) -> Dict[str, Any]:
    if not get_workspace_state:
        return {}
    try:
        state = get_workspace_state(workspace_path, repo_name) or {}
        if isinstance(state, dict):
            return state
        return {}
    except Exception:
        return {}


def get_indexing_state(*, workspace_path: str, repo_name: Optional[str]) -> str:
    if not get_workspace_state:
        return ""
    try:
        st = get_workspace_state(workspace_path, repo_name) or {}
        idx_status = st.get("indexing_status") or {}
        if isinstance(idx_status, dict):
            return str(idx_status.get("state") or "")
    except Exception:
        return ""
    return ""


def build_admin_collections_view(*, collections: Any, work_dir: str) -> List[Dict[str, Any]]:
    env_hash = current_env_indexing_hash()
    mapping_index = collection_mapping_index(work_dir=work_dir)

    enriched: List[Dict[str, Any]] = []
    for c in collections or []:
        try:
            coll_name = str(getattr(c, "qdrant_collection", None) or c.get("qdrant_collection") or "").strip()  # type: ignore[union-attr]
        except Exception:
            coll_name = ""

        applied_hash = ""
        pending_hash = ""
        indexing_state = ""
        indexing_started_at = ""
        progress_files_processed: Optional[int] = None
        progress_total_files: Optional[int] = None
        progress_current_file = ""
        repo_name = None
        container_path = None
        mapping_count = 0

        try:
            matches = mapping_index.get(coll_name) or []
            mapping_count = len(matches)
            if matches:
                m = matches[0]
                repo_name = m.get("repo_name")
                container_path = m.get("container_path")
        except Exception:
            mapping_count = 0

        if container_path and get_workspace_state:
            try:
                st = get_workspace_state(str(container_path), repo_name) or {}
                applied_hash = str(st.get("indexing_config_hash") or "")
                pending_hash = str(st.get("indexing_config_pending_hash") or "")
                idx_status = st.get("indexing_status") or {}
                if isinstance(idx_status, dict):
                    indexing_state = str(idx_status.get("state") or "")
                    indexing_started_at = str(idx_status.get("started_at") or "")
                    progress = idx_status.get("progress") or {}
                    if isinstance(progress, dict):
                        try:
                            progress_files_processed = int(progress.get("files_processed"))
                        except Exception:
                            progress_files_processed = None
                        try:
                            progress_total_files = int(progress.get("total_files"))
                        except Exception:
                            progress_total_files = None
                        try:
                            progress_current_file = str(progress.get("current_file") or "")
                        except Exception:
                            progress_current_file = ""
            except Exception:
                applied_hash = ""
                indexing_state = ""
                indexing_started_at = ""
                progress_files_processed = None
                progress_total_files = None
                progress_current_file = ""

        # Add staging information
        staging_status = "none"
        staging_collection = ""
        staging_state = ""
        if container_path and get_workspace_state:
            try:
                st = get_workspace_state(str(container_path), repo_name) or {}
                staging_info = st.get("staging") or {}
                if isinstance(staging_info, dict) and staging_info.get("collection"):
                    staging_status = "active"
                    staging_collection = str(staging_info.get("collection") or "")
                    staging_status_info = staging_info.get("status") or {}
                    if isinstance(staging_status_info, dict):
                        staging_state = str(staging_status_info.get("state") or "")
                else:
                    staging_status = "none"
            except Exception:
                staging_status = "none"

        # "maintenance needed" should reflect actual config drift requiring a maintenance reindex.
        # Pending hashes can exist during staging / queued rebuild flows; keep them visible in the UI
        # but do not treat them as drift.
        needs_reindex = bool(env_hash and applied_hash and env_hash != applied_hash)

        try:
            cid = getattr(c, "id", None) if hasattr(c, "id") else c.get("id")  # type: ignore[union-attr]
        except Exception:
            cid = None

        enriched.append(
            {
                "id": cid,
                "qdrant_collection": coll_name,
                "container_path": str(container_path) if container_path else "",
                "repo_name": str(repo_name) if repo_name else "",
                "mapping_count": mapping_count,
                "indexing_state": indexing_state,
                "indexing_started_at": indexing_started_at,
                "progress_files_processed": progress_files_processed,
                "progress_total_files": progress_total_files,
                "progress_current_file": progress_current_file,
                "applied_indexing_hash": applied_hash,
                "pending_indexing_hash": pending_hash,
                "current_indexing_hash": env_hash,
                "needs_reindex": needs_reindex,
                "has_mapping": bool(container_path),
                "staging_status": staging_status,
                "staging_collection": staging_collection,
                "staging_state": staging_state,
            }
        )

    return enriched


def delete_collection_qdrant(*, qdrant_url: str, api_key: Optional[str], collection: str) -> None:
    if QdrantClient is None:
        return
    name = (collection or "").strip()
    if not name:
        return
    try:
        cli = QdrantClient(url=qdrant_url, api_key=api_key or None)
    except Exception:
        return
    try:
        cli.delete_collection(collection_name=name)
    except Exception:
        pass
    finally:
        try:
            cli.close()
        except Exception:
            pass


def recreate_collection_qdrant(*, qdrant_url: str, api_key: Optional[str], collection: str) -> None:
    if QdrantClient is None:
        return
    name = (collection or "").strip()
    if not name:
        return
    try:
        cli = QdrantClient(url=qdrant_url, api_key=api_key or None)
    except Exception:
        return
    try:
        try:
            cli.delete_collection(collection_name=name)
        except Exception:
            pass
    finally:
        try:
            cli.close()
        except Exception:
            pass


def spawn_ingest_code(
    *,
    root: str,
    work_dir: str,
    collection: str,
    recreate: bool,
    repo_name: Optional[str],
    env_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    script_path = str((Path(__file__).resolve().parent / "ingest_code.py").resolve())
    cmd = [sys.executable or "python3", script_path, "--root", root, "--no-skip-unchanged"]
    if recreate:
        cmd.append("--recreate")

    env = os.environ.copy()
    # Apply per-run env overrides (e.g. pending env snapshots for staging rebuild).
    if isinstance(env_overrides, dict):
        for k, v in env_overrides.items():
            try:
                key = str(k)
            except Exception:
                continue
            if not key:
                continue
            try:
                env[key] = "" if v is None else str(v)
            except Exception:
                continue
        # When we provide env overrides for a run (e.g. staging rebuild), we also want to
        # force ingest_code to honor the explicit COLLECTION_NAME instead of routing based
        # on per-repo state/serving_collection in multi-repo mode.
        env["CTXCE_FORCE_COLLECTION_NAME"] = "1"
    env["COLLECTION_NAME"] = collection
    env["WATCH_ROOT"] = work_dir
    env["WORKSPACE_PATH"] = work_dir

    try:
        if update_indexing_status:
            update_indexing_status(
                workspace_path=root,
                repo_name=repo_name,
                status={
                    "state": "initializing",
                    "started_at": datetime.now().isoformat(),
                    "progress": {"files_processed": 0, "total_files": None, "current_file": None},
                },
            )
    except Exception:
        pass

    subprocess.Popen(cmd, env=env)


def _determine_embedding_dim(model_name: str) -> int:
    if get_model_dimension:
        try:
            return int(get_model_dimension(model_name))
        except Exception:
            pass
    try:
        from fastembed import TextEmbedding  # type: ignore

        model = TextEmbedding(model_name=model_name)
        return len(next(model.embed(["dimension probe"])))
    except Exception:
        return 1536


def _normalize_cloned_collection_schema(*, collection_name: str, qdrant_url: str) -> None:
    if QdrantClient is None:
        return
    vector_name = None
    model_name = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    dim = _determine_embedding_dim(model_name)
    if _sanitize_vector_name is not None:
        try:
            vector_name = _sanitize_vector_name(model_name)
        except Exception:
            vector_name = None
    try:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    except Exception:
        return
    try:
        # IMPORTANT: This function is called on a freshly cloned "*_old" collection which may
        # be actively serving traffic during a migration. We must not recreate/modify the
        # vector schema here, since Qdrant can't add vector names in-place and the fallback
        # logic in ingest_code can delete+recreate collections (which would drop copied points).
        #
        # We only ensure payload indexes for query performance.
        if ensure_payload_indexes is not None:
            ensure_payload_indexes(client, collection_name)
    except Exception as exc:
        try:
            print(f"[staging] Warning: failed to normalize cloned collection {collection_name}: {exc}")
        except Exception:
            pass
    finally:
        try:
            client.close()
        except Exception:
            pass


def _get_collection_point_count(*, collection_name: str, qdrant_url: str) -> Optional[int]:
    if QdrantClient is None:
        return None
    try:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    except Exception:
        return None
    try:
        try:
            result = client.count(collection_name=collection_name, exact=True)
            return int(getattr(result, "count", 0))
        except Exception:
            return None
    finally:
        try:
            client.close()
        except Exception:
            pass


def _wait_for_clone_points(
    *,
    source_collection: str,
    cloned_collection: str,
    qdrant_url: str,
    expected_count: Optional[int],
    timeout_seconds: int = 60,
) -> None:
    if QdrantClient is None:
        return
    try:
        client = QdrantClient(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)
    except Exception:
        return

    try:
        if expected_count is None or expected_count <= 0:
            # Nothing to wait for (empty collection or unknown count)
            return

        deadline = time.time() + timeout_seconds
        while True:
            try:
                result = client.count(collection_name=cloned_collection, exact=True)
                clone_count = int(getattr(result, "count", 0))
            except Exception:
                clone_count = 0

            if clone_count >= expected_count:
                try:
                    print(
                        f"[staging] Clone verification succeeded: {cloned_collection} has "
                        f"{clone_count} points (expected >= {expected_count})."
                    )
                except Exception:
                    pass
                return

            if time.time() > deadline:
                raise RuntimeError(
                    f"Cloned collection {cloned_collection} only has {clone_count} points "
                    f"(expected at least {expected_count})"
                )

            time.sleep(2)
    finally:
        try:
            client.close()
        except Exception:
            pass


def start_staging_rebuild(*, collection: str, work_dir: str) -> str:
    if not _staging_enabled():
        raise RuntimeError("Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)")
    root, repo_name = resolve_collection_root(collection=collection, work_dir=work_dir)
    if not root:
        raise RuntimeError("No workspace mapping found for collection")

    state = _get_workspace_state_safe(root, repo_name)
    current_staging = state.get("staging") or {}
    if isinstance(current_staging, dict) and current_staging.get("collection"):
        raise RuntimeError("A staging collection is already running for this workspace")

    # Copy current collection to <collection>_old
    old_collection = f"{collection}_old"
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    source_point_count = _get_collection_point_count(collection_name=collection, qdrant_url=qdrant_url)

    global copy_collection_qdrant
    if copy_collection_qdrant is None:
        from scripts.collection_admin import copy_collection_qdrant as _ccq  # re-import for container
        copy_collection_qdrant = _ccq  # type: ignore

    if not callable(copy_collection_qdrant):
        raise RuntimeError("copy_collection_qdrant unavailable (import failed)")

    try:
        print(f"[staging] Copying collection {collection} -> {old_collection} (overwrite=True)")
        try:
            print(
                f"[staging] copy_collection_qdrant callable={callable(copy_collection_qdrant)} type={type(copy_collection_qdrant)} module={getattr(copy_collection_qdrant, '__module__', '?')}"
            )
        except Exception:
            pass
        copy_collection_qdrant(
            source=collection,
            target=old_collection,
            qdrant_url=qdrant_url,
            overwrite=True,
        )
        print(f"[staging] Copy completed for {old_collection}")
    except Exception as exc:
        print(f"[staging] ERROR copying {collection} -> {old_collection}: {exc!r}")
        try:
            print("[staging] TRACEBACK (copy)")
            print(traceback.format_exc())
        except Exception:
            pass
        raise

    try:
        _wait_for_clone_points(
            source_collection=collection,
            cloned_collection=old_collection,
            qdrant_url=qdrant_url,
            expected_count=source_point_count,
            timeout_seconds=90,
        )
    except Exception as exc:
        print(f"[staging] ERROR verifying clone {old_collection}: {exc}")
        raise

    _normalize_cloned_collection_schema(collection_name=old_collection, qdrant_url=qdrant_url)

    # Duplicate workspace slug/state to <slug>_old so watcher/indexer can serve reads from the clone.
    if repo_name:
        work_root = Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
        canonical_dir = work_root / repo_name
        old_dir = work_root / f"{repo_name}_old"
        if canonical_dir.exists():
            shutil.copytree(canonical_dir, old_dir, dirs_exist_ok=True)

        old_state = state.copy()
        # Preserve the env/config that was serving traffic; drop pending snapshots.
        old_state["qdrant_collection"] = old_collection
        old_state["serving_collection"] = old_collection
        old_state["serving_repo_slug"] = f"{repo_name}_old"
        old_state["active_repo_slug"] = old_state.get("active_repo_slug") or repo_name
        try:
            old_state["indexing_status"] = {"state": "idle"}
        except Exception:
            pass
        old_state.pop("indexing_config_pending", None)
        old_state.pop("indexing_config_pending_hash", None)
        old_state.pop("indexing_env_pending", None)
        old_state.pop("staging", None)
        update_workspace_state(
            workspace_path=str(old_dir),
            repo_name=f"{repo_name}_old",
            updates=old_state,
        )

    # Prepare canonical slug for rebuild (pending env)
    pending_cfg = state.get("indexing_config_pending") or state.get("indexing_config")
    pending_hash = state.get("indexing_config_pending_hash") or state.get("indexing_config_hash")
    pending_env = state.get("indexing_env_pending") or dict(os.environ)
    env_hash = pending_hash or current_env_indexing_hash()

    if not pending_cfg and get_indexing_config_snapshot:
        pending_cfg = get_indexing_config_snapshot() if callable(get_indexing_config_snapshot) else get_indexing_config_snapshot
    if not pending_hash and pending_cfg and compute_indexing_config_hash:
        pending_hash = compute_indexing_config_hash(pending_cfg)

    if set_staging_state:
        staging_info = {
            "collection": old_collection,
            "started_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "env_hash": pending_hash or env_hash,
            "indexing_config": pending_cfg,
            "indexing_config_hash": pending_hash,
            "environment": pending_env,
            "status": {"state": "initializing"},
            "workspace_path": root,
            "repo_name": repo_name,
        }
        set_staging_state(workspace_path=root, repo_name=repo_name, staging=staging_info)

    update_workspace_state(
        workspace_path=root,
        repo_name=repo_name,
        updates={
            "serving_collection": old_collection,
            "serving_repo_slug": f"{repo_name}_old",
            "active_repo_slug": repo_name,
            "qdrant_collection": collection,
        },
    )

    if update_staging_status:
        update_staging_status(
            workspace_path=root,
            repo_name=repo_name,
            status={
                "state": "initializing",
                "started_at": datetime.utcnow().isoformat(),
                "progress": {"files_processed": 0, "total_files": None, "current_file": None},
            },
        )

    recreate_collection_qdrant(
        qdrant_url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        collection=collection,
    )
    spawn_ingest_code(
        root=root,
        work_dir=work_dir,
        collection=collection,
        recreate=True,
        repo_name=repo_name,
        env_overrides=pending_env if isinstance(pending_env, dict) else None,
    )
    return collection


def activate_staging_rebuild(*, collection: str, work_dir: str) -> None:
    if not _staging_enabled():
        raise RuntimeError("Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)")
    root, repo_name = resolve_collection_root(collection=collection, work_dir=work_dir)
    if not root:
        raise RuntimeError("No workspace mapping found for collection")
    state = _get_workspace_state_safe(root, repo_name)
    staging = state.get("staging") or {}

    # Idempotent: allow Activate to run even if staging metadata was lost,
    # as long as serving is still pointed at *_old (or the clone artifacts exist).
    staging_active = False
    try:
        if isinstance(staging, dict) and staging.get("collection"):
            staging_active = True
    except Exception:
        staging_active = False
    try:
        if str(state.get("serving_collection") or "").strip() == f"{collection}_old":
            staging_active = True
    except Exception:
        pass
    try:
        if str(state.get("serving_repo_slug") or "").strip().endswith("_old"):
            staging_active = True
    except Exception:
        pass

    if not staging_active:
        # Nothing to activate.
        return
    # This staging workflow serves reads from <collection>_old while recreating/reindexing
    # the primary <collection>. "Activate" means: switch serving back to the rebuilt primary
    # and remove the *_old clone.

    _cleanup_old_clone(
        collection=collection,
        repo_name=repo_name,
        delete_collection=True,
        work_dir=work_dir,
        workspace_root=root,
    )

    # Reset serving state back to the primary collection and clear staging metadata.
    if update_workspace_state and repo_name:
        try:
            update_workspace_state(
                workspace_path=root,
                repo_name=repo_name,
                updates={
                    "serving_collection": collection,
                    "serving_repo_slug": repo_name,
                    "active_repo_slug": repo_name,
                    "qdrant_collection": collection,
                },
            )
        except Exception:
            pass

    if clear_staging_collection:
        try:
            clear_staging_collection(workspace_path=root, repo_name=repo_name)
        except Exception:
            pass


def abort_staging_rebuild(
    *,
    collection: str,
    work_dir: str,
    delete_collection: bool = True,
) -> None:
    if not _staging_enabled():
        raise RuntimeError("Staging is disabled (set CTXCE_STAGING_ENABLED=1 to enable)")
    if not clear_staging_collection:
        raise RuntimeError("clear_staging_collection helper unavailable")
    root, repo_name = resolve_collection_root(collection=collection, work_dir=work_dir)
    if not root:
        raise RuntimeError("No workspace mapping found for collection")
    state = _get_workspace_state_safe(root, repo_name)
    staging = state.get("staging") or {}

    # Idempotent: allow Abort to run even if staging metadata was lost,
    # as long as serving is still pointed at *_old (or the clone artifacts exist).
    staging_active = False
    try:
        if isinstance(staging, dict) and staging.get("collection"):
            staging_active = True
    except Exception:
        staging_active = False
    try:
        if str(state.get("serving_collection") or "").strip() == f"{collection}_old":
            staging_active = True
    except Exception:
        pass
    try:
        if str(state.get("serving_repo_slug") or "").strip().endswith("_old"):
            staging_active = True
    except Exception:
        pass

    if not staging_active:
        # Nothing to abort.
        return

    # Staging rebuild serves traffic from <collection>_old while recreating <collection>.
    # Abort should restore serving state to the primary collection and remove the *_old clone.
    _cleanup_old_clone(
        collection=collection,
        repo_name=repo_name,
        delete_collection=delete_collection,
        work_dir=work_dir,
        workspace_root=root,
    )

    # Reset serving state back to the primary collection.
    if update_workspace_state and repo_name:
        try:
            update_workspace_state(
                workspace_path=root,
                repo_name=repo_name,
                updates={
                    "serving_collection": collection,
                    "serving_repo_slug": repo_name,
                    "active_repo_slug": repo_name,
                    "qdrant_collection": collection,
                },
            )
        except Exception:
            pass

    clear_staging_collection(workspace_path=root, repo_name=repo_name)

    # Cleanup handled by _cleanup_old_clone
