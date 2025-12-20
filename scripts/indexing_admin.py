import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None  # type: ignore

try:
    from scripts.workspace_state import (
        get_collection_mappings,
        get_workspace_state,
        update_indexing_status,
        get_indexing_config_snapshot,
        compute_indexing_config_hash,
    )
except Exception:
    get_collection_mappings = None  # type: ignore
    get_workspace_state = None  # type: ignore
    update_indexing_status = None  # type: ignore
    get_indexing_config_snapshot = None  # type: ignore
    compute_indexing_config_hash = None  # type: ignore


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
                "current_indexing_hash": env_hash,
                "needs_reindex": needs_reindex,
                "has_mapping": bool(container_path),
            }
        )

    return enriched


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
) -> None:
    script_path = str((Path(__file__).resolve().parent / "ingest_code.py").resolve())
    cmd = [sys.executable or "python3", script_path, "--root", root, "--no-skip-unchanged"]
    if recreate:
        cmd.append("--recreate")

    env = os.environ.copy()
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
