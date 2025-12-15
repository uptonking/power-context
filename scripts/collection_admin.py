import os
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from scripts.auth_backend import mark_collection_deleted

try:
    from scripts.qdrant_client_manager import pooled_qdrant_client
except Exception:
    pooled_qdrant_client = None

try:
    from scripts.workspace_state import get_collection_mappings
except Exception:
    get_collection_mappings = None


_SLUGGED_REPO_RE = re.compile(r"^.+-[0-9a-f]{16}$")
_MARKER_NAME = ".ctxce_managed_upload"


def _resolve_work_root(work_dir: Optional[str] = None) -> Path:
    return Path(
        work_dir
        or os.environ.get("WORK_DIR")
        or os.environ.get("WORKDIR")
        or "/work"
    ).resolve()


def _resolve_codebase_root(work_root: Path) -> Path:
    env_root = (
        os.environ.get("CTXCE_CODEBASE_ROOT")
        or os.environ.get("CODEBASE_ROOT")
        or ""
    ).strip()

    candidates = []
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
            shutil.rmtree(p)
            return True
        p.unlink()
        return True
    except Exception:
        return False


def _read_state_collection(state_path: Path) -> str:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return str(data.get("qdrant_collection") or "").strip()
    except Exception:
        pass
    return ""


def _managed_upload_marker_path(
    *,
    work_root: Path,
    slug_name: str,
    marker_root: Optional[Path] = None,
) -> Path:
    # Marker is stored with per-repo metadata, not inside the repo workspace tree.
    base = marker_root or work_root
    return base / ".codebase" / "repos" / slug_name / _MARKER_NAME


def _is_managed_upload_workspace_dir(
    p: Path,
    *,
    work_root: Path,
    marker_root: Optional[Path] = None,
) -> bool:
    try:
        if not p.is_dir():
            return False
        if p.parent.resolve() != work_root:
            return False
        if not _SLUGGED_REPO_RE.match(p.name or ""):
            return False
        return _managed_upload_marker_path(
            work_root=work_root,
            marker_root=marker_root,
            slug_name=p.name,
        ).exists()
    except Exception:
        return False


def _cleanup_state_files_for_mapping(
    mapping: Dict[str, Any],
    *,
    work_root: Optional[Path] = None,
    codebase_root: Optional[Path] = None,
) -> int:
    removed = 0
    state_file = mapping.get("state_file")
    if isinstance(state_file, str) and state_file.strip():
        p = Path(state_file)
        state_dir = p.parent
        try:
            base = (
                codebase_root
                or work_root
                or Path(os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work")
            ).resolve()
        except Exception:
            base = codebase_root or work_root or Path(
                os.environ.get("WORK_DIR") or os.environ.get("WORKDIR") or "/work"
            )

        # Multi-repo mode stores per-repo metadata under /work/.codebase/repos/<repo>/
        try:
            repos_root = (base / ".codebase" / "repos").resolve()
            sd = state_dir.resolve()
            if sd != repos_root and str(sd).startswith(str(repos_root) + os.sep):
                if _delete_path_tree(sd):
                    removed += 1
                return removed
        except Exception:
            pass

        # Single-repo mode stores metadata under <workspace>/.codebase/
        if _delete_path_tree(p):
            removed += 1
        _delete_path_tree(state_dir / "cache.json")
        try:
            for sym in state_dir.glob("symbols_*.json"):
                _delete_path_tree(sym)
        except Exception:
            pass
    return removed


def delete_collection_everywhere(
    *,
    collection: str,
    work_dir: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    cleanup_fs: bool = True,
) -> Dict[str, Any]:
    enabled = (
        str(os.environ.get("CTXCE_ADMIN_COLLECTION_DELETE_ENABLED", "0")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    if not enabled:
        raise PermissionError("Collection deletion is disabled by server configuration")

    name = (collection or "").strip()
    if not name:
        raise ValueError("collection is required")

    work_root = _resolve_work_root(work_dir)
    codebase_root = _resolve_codebase_root(work_root)

    out: Dict[str, Any] = {
        "collection": name,
        "qdrant_deleted": False,
        "registry_marked_deleted": False,
        "deleted_state_files": 0,
        "deleted_managed_workspaces": 0,
    }

    # 1) Delete Qdrant collection
    try:
        if pooled_qdrant_client is not None:
            with pooled_qdrant_client(url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY")) as cli:
                try:
                    cli.delete_collection(collection_name=name)
                    out["qdrant_deleted"] = True
                except Exception:
                    out["qdrant_deleted"] = False
    except Exception:
        out["qdrant_deleted"] = False

    # 2) Mark deleted in registry DB
    try:
        mark_collection_deleted(name)
        out["registry_marked_deleted"] = True
    except Exception:
        out["registry_marked_deleted"] = False

    # 3) Cleanup workspace state metadata + managed upload workspaces
    if not cleanup_fs:
        return out

    mappings = []
    try:
        if get_collection_mappings is not None:
            mappings = get_collection_mappings(search_root=str(codebase_root)) or []
    except Exception:
        mappings = []

    # NOTE: logically linked worktrees still share the
    # primary repo's qdrant_collection in their state.json files. When the UI targets the lineage
    # collection directly, no mapping below matches, so filesystem cleanup is a no-op and we keep
    # both the metadata (`.codebase/repos/<slug>`) and the workspace on disk. This conservative
    # behavior is intentional until we have branch-aware deletion semantics—we do not want to
    # cascade-delete shared worktrees that may host future branch/version state.
    for m in mappings:
        try:
            if str(m.get("collection_name") or "").strip() != name:
                continue
            container_path = m.get("container_path")
            if isinstance(container_path, str) and container_path.strip():
                p = Path(container_path)
                try:
                    p = p.resolve()
                except Exception:
                    pass
                if _is_managed_upload_workspace_dir(p, work_root=work_root, marker_root=codebase_root):
                    if _delete_path_tree(p):
                        out["deleted_managed_workspaces"] += 1

            # Cleanup state metadata after workspace deletion so the marker still exists
            # when authorizing the filesystem delete.
            out["deleted_state_files"] += _cleanup_state_files_for_mapping(
                m,
                work_root=work_root,
                codebase_root=codebase_root,
            )
        except Exception:
            continue

    return out
