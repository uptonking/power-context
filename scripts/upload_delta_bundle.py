import os
import json
import tarfile
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any


try:
    from scripts.workspace_state import _extract_repo_name_from_path
except ImportError:
    _extract_repo_name_from_path = None


logger = logging.getLogger(__name__)

WORK_DIR = os.environ.get("WORK_DIR", "/work")


def get_workspace_key(workspace_path: str) -> str:
    """Generate 16-char hash for collision avoidance in remote uploads.

    Remote uploads may have identical folder names from different users,
    so uses longer hash than local indexing (8-chars) to ensure uniqueness.

    Both host paths (/home/user/project/repo) and container paths (/work/repo)
    should generate the same key for the same repository.
    """
    repo_name = Path(workspace_path).name
    return hashlib.sha256(repo_name.encode("utf-8")).hexdigest()[:16]


def _cleanup_empty_dirs(path: Path, stop_at: Path) -> None:
    """Recursively remove empty directories up to stop_at (exclusive)."""
    try:
        path = path.resolve()
        stop_at = stop_at.resolve()
    except Exception:
        pass
    while True:
        try:
            if path == stop_at or not path.exists() or not path.is_dir():
                break
            if any(path.iterdir()):
                break
            path.rmdir()
            path = path.parent
        except Exception:
            break


def process_delta_bundle(workspace_path: str, bundle_path: Path, manifest: Dict[str, Any]) -> Dict[str, int]:
    """Process delta bundle and return operation counts."""
    operations_count = {
        "created": 0,
        "updated": 0,
        "deleted": 0,
        "moved": 0,
        "skipped": 0,
        "failed": 0,
    }

    try:
        # CRITICAL: Always materialize writes under WORK_DIR using a slugged repo directory.
        # Do NOT write directly into the client-supplied workspace_path, since that may be a host
        # path (e.g. /home/user/repo) that is not mounted/visible to the watcher/indexer.
        if _extract_repo_name_from_path:
            repo_name = _extract_repo_name_from_path(workspace_path)
            if not repo_name:
                repo_name = Path(workspace_path).name
        else:
            repo_name = Path(workspace_path).name

        # Workspace slug: <repo_name>-<16charhash>. This ensures uniqueness across users/workspaces
        # that may share the same leaf folder name.
        workspace_key = get_workspace_key(workspace_path)
        workspace = Path(WORK_DIR) / f"{repo_name}-{workspace_key}"
        workspace.mkdir(parents=True, exist_ok=True)
        slug_repo_name = f"{repo_name}-{workspace_key}"

        workspace_root = workspace.resolve()

        def _safe_join(base: Path, rel: str) -> Path:
            # SECURITY: Prevent path traversal / absolute-path writes by ensuring the resolved
            # candidate path stays within the intended workspace root.
            rp = Path(str(rel))
            if str(rp) in {".", ""}:
                raise ValueError("Invalid operation path")
            if rp.is_absolute():
                raise ValueError(f"Absolute paths are not allowed: {rel}")
            base_resolved = base.resolve()
            candidate = (base_resolved / rp).resolve()
            try:
                ok = candidate.is_relative_to(base_resolved)
            except Exception:
                ok = os.path.commonpath([str(base_resolved), str(candidate)]) == str(base_resolved)
            if not ok:
                raise ValueError(f"Path escapes workspace: {rel}")
            return candidate

        with tarfile.open(bundle_path, "r:gz") as tar:
            ops_member = None
            for member in tar.getnames():
                if member.endswith("metadata/operations.json"):
                    ops_member = member
                    break

            if not ops_member:
                raise ValueError("operations.json not found in bundle")

            ops_file = tar.extractfile(ops_member)
            if not ops_file:
                raise ValueError("Cannot extract operations.json")

            operations_data = json.loads(ops_file.read().decode("utf-8"))
            operations = operations_data.get("operations", [])

            # Best-effort: extract git history metadata for watcher to ingest
            try:
                git_member = None
                for member in tar.getnames():
                    if member.endswith("metadata/git_history.json"):
                        git_member = member
                        break
                if git_member:
                    git_file = tar.extractfile(git_member)
                    if git_file:
                        history_bytes = git_file.read()
                        history_dir = workspace / ".remote-git"
                        history_dir.mkdir(parents=True, exist_ok=True)
                        bundle_id = manifest.get("bundle_id") or "unknown"
                        history_path = history_dir / f"git_history_{bundle_id}.json"
                        try:
                            history_path.write_bytes(history_bytes)
                        except Exception as write_err:
                            logger.debug(
                                f"[upload_service] Failed to write git history manifest: {write_err}",
                            )
            except Exception as git_err:
                logger.debug(f"[upload_service] Error extracting git history metadata: {git_err}")

            for operation in operations:
                op_type = operation.get("operation")
                rel_path = operation.get("path")

                if not rel_path:
                    operations_count["skipped"] += 1
                    continue

                # Defensive guard: if the operation path already includes the slugged repo name
                # ("<repo>-<hash>/..."), then writing it under workspace_root would create
                # a nested slug directory ("slug/slug/..."), which is almost always client misuse.
                if rel_path == slug_repo_name or rel_path.startswith(slug_repo_name + "/"):
                    msg = (
                        f"[upload_service] Refusing to apply operation {op_type} for suspicious path {rel_path} "
                        f"which already contains workspace slug {slug_repo_name}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                target_path = _safe_join(workspace_root, rel_path)

                safe_source_path = None
                source_rel_path = None
                if op_type == "moved":
                    source_rel_path = operation.get("source_path") or operation.get("source_relative_path")
                    if source_rel_path:
                        safe_source_path = _safe_join(workspace_root, source_rel_path)

                try:
                    if op_type == "created":
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/created/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                operations_count["created"] += 1
                            else:
                                operations_count["failed"] += 1
                        else:
                            operations_count["failed"] += 1

                    elif op_type == "updated":
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/updated/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                operations_count["updated"] += 1
                            else:
                                operations_count["failed"] += 1
                        else:
                            operations_count["failed"] += 1

                    elif op_type == "moved":
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/moved/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                operations_count["moved"] += 1
                            else:
                                operations_count["failed"] += 1
                        else:
                            operations_count["failed"] += 1

                        if safe_source_path is not None and source_rel_path:
                            if safe_source_path.exists():
                                try:
                                    safe_source_path.unlink()
                                    operations_count["deleted"] += 1
                                    _cleanup_empty_dirs(safe_source_path.parent, workspace)
                                except Exception as del_err:
                                    logger.error(
                                        f"Error deleting source file for move {source_rel_path}: {del_err}",
                                    )

                    elif op_type == "deleted":
                        if target_path.exists():
                            target_path.unlink()
                            _cleanup_empty_dirs(target_path.parent, workspace)
                            operations_count["deleted"] += 1
                        else:
                            operations_count["skipped"] += 1

                    else:
                        operations_count["skipped"] += 1

                except Exception as e:
                    logger.error(f"Error processing operation {op_type} for {rel_path}: {e}")
                    operations_count["failed"] += 1

        return operations_count

    except Exception as e:
        logger.error(f"Error processing delta bundle: {e}")
        raise
