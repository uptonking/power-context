import io
import json
import tarfile
from pathlib import Path

import pytest


def _write_bundle(tmp_path: Path, operations: list[dict]) -> Path:
    bundle_path = tmp_path / "bundle.tar.gz"
    payload = json.dumps({"operations": operations}).encode("utf-8")

    with tarfile.open(bundle_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="metadata/operations.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    return bundle_path


def test_process_delta_bundle_rejects_traversal_created(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    bundle = _write_bundle(
        tmp_path,
        [{"operation": "created", "path": "../../evil.txt"}],
    )

    with pytest.raises(ValueError, match="escapes workspace"):
        us.process_delta_bundle(
            workspace_path="/home/user/repo",
            bundle_path=bundle,
            manifest={"bundle_id": "b1"},
        )


def test_process_delta_bundle_rejects_absolute_paths(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    bundle = _write_bundle(
        tmp_path,
        [{"operation": "created", "path": "/etc/passwd"}],
    )

    with pytest.raises(ValueError, match="Absolute paths"):
        us.process_delta_bundle(
            workspace_path="/home/user/repo",
            bundle_path=bundle,
            manifest={"bundle_id": "b1"},
        )


def test_process_delta_bundle_rejects_traversal_moved_source(tmp_path, monkeypatch):
    import scripts.upload_delta_bundle as us

    work_dir = tmp_path / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(us, "WORK_DIR", str(work_dir))

    bundle = _write_bundle(
        tmp_path,
        [
            {
                "operation": "moved",
                "path": "dst.txt",
                "source_path": "../../escape.txt",
            }
        ],
    )

    with pytest.raises(ValueError, match="escapes workspace"):
        us.process_delta_bundle(
            workspace_path="/home/user/repo",
            bundle_path=bundle,
            manifest={"bundle_id": "b1"},
        )
