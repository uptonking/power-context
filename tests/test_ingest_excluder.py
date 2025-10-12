import os
from pathlib import Path
import importlib

ing = importlib.import_module("scripts.ingest_code")


def test_excluder_defaults_and_ignore(tmp_path, monkeypatch):
    root = tmp_path
    # Create files and dirs
    (root / ".git").mkdir()
    (root / "node_modules").mkdir()
    (root / "keep.py").write_text("print('ok')\n")
    (root / "skip.log").write_text("log\n")

    # .qdrantignore excludes *.log and folder tmp/
    (root / ".qdrantignore").write_text("""\n*.log\ntmp/\n""")

    excl = ing._Excluder(root)
    # Defaults should exclude .git and node_modules directories
    assert excl.exclude_dir(Path(".git"))
    assert excl.exclude_dir(Path("node_modules"))
    # .qdrantignore should exclude .log files
    assert excl.exclude_file(Path("skip.log"))
    # keep.py should be allowed
    assert not excl.exclude_file(Path("keep.py"))


def test_excluder_env_overrides(tmp_path, monkeypatch):
    root = tmp_path
    (root / "a").mkdir()
    (root / "a" / "file.tmp").write_text("x\n")

    # Disable defaults and add custom patterns
    monkeypatch.setenv("QDRANT_DEFAULT_EXCLUDES", "0")
    monkeypatch.setenv("QDRANT_EXCLUDE_DIRS", "build,dist")
    monkeypatch.setenv("QDRANT_EXCLUDE_FILES", "*.tmp,*.bak")

    excl = ing._Excluder(root)
    assert not excl.exclude_dir(Path(".git"))  # defaults off
    assert excl.exclude_file(Path("a/file.tmp"))

