import os
import uuid
import subprocess
import sys
import pytest
import importlib

pytestmark = pytest.mark.integration


# Always enabled; smoke runs as part of full suite


@pytest.fixture(scope="module")
def qdrant_container():
    try:
        from testcontainers.core.container import DockerContainer
    except Exception:
        pytest.skip("testcontainers not available")
    import time, urllib.request

    # Disable ryuk to avoid port mapping flakiness in some environments
    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    os.environ.setdefault("TESTCONTAINERS_RYUK_TIMEOUT", "0")
    container = (
        DockerContainer("qdrant/qdrant:latest")
        .with_env("TESTCONTAINERS_RYUK_DISABLED", "true")
        .with_env("TESTCONTAINERS_RYUK_TIMEOUT", "0")
        .with_exposed_ports(6333)
    )
    ready = False
    try:
        container.start()
        host = container.get_container_host_ip()
        deadline = time.time() + 30
        last_exc: Exception | None = None
        port = None
        while time.time() < deadline:
            try:
                port = int(container.get_exposed_port(6333))
                break
            except Exception as exc:  # pragma: no cover - retry only in flaky envs
                last_exc = exc
                time.sleep(0.25)
        else:
            raise RuntimeError(f"qdrant port mapping unavailable: {last_exc}")
        url = f"http://{host}:{port}"
        # Poll readiness endpoint up to 60s
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url + "/readyz", timeout=2) as r:
                    if 200 <= r.status < 300:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            raise RuntimeError("qdrant container failed readiness check within timeout")
        yield url
    finally:
        try:
            container.stop()
        except Exception:
            pass


@pytest.mark.integration
def test_hybrid_cli_runs_basic(tmp_path, qdrant_container):
    env = os.environ.copy()
    env["QDRANT_URL"] = qdrant_container
    collection_name = f"test-{uuid.uuid4().hex[:8]}"
    env["COLLECTION_NAME"] = collection_name

    # Warm the FastEmbed model cache to avoid long first-run download inside subprocess
    try:
        from fastembed import TextEmbedding

        TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    except Exception:
        pass

    # Create a tiny repo and index it so the CLI has something to return
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "def test():\n    return 1\n", encoding="utf-8"
    )
    ing = importlib.import_module("scripts.ingest_code")
    prev_collection = os.environ.get("COLLECTION_NAME")
    os.environ["COLLECTION_NAME"] = collection_name
    try:
        ing.index_repo(
            root=tmp_path,
            qdrant_url=qdrant_container,
            api_key="",
            collection=collection_name,
            model_name="BAAI/bge-base-en-v1.5",
            recreate=True,
        )
    finally:
        if prev_collection is None:
            os.environ.pop("COLLECTION_NAME", None)
        else:
            os.environ["COLLECTION_NAME"] = prev_collection

    # Use the real model; allow time to download on first run
    env["EMBEDDING_MODEL"] = "BAAI/bge-base-en-v1.5"
    cmd = [
        sys.executable,
        "scripts/hybrid_search.py",
        "--query",
        "test",
        "--limit",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    # Assert successful run with real model and output
    assert proc.returncode == 0
    assert (proc.stdout or "").strip() != ""
