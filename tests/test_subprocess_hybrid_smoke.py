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
    container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
    container.start()
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(6333))
    url = f"http://{host}:{port}"
    # Poll readiness endpoint up to 60s
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as r:
                if 200 <= r.status < 300:
                    break
        except Exception:
            pass
        time.sleep(1)
    yield url
    container.stop()


@pytest.mark.integration
def test_hybrid_cli_runs_basic(tmp_path, qdrant_container):
    env = os.environ.copy()
    env["QDRANT_URL"] = qdrant_container
    env.setdefault("COLLECTION_NAME", f"test-{uuid.uuid4().hex[:8]}")

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
    ing.index_repo(
        root=tmp_path,
        qdrant_url=qdrant_container,
        api_key="",
        collection=env["COLLECTION_NAME"],
        model_name="BAAI/bge-base-en-v1.5",
        recreate=True,
    )

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
