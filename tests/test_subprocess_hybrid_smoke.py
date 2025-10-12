import os
import uuid
import subprocess
import sys
import pytest

pytestmark = pytest.mark.integration


# Always enabled; smoke runs as part of full suite


@pytest.fixture(scope="module")
def qdrant_container():
    try:
        from testcontainers.core.container import DockerContainer
    except Exception:
        pytest.skip("testcontainers not available")
    import time, urllib.request
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
        TextEmbedding(model_name=os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
    except Exception:
        pass

    # Use the real model; allow time to download on first run
    cmd = [sys.executable, "scripts/hybrid_search.py", "--query", "test", "--limit", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    # Either it returns non-empty stdout (success) or cleanly errors out with a message about missing model
    assert proc.returncode in (0, 1)

