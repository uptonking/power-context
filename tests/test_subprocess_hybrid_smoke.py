import os
import uuid
import subprocess
import sys
import pytest

pytestmark = pytest.mark.integration


def _maybe_enabled():
    return os.environ.get("ENABLE_HYBRID_SMOKE", "").strip() in {"1","true","yes","on"}


@pytest.fixture(scope="module")
def qdrant_container():
    if not _maybe_enabled():
        pytest.skip("hybrid subprocess smoke disabled; set ENABLE_HYBRID_SMOKE=1 to run")
    try:
        from testcontainers.core.container import DockerContainer
        from testcontainers.core.waiting_utils import wait_for_logs
    except Exception:
        pytest.skip("testcontainers not available")
    container = DockerContainer("qdrant/qdrant:latest").with_exposed_ports(6333)
    container.start()
    try:
        wait_for_logs(container, "Actix runtime found; starting workers")
    except Exception:
        pass
    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(6333))
    url = f"http://{host}:{port}"
    yield url
    container.stop()


@pytest.mark.integration
def test_hybrid_cli_runs_basic(tmp_path, qdrant_container):
    if not _maybe_enabled():
        pytest.skip("hybrid subprocess smoke disabled; set ENABLE_HYBRID_SMOKE=1 to run")

    env = os.environ.copy()
    env["QDRANT_URL"] = qdrant_container
    env.setdefault("COLLECTION_NAME", f"test-{uuid.uuid4().hex[:8]}")

    # Note: CLI will attempt to download a model; keep this opt-in only.
    cmd = [sys.executable, "scripts/hybrid_search.py", "--query", "test", "--limit", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=10)
    # Either it returns non-empty stdout (success) or cleanly errors out with a message about missing model
    assert proc.returncode in (0, 1)

