#!/usr/bin/env bash
set -euo pipefail
until curl -fsS "${QDRANT_URL:-http://localhost:6333}/" >/dev/null; do
  echo "Waiting for Qdrant at ${QDRANT_URL:-http://localhost:6333} ..."
  sleep 1
done
echo "Qdrant is up."

