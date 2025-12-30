# Unified Context-Engine image for Kubernetes deployment
# Supports multiple roles: memory, indexer, watcher, llamacpp
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORK_ROOTS="/work,/app"

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for all services
# Pin mcp/fastmcp versions to match requirements.txt for consistency across services
RUN pip install --no-cache-dir --upgrade \
    qdrant-client \
    fastembed \
    watchdog \
    onnxruntime \
    tokenizers \
    tree_sitter \
    tree_sitter_languages \
    'mcp==1.17.0' \
    'fastmcp==2.12.4'

# Copy scripts for all services
COPY scripts /app/scripts

# Create directories
WORKDIR /work

# Expose all necessary ports
EXPOSE 8000 8001 8002 8003 18000 18001 18002 18003

# Default to memory server
CMD ["python", "/app/scripts/mcp_memory_server.py"]