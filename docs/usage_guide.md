# Context-Engine Real-Time Code Ingestion: Usage Guide

This guide provides comprehensive instructions for using the Context-Engine real-time code ingestion system with both local and remote upload capabilities.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local vs Remote Mode](#local-vs-remote-mode)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Deployment](#deployment)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ with required dependencies
- Access to a Qdrant instance (local or remote)

### Basic Local Mode Setup

1. **Clone and setup the repository:**
```bash
git clone <repository-url>
cd Context-Engine
cp .env.example .env
```

2. **Start the services:**
```bash
make up
```

3. **Index your codebase:**
```bash
make index
```

4. **Start watching for changes:**
```bash
make watch
```

### Basic Remote Mode Setup

1. **Deploy the upload service:**
```bash
# Deploy to Kubernetes
kubectl apply -f deploy/kubernetes/upload-pvc.yaml
kubectl apply -f deploy/kubernetes/upload-service.yaml
```

2. **Start remote watching:**
```bash
make watch-remote REMOTE_UPLOAD_ENDPOINT=http://your-upload-service:8002
```

## Local vs Remote Mode

### Local Mode

**Use Case:** Single developer, local development environment

**How it works:**
- Files are processed directly on the local machine
- Changes are indexed directly into local Qdrant instance
- No network dependencies for indexing

**Pros:**
- ✅ Fast response time (no network latency)
- ✅ Works offline
- ✅ Simple setup
- ✅ No additional infrastructure needed

**Cons:**
- ❌ Limited to single machine
- ❌ No collaboration features
- ❌ Each developer maintains separate index

**Command:**
```bash
make watch
```

### Remote Mode

**Use Case:** Team collaboration, distributed development, centralized indexing

**How it works:**
- Files are packaged into delta bundles
- Bundles are uploaded to remote upload service
- Remote service processes and indexes changes
- All clients sync from the same central index

**Pros:**
- ✅ Centralized index for team collaboration
- ✅ Consistent search results across team
- ✅ Reduced local resource usage
- ✅ Better for large codebases
- ✅ Supports distributed teams

**Cons:**
- ❌ Requires network connectivity
- ❌ Additional infrastructure
- ❌ Network latency
- ❌ More complex setup

**Command:**
```bash
make watch-remote REMOTE_UPLOAD_ENDPOINT=http://your-server:8002
```

## Configuration

### Environment Variables

#### Core Configuration
```bash
# Qdrant connection
QDRANT_URL=http://qdrant:6333
COLLECTION_NAME=my-collection

# Workspace configuration
WATCH_ROOT=/work
WORKSPACE_PATH=/work

# Embedding model
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

#### Remote Upload Configuration
```bash
# Enable remote mode
REMOTE_UPLOAD_ENABLED=1

# Upload service endpoint
REMOTE_UPLOAD_ENDPOINT=http://your-server:8002

# Upload behavior
REMOTE_UPLOAD_MAX_RETRIES=3
REMOTE_UPLOAD_TIMEOUT=30

# Watch behavior
WATCH_DEBOUNCE_SECS=1.0
```

#### File Filtering
```bash
# Ignore file location
QDRANT_IGNORE_FILE=.qdrantignore
```

### Example .env Files

#### Local Development (.env.local)
```bash
# Local development configuration
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=my-dev-collection
WATCH_ROOT=/Users/developer/my-project
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
WATCH_DEBOUNCE_SECS=0.5
```

#### Team Collaboration (.env.remote)
```bash
# Remote team configuration
QDRANT_URL=http://qdrant.team.svc.cluster.local:6333
COLLECTION_NAME=team-shared-collection
WATCH_ROOT=/workspace
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Remote upload settings
REMOTE_UPLOAD_ENABLED=1
REMOTE_UPLOAD_ENDPOINT=http://upload-service.team.svc.cluster.local:8002
REMOTE_UPLOAD_MAX_RETRIES=5
REMOTE_UPLOAD_TIMEOUT=60
WATCH_DEBOUNCE_SECS=2.0
```

#### Production (.env.prod)
```bash
# Production configuration
QDRANT_URL=https://qdrant.production.com
COLLECTION_NAME=prod-codebase
WATCH_ROOT=/app/workspace
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Remote upload with high reliability
REMOTE_UPLOAD_ENABLED=1
REMOTE_UPLOAD_ENDPOINT=https://upload-api.production.com
REMOTE_UPLOAD_MAX_RETRIES=10
REMOTE_UPLOAD_TIMEOUT=120
WATCH_DEBOUNCE_SECS=3.0
```

## Usage Examples

### Basic Development Workflow

```bash
# 1. Start services
make up

# 2. Initial indexing
make reindex

# 3. Start watching (local mode)
make watch

# In another terminal, make changes to your code...
# Changes will be automatically indexed
```

### Team Collaboration Workflow

```bash
# 1. Deploy infrastructure (once)
kubectl apply -f deploy/kubernetes/

# 2. Each developer starts remote watching
make watch-remote REMOTE_UPLOAD_ENDPOINT=https://upload.team.com:8002

# 3. Developers make changes...
# All changes are synchronized across the team
```

### Hybrid Workflow (Local + Remote)

```bash
# Use local mode for fast iteration
make watch

# Switch to remote mode when ready to share
make watch-remote REMOTE_UPLOAD_ENDPOINT=https://upload.team.com:8002
```

### Advanced Indexing

```bash
# Index specific path
make index-path REPO_PATH=/path/to/repo RECREATE=1

# Index current directory with custom collection
make index-here REPO_NAME=my-project COLLECTION=my-project-collection

# Warm up search caches
make warm

# Run health checks
make health
```

## Deployment

### Local Development

1. **Using Docker Compose:**
```bash
# Start all services
make up

# View logs
make logs

# Check status
make ps
```

2. **Manual Setup:**
```bash
# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start watching
python scripts/watch_index.py
```

### Kubernetes Deployment

1. **Prerequisites:**
```bash
# Kubernetes cluster with storage support
kubectl cluster-info

# Install required manifests
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
```

2. **Deploy Core Services:**
```bash
# Deploy Qdrant
kubectl apply -f deploy/kubernetes/qdrant.yaml

# Deploy upload service with persistent storage
kubectl apply -f deploy/kubernetes/upload-pvc.yaml
kubectl apply -f deploy/kubernetes/upload-service.yaml

# Deploy indexer services
kubectl apply -f deploy/kubernetes/indexer-services.yaml
```

3. **Configure Access:**
```bash
# Check service status
kubectl get pods -n context-engine

# Get upload service endpoint
kubectl get svc upload-service -n context-engine

# Port forward for local testing
kubectl port-forward svc/upload-service 8002:8002 -n context-engine
```

### Production Considerations

1. **High Availability:**
```yaml
# Example: Multiple replicas for upload service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: upload-service
```

2. **Resource Limits:**
```yaml
# Example: Resource constraints
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

3. **Monitoring:**
```yaml
# Example: Health checks
livenessProbe:
  httpGet:
    path: /health
    port: 8002
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /health
    port: 8002
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Troubleshooting

### Common Issues

#### 1. "No module named 'qdrant_client'"
**Solution:**
```bash
pip install qdrant-client fastembed watchdog requests
```

#### 2. "Remote mode not enabled"
**Solution:**
```bash
export REMOTE_UPLOAD_ENABLED=1
# Or add to .env file
echo "REMOTE_UPLOAD_ENABLED=1" >> .env
```

#### 3. "Upload failed: Connection refused"
**Solutions:**
- Check upload service is running: `kubectl get pods`
- Verify endpoint URL: `curl http://your-endpoint:8002/health`
- Check network connectivity: `telnet your-endpoint 8002`

#### 4. "Sequence mismatch" errors
**Solutions:**
- Client will attempt automatic recovery
- Force upload if needed: Set `force=true` in upload request
- Reset sequence: Delete `.codebase/delta_bundles/last_sequence.txt`

#### 5. "Bundle too large" errors
**Solutions:**
- Increase `MAX_BUNDLE_SIZE_MB` on upload service
- Reduce number of changes before upload (adjust debounce)
- Split large changes into smaller commits

#### 6. "Indexing is slow"
**Solutions:**
- Use faster embedding model
- Increase `WATCH_DEBOUNCE_SECS` to reduce frequency
- Upgrade hardware (more CPU/RAM)
- Use remote mode to offload processing

### Debug Mode

#### Enable Debug Logging
```bash
# Set log level
export PYTHONPATH=.
export UPLOAD_SERVICE_LOG_LEVEL=debug

# Run with debug output
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from scripts.remote_upload_client import RemoteUploadClient
# Your debug code here
"
```

#### Check System Status
```bash
# Check Qdrant
curl http://localhost:6333/collections

# Check upload service
curl http://localhost:8002/health

# Check workspace state
ls -la .codebase/
cat .codebase/workspace_state.json
```

#### Monitor File Changes
```bash
# Watch file system events (Linux)
inotifywait -m -r -e modify,create,delete,move /path/to/watch

# Watch file system events (macOS)
fswatch -r /path/to/watch
```

### Performance Tuning

#### Optimize for Large Codebases
```bash
# Increase debounce to reduce processing frequency
WATCH_DEBOUNCE_SECS=5.0

# Use larger batch sizes
BATCH_SIZE=1000

# Increase timeouts
REMOTE_UPLOAD_TIMEOUT=120
QDRANT_TIMEOUT=60
```

#### Optimize for Real-time Response
```bash
# Reduce debounce for faster response
WATCH_DEBOUNCE_SECS=0.1

# Use smaller batches for faster processing
BATCH_SIZE=100

# Reduce timeouts
REMOTE_UPLOAD_TIMEOUT=30
QDRANT_TIMEOUT=20
```

## Advanced Configuration

### Custom File Filtering

Create `.qdrantignore` in your workspace root:

```
# Ignore patterns
*.log
*.tmp
node_modules/
.git/
build/
dist/
*.min.js
*.min.css

# Ignore specific directories
tests/fixtures/
docs/generated/
```

### Custom Embedding Models

```bash
# Use different model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Custom model (local path)
EMBEDDING_MODEL=/path/to/custom/model

# Model-specific settings
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32
```

### Multi-Collection Setup

```bash
# Different collections for different projects
COLLECTION_NAME=project-alpha

# Or use environment-specific collections
COLLECTION_NAME=${PROJECT_NAME}-${ENVIRONMENT}
```

### Integration with CI/CD

#### GitHub Actions Example
```yaml
name: Index Code Changes
on: [push]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Index changes
        env:
          REMOTE_UPLOAD_ENABLED: 1
          REMOTE_UPLOAD_ENDPOINT: ${{ secrets.UPLOAD_ENDPOINT }}
        run: |
          python scripts/watch_index.py --once
```

#### Jenkins Pipeline Example
```groovy
pipeline {
    agent any
    environment {
        REMOTE_UPLOAD_ENABLED = '1'
        REMOTE_UPLOAD_ENDPOINT = credentials('upload-endpoint')
    }
    stages {
        stage('Index') {
            steps {
                sh 'python scripts/watch_index.py --once'
            }
        }
    }
}
```

### Monitoring and Alerting

#### Prometheus Metrics
```yaml
# Example Prometheus configuration
scrape_configs:
  - job_name: 'context-engine'
    static_configs:
      - targets: ['upload-service:8002']
    metrics_path: '/metrics'
```

#### Grafana Dashboard
- Upload success rate
- Processing time
- Queue depth
- Error rates
- Resource usage

#### Alerting Rules
```yaml
# Example alerting rules
groups:
  - name: context-engine
    rules:
      - alert: HighErrorRate
        expr: upload_error_rate > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High upload error rate detected"
```

This comprehensive guide should help you get the most out of the Context-Engine real-time code ingestion system. For more specific issues or advanced use cases, refer to the individual component documentation or reach out to the development team.