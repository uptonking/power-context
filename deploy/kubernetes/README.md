# Kubernetes Deployment Guide

## Overview

This directory contains Kubernetes manifests for deploying Context Engine on a remote cluster using **Kustomize**. This enables:

- **Remote development** from thin clients with cluster-based heavy lifting
- **Multi-repository indexing** with unified `codebase` collection
- **Scalable architecture** with independent watcher deployments per repo
- **Kustomize-based configuration** for easy customization and overlays

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Qdrant     │  │  Memory MCP  │  │ Indexer MCP  │     │
│  │ StatefulSet  │  │  Deployment  │  │  Deployment  │     │
│  │  Port: 6333  │  │  Port: 8000  │  │  Port: 8001  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         │    ┌─────────────┴──────────────────┘             │
│         │    │                                              │
│  ┌──────▼────▼──────────────────────────────────────────┐  │
│  │           PersistentVolume (qdrant-storage)          │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Watcher    │  │   Watcher    │  │   Watcher    │     │
│  │  (repo-1)    │  │  (repo-2)    │  │  (repo-3)    │     │
│  │  Deployment  │  │  Deployment  │  │  Deployment  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│  ┌──────▼──────────────────▼──────────────────▼─────────┐  │
│  │           HostPath Volume (repos)                     │  │
│  │  /tmp/context-engine-repos/repo-1/                    │  │
│  │  /tmp/context-engine-repos/repo-2/                    │  │
│  │  /tmp/context-engine-repos/repo-3/                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Kubernetes cluster (1.19+)
- `kubectl` configured to access your cluster
- `kustomize` (optional, kubectl has built-in support)
- Docker image built and pushed to a registry

### 1. Build and Push Image

```bash
# Build unified image
docker build -t your-registry/context-engine:latest .

# Push to registry
docker push your-registry/context-engine:latest
```

### 2. Update Image References

Edit `kustomization.yaml` to use your registry:

```yaml
images:
  - name: context-engine
    newName: your-registry/context-engine
    newTag: latest
```

### 3. Deploy Using Kustomize

```bash
# Option 1: Using kubectl with kustomize
kubectl apply -k .

# Option 2: Using kustomize CLI
kustomize build . | kubectl apply -f -

# Option 3: Using the deploy script
./deploy.sh --registry your-registry --tag latest
```

### 4. Deploy Using Makefile

```bash
# Deploy all services
make deploy

# Or deploy core services only
make deploy-core

# Check status
make status
```

### 5. Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n context-engine

# Check services
kubectl get svc -n context-engine

# View logs
make logs-service SERVICE=mcp-memory
```

### 6. Access Services

```bash
# Port forward to localhost
make port-forward

# Or access via NodePort
# Qdrant: http://<node-ip>:30333
# MCP Memory: http://<node-ip>:30800
# MCP Indexer: http://<node-ip>:30802
```

## Configuration

### Automatic Model Download

The Llama.cpp deployment includes an **init container** that automatically downloads the model on first startup:

- **Default Model**: Qwen2.5-1.5B-Instruct (Q8_0 quantization, ~1.7GB)
- **Download Location**: `/tmp/context-engine-models/` on the Kubernetes node
- **Behavior**: Downloads only if model doesn't exist (idempotent)
- **Good balance**: Fast, accurate, small footprint

To use a different model, edit `configmap.yaml`:

```yaml
# Model download configuration
LLAMACPP_MODEL_URL: "https://huggingface.co/your-org/your-model/resolve/main/model.gguf"
LLAMACPP_MODEL_NAME: "model.gguf"
```

**Alternative Models**:
- **Qwen2.5-0.5B-Instruct-Q8** (~500MB) - Tiny, very fast
- **Qwen2.5-1.5B-Instruct-Q8** (default, ~1.7GB) - Best balance
- **Granite-3.0-3B-Instruct-Q8** (~3.2GB) - Higher quality
- **Phi-3-mini-4k-instruct-Q8** (~4GB) - High quality

### Environment Variables (ConfigMap)

Key environment variables in `configmap.yaml`:

```yaml
COLLECTION_NAME: "codebase"           # Unified collection for all repos
EMBEDDING_MODEL: "BAAI/bge-base-en-v1.5"
QDRANT_URL: "http://qdrant:6333"
INDEX_MICRO_CHUNKS: "1"
MAX_MICRO_CHUNKS_PER_FILE: "200"
WATCH_DEBOUNCE_SECS: "1.5"
```

### Persistent Volumes

Two persistent volumes are required:

1. **qdrant-storage**: Stores Qdrant vector database
   - Size: 50Gi (adjust based on codebase size)
   - Access: ReadWriteOnce

2. **repos-storage**: Stores repository code
   - Size: 100Gi (adjust based on number/size of repos)
   - Access: ReadWriteMany (required for multiple watchers)

### Resource Requests/Limits

Adjust based on your cluster capacity:

```yaml
# Qdrant (memory-intensive)
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"

# MCP Servers (moderate)
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "4Gi"
    cpu: "2"

# Watchers (light)
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1"
```

## Multi-Repository Setup

### Adding a New Repository

1. **Upload repository code** to the repos volume:
   ```bash
   # Using uploader service
   python scripts/upload_repo.py --repo-name my-service --path /local/path/to/repo
   
   # Or using kubectl cp
   kubectl cp /local/path/to/repo context-engine/uploader-pod:/repos/my-service
   ```

2. **Create watcher deployment** for the new repo:
   ```bash
   # Copy and modify watcher template
   cp watcher-backend-deployment.yaml watcher-my-service-deployment.yaml
   
   # Edit: change WATCH_ROOT, REPO_NAME, and volume subPath
   # Then apply
   kubectl apply -f watcher-my-service-deployment.yaml -n context-engine
   ```

3. **Verify indexing**:
   ```bash
   # Check watcher logs
   kubectl logs -f deployment/watcher-my-service -n context-engine
   
   # Check collection status via MCP
   curl http://indexer-mcp-service:8001/sse
   ```

### Repository Volume Structure

```
/repos/
├── backend/
│   ├── .codebase/
│   │   └── state.json
│   ├── src/
│   ├── tests/
│   └── ...
├── frontend/
│   ├── .codebase/
│   │   └── state.json
│   ├── src/
│   └── ...
└── ml-service/
    ├── .codebase/
    │   └── state.json
    ├── models/
    └── ...
```

## Accessing Services

### From Within Cluster

Services are accessible via Kubernetes DNS:

- Qdrant: `http://qdrant:6333`
- Memory MCP: `http://mcp-memory:8000/sse`
- Indexer MCP: `http://mcp-indexer:8001/sse`

### From Outside Cluster

#### Option 1: Port Forwarding (Development)

```bash
# Forward MCP services to localhost
kubectl port-forward -n context-engine svc/mcp-memory 8000:8000
kubectl port-forward -n context-engine svc/mcp-indexer 8001:8001
kubectl port-forward -n context-engine svc/qdrant 6333:6333
```

Then configure your IDE to use `http://localhost:8000/sse` and `http://localhost:8001/sse`.

#### Option 2: Ingress (Production)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: context-engine-ingress
  namespace: context-engine
spec:
  rules:
  - host: mcp.your-domain.com
    http:
      paths:
      - path: /memory
        pathType: Prefix
        backend:
          service:
            name: mcp-memory
            port:
              number: 8000
      - path: /indexer
        pathType: Prefix
        backend:
          service:
            name: mcp-indexer
            port:
              number: 8001
```

#### Option 3: LoadBalancer (Cloud)

Change service type to `LoadBalancer` in service manifests:

```yaml
spec:
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 8000
```

## Monitoring and Maintenance

### Health Checks

```bash
# Check Qdrant health
kubectl exec -n context-engine qdrant-0 -- curl -f http://localhost:6333/readyz

# Check MCP server health
kubectl exec -n context-engine deployment/mcp-memory -- curl -f http://localhost:18000/health
kubectl exec -n context-engine deployment/mcp-indexer -- curl -f http://localhost:18001/health
```

### Logs

```bash
# View logs for specific service
kubectl logs -f -n context-engine deployment/mcp-memory
kubectl logs -f -n context-engine deployment/mcp-indexer
kubectl logs -f -n context-engine deployment/watcher-backend

# View logs for all watchers
kubectl logs -f -n context-engine -l app=watcher
```

### Collection Status

```bash
# Port forward indexer MCP
kubectl port-forward -n context-engine svc/mcp-indexer 8001:8001

# Check collection status
curl -X POST http://localhost:8001/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"qdrant_status","arguments":{}}}'
```

### Backup and Restore

#### Backup Qdrant Data

```bash
# Create snapshot
kubectl exec -n context-engine qdrant-0 -- \
  curl -X POST http://localhost:6333/collections/codebase/snapshots

# Copy snapshot to local
kubectl cp context-engine/qdrant-0:/qdrant/storage/snapshots/codebase-snapshot.tar \
  ./backup/codebase-snapshot.tar
```

#### Restore Qdrant Data

```bash
# Copy snapshot to pod
kubectl cp ./backup/codebase-snapshot.tar \
  context-engine/qdrant-0:/qdrant/storage/snapshots/

# Restore snapshot
kubectl exec -n context-engine qdrant-0 -- \
  curl -X PUT http://localhost:6333/collections/codebase/snapshots/upload \
  -F 'snapshot=@/qdrant/storage/snapshots/codebase-snapshot.tar'
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod -n context-engine <pod-name>

# Check events
kubectl get events -n context-engine --sort-by='.lastTimestamp'
```

### Persistent Volume Issues

```bash
# Check PV/PVC status
kubectl get pv,pvc -n context-engine

# Check PVC events
kubectl describe pvc -n context-engine <pvc-name>
```

### Watcher Not Indexing

```bash
# Check watcher logs
kubectl logs -f -n context-engine deployment/watcher-backend

# Verify volume mount
kubectl exec -n context-engine deployment/watcher-backend -- ls -la /repos/backend

# Check Qdrant connectivity
kubectl exec -n context-engine deployment/watcher-backend -- \
  curl -f http://qdrant:6333/readyz
```

### MCP Connection Issues

```bash
# Test SSE endpoint
kubectl exec -n context-engine deployment/mcp-indexer -- \
  curl -H "Accept: text/event-stream" http://localhost:8001/sse

# Check service endpoints
kubectl get endpoints -n context-engine
```

## Scaling

### Horizontal Scaling

- **MCP Servers**: Can run multiple replicas behind a service
- **Watchers**: One per repository (do not scale horizontally)
- **Qdrant**: Single instance (StatefulSet with replicas=1)

### Vertical Scaling

Adjust resource requests/limits based on workload:

```bash
# Edit deployment
kubectl edit deployment -n context-engine mcp-indexer

# Or patch
kubectl patch deployment -n context-engine mcp-indexer -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"mcp-indexer","resources":{"requests":{"memory":"4Gi"}}}]}}}}'
```

## Security Considerations

1. **Network Policies**: Restrict pod-to-pod communication
2. **RBAC**: Limit service account permissions
3. **Secrets Management**: Use Kubernetes secrets or external secret managers
4. **TLS**: Enable TLS for external access via Ingress
5. **Resource Quotas**: Set namespace resource quotas

## See Also

- [Multi-Repository Collections Guide](../../docs/MULTI_REPO_COLLECTIONS.md)
- [MCP API Reference](../../docs/MCP_API.md)
- [Architecture Overview](../../docs/ARCHITECTURE.md)

