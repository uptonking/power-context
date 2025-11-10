# Context-Engine Kubernetes Deployment

This directory contains Kubernetes manifests for deploying Context-Engine on a Kubernetes cluster. The deployment maintains local-first defaults while providing optional remote hosting capabilities.

## Architecture Overview

### Services Deployed

| Service | Port(s) | Description | Protocol |
|---------|---------|-------------|----------|
| **qdrant** | 6333, 6334 | Vector database | HTTP/gRPC |
| **mcp-memory** | 8000, 18000 | Memory server (SSE) | SSE |
| **mcp-memory-http** | 8002, 18002 | Memory server (HTTP) | HTTP |
| **mcp-indexer** | 8001, 18001 | Indexer server (SSE) | SSE |
| **mcp-indexer-http** | 8003, 18003 | Indexer server (HTTP) | HTTP |
| **watcher** | - | File change monitoring | - |
| **llamacpp** (optional) | 8080 | Text generation | HTTP |

### NodePort Mappings

For local development or direct access, services are exposed via NodePort:

| Service | NodePort | Local Access |
|---------|----------|--------------|
| qdrant | 30333, 30334 | `http://<node-ip>:30333` |
| mcp-memory | 30800, 30801 | `http://<node-ip>:30800` |
| mcp-indexer | 30802, 30803 | `http://<node-ip>:30802` |
| mcp-memory-http | 30804, 30805 | `http://<node-ip>:30804` |
| mcp-indexer-http | 30806, 30807 | `http://<node-ip>:30806` |
| llamacpp | 30808 | `http://<node-ip>:30808` |

## Prerequisites

1. **Kubernetes Cluster** (v1.20+)
2. **kubectl** configured to access your cluster
3. **Docker images** built and pushed to registry:
   ```bash
   # Build all service images
   ./build-images.sh --push

   # Or build individually
   docker build -f Dockerfile.mcp -t context-engine-memory:latest .
   docker build -f Dockerfile.mcp-indexer -t context-engine-indexer:latest .
   docker build -f Dockerfile.indexer -t context-engine-indexer-service:latest .
   ```

4. **Source Code Access**: Source code should be pre-distributed to all cluster nodes at `/tmp/context-engine-work`

## Quick Start

### Manual Deployment

### 1. Deploy Core Services

```bash
# Deploy namespace and configuration
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml

# Deploy Qdrant database
kubectl apply -f qdrant.yaml

# Wait for Qdrant to be ready
kubectl wait --for=condition=ready pod -l component=qdrant -n context-engine --timeout=300s

# Initialize indexes
kubectl apply -f indexer-services.yaml
```

### 2. Deploy MCP Servers

```bash
# Deploy MCP Memory and Indexer servers (SSE)
kubectl apply -f mcp-memory.yaml
kubectl apply -f mcp-indexer.yaml

# Deploy HTTP versions (optional)
kubectl apply -f mcp-http.yaml
```

### 3. Deploy Optional Services

```bash
# Deploy Llama.cpp (optional, for text generation)
kubectl apply -f llamacpp.yaml

# Deploy Ingress (optional, for domain-based access)
kubectl apply -f ingress.yaml
```

### 4. Verify Deployment

```bash
# Check all pods
kubectl get pods -n context-engine

# Check services
kubectl get services -n context-engine

# Check logs for any service
kubectl logs -f deployment/mcp-memory -n context-engine
```

## Configuration

### Environment Variables

All configuration is managed through the `context-engine-config` ConfigMap in `configmap.yaml`. Key variables include:

- **QDRANT_URL**: Database connection (automatically set to Kubernetes service)
- **COLLECTION_NAME**: Default collection name (`my-collection`)
- **EMBEDDING_MODEL**: Embedding model (`BAAI/bge-base-en-v1.5`)
- **EMBEDDING_PROVIDER**: Provider (`fastembed`)

### Persistent Storage

- **Qdrant data**: 20Gi persistent volume claim
- **Work directory**: HostPath mounted to `/tmp/context-engine-work`
- **Models directory**: HostPath mounted to `/tmp/context-engine-models`

### Customization

1. **Storage Class**: Modify `qdrant.yaml` to use your cluster's storage class
2. **Resources**: Adjust memory/CPU limits in each deployment
3. **Host Paths**: Update volume mounts to match your environment
4. **Ingress**: Configure `ingress.yaml` with your domain and SSL

## Source Code Management

The deployment uses hostPath volumes to access source code on cluster nodes. Source code must be pre-distributed to all cluster nodes at the configured paths.

## Development Workflow

### Local Development

For local development, you can continue using `docker-compose.yml` as before:

```bash
# Local development (unchanged)
docker-compose up -d
```

### Kubernetes Development

For Kubernetes-based development:

1. **Build and Push Image**:
   ```bash
   docker build -t your-registry/context-engine:latest .
   docker push your-registry/context-engine:latest
   ```

2. **Update Image References**:
   ```bash
   # Update all manifests to use your image
   sed -i 's|context-engine:latest|your-registry/context-engine:latest|g' *.yaml
   ```

3. **Deploy Changes**:
   ```bash
   kubectl apply -f .
   ```

### File Synchronization

The Kubernetes deployment uses HostPath volumes to sync files:

```bash
# Mount your local code directory
sudo mkdir -p /tmp/context-engine-work
sudo cp -r /path/to/your/code/* /tmp/context-engine-work/

# Mount models (if using Llama.cpp)
sudo mkdir -p /tmp/context-engine-models
sudo cp /path/to/your/models/* /tmp/context-engine-models/
```

## Monitoring and Troubleshooting

### Health Checks

All services include liveness and readiness probes:

- **HTTP Services**: `/health` endpoint
- **Qdrant**: `/ready` and `/health` endpoints

### Logs

```bash
# View all logs
kubectl logs -f deployment/mcp-memory -n context-engine

# View watcher logs for indexing activity
kubectl logs -f deployment/watcher -n context-engine

# View Qdrant logs
kubectl logs -f statefulset/qdrant -n context-engine
```

### Common Issues

1. **Storage Class Not Found**:
   ```bash
   kubectl get storageclass
   # Update qdrant.yaml to use available storage class
   ```

2. **HostPath Permissions**:
   ```bash
   # Ensure host directories are accessible
   sudo chmod -R 755 /tmp/context-engine-work
   ```

3. **Image Pull Errors**:
   ```bash
   # Check image registry access
   kubectl describe pod <pod-name> -n context-engine
   ```

## Scaling and High Availability

### Scaling MCP Servers

```bash
# Scale memory servers
kubectl scale deployment mcp-memory --replicas=3 -n context-engine

# Scale indexer servers
kubectl scale deployment mcp-indexer --replicas=2 -n context-engine
```

### High Availability Qdrant

For production, consider:

1. **Qdrant Cloud**: Managed service with automatic scaling
2. **Multi-replica StatefulSet**: Configure Qdrant clustering
3. **External Database**: Use managed vector database

## Security Considerations

1. **Network Policies**: Restrict inter-service communication
2. **RBAC**: Implement proper role-based access control
3. **Secrets Management**: Use Kubernetes Secrets for sensitive data
4. **TLS**: Configure Ingress with SSL/TLS certificates

## Migration from Docker Compose

### Data Migration

1. **Export Qdrant Data**:
   ```bash
   docker exec qdrant-db python -c "
   import requests
   response = requests.get('http://localhost:6333/collections/my-collection')
   print(response.json())
   "
   ```

2. **Import to Kubernetes**:
   ```bash
   # Copy data to Kubernetes PVC
   kubectl cp qdrant-backup.json qdrant-0:/qdrant/storage/ -n context-engine
   ```

### Configuration Migration

The Kubernetes deployment maintains the same environment variables as Docker Compose. Most settings should work without changes.

## Production Deployment Checklist

- [ ] Use LoadBalancer services instead of NodePort
- [ ] Configure proper Ingress with SSL certificates
- [ ] Set up monitoring and logging (Prometheus, Grafana)
- [ ] Implement backup strategy for Qdrant data
- [ ] Configure resource limits and requests appropriately
- [ ] Set up horizontal pod autoscaling
- [ ] Implement security policies and network segmentation
- [ ] Configure proper secrets management
- [ ] Set up CI/CD pipeline for automated deployments

## Support

For issues with Kubernetes deployment:

1. Check the service logs: `kubectl logs -f <deployment> -n context-engine`
2. Verify resource usage: `kubectl top pods -n context-engine`
3. Check events: `kubectl get events -n context-engine --sort-by=.metadata.creationTimestamp`

For application issues, refer to the main project documentation.