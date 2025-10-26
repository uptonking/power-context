# Delta Upload Service

This document describes the HTTP upload service for receiving and processing delta bundles in Context-Engine.

## Overview

The delta upload service is a FastAPI-based HTTP service that:
- Receives delta bundles from remote upload clients
- Extracts and processes file operations (create, update, delete, move)
- Integrates with existing indexing pipeline via `ingest_code.py`
- Provides health checks and status monitoring
- Supports CephFS persistent storage for Kubernetes deployment

## API Endpoints

### Health Check
```
GET /health
```

Returns service health status and configuration.

### Status
```
GET /api/v1/delta/status?workspace_path=/path/to/workspace
```

Returns upload status for a specific workspace.

### Upload Delta Bundle
```
POST /api/v1/delta/upload
Content-Type: multipart/form-data
```

Parameters:
- `bundle`: Delta bundle tarball file
- `workspace_path`: Target workspace path
- `collection_name`: Override collection name (optional)
- `sequence_number`: Expected sequence number (optional)
- `force`: Force upload even if sequence mismatch (optional)

## Delta Bundle Format

Delta bundles are tar.gz archives with the following structure:

```
delta-bundle.tar.gz
├── manifest.json          # Bundle metadata
├── files/                 # File content
│   ├── created/           # New files
│   ├── updated/           # Modified files
│   └── moved/            # Moved files (at destination)
└── metadata/             # File metadata
    ├── hashes.json        # Content hashes
    └── operations.json   # Detailed operation metadata
```

### Manifest Format

```json
{
  "version": "1.0",
  "bundle_id": "uuid-v4",
  "workspace_path": "/absolute/path/to/workspace",
  "collection_name": "workspace-collection",
  "created_at": "2025-01-26T02:00:00.000Z",
  "sequence_number": 42,
  "parent_sequence": 41,
  "operations": {
    "created": 5,
    "updated": 3,
    "deleted": 2,
    "moved": 1
  },
  "total_files": 11,
  "total_size_bytes": 1048576,
  "compression": "gzip",
  "encoding": "utf-8"
}
```

## Deployment

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
python scripts/upload_service.py
```

The service will start on `http://localhost:8002` by default.

### Docker

Build the image:
```bash
docker build -f Dockerfile.upload-service -t context-engine-upload-service .
```

Run the container:
```bash
docker run -p 8002:8002 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e WORK_DIR=/work \
  -v /path/to/work:/work \
  context-engine-upload-service
```

### Kubernetes

1. Apply the namespace and config:
```bash
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml
```

2. Create persistent volumes (adjust storage class as needed):
```bash
kubectl apply -f deploy/kubernetes/upload-pvc.yaml
```

3. Deploy the service:
```bash
kubectl apply -f deploy/kubernetes/upload-service.yaml
```

The service will be available at:
- Internal: `http://upload-service.context-engine.svc.cluster.local:8002`
- External: `http://<node-ip>:30804` (NodePort)

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|----------|-------------|
| `UPLOAD_SERVICE_HOST` | `0.0.0.0` | Service bind address |
| `UPLOAD_SERVICE_PORT` | `8002` | Service port |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant server URL |
| `COLLECTION_NAME` | `my-collection` | Default collection name |
| `WORK_DIR` | `/work` | Workspace directory |
| `MAX_BUNDLE_SIZE_MB` | `100` | Maximum bundle size |
| `UPLOAD_TIMEOUT_SECS` | `300` | Upload timeout |

## Integration

### With Remote Upload Client

The upload service integrates with the remote upload client in `scripts/remote_upload_client.py`:

```python
from scripts.remote_upload_client import RemoteUploadClient

client = RemoteUploadClient(
    upload_endpoint="http://upload-service:8002",
    workspace_path="/path/to/workspace",
    collection_name="my-collection"
)

# Upload changes
success = client.process_and_upload_changes(changed_files)
```

### With Existing Indexing Pipeline

The service reuses the existing indexing pipeline:

- Calls `ingest_code.index_repo()` for changed files
- Uses `workspace_state.py` for state management
- Integrates with existing Qdrant connection patterns
- Supports hash-based caching and change detection

## Testing

Run the test suite:

```bash
python scripts/test_upload_service.py --url http://localhost:8002
```

This will test:
- Health check endpoint
- Status endpoint
- Upload endpoint with sample delta bundle

## Monitoring

### Health Checks

The service provides liveness and readiness probes:
- Liveness: `/health` every 10 seconds after 30s delay
- Readiness: `/health` every 5 seconds after 10s delay

### Logging

Logs include:
- Request/response details
- Bundle processing status
- Error details and stack traces
- Integration with existing logging patterns

### Metrics

The service tracks:
- Upload success/failure rates
- Processing times
- Operation counts (create, update, delete, move)
- Indexed points count

## Security Considerations

For production deployment:

1. **Authentication**: Add API key or JWT authentication
2. **Authorization**: Implement workspace-based access control
3. **Input Validation**: Enhanced bundle validation and sanitization
4. **Rate Limiting**: Add request rate limiting
5. **TLS**: Enable HTTPS for production

## Troubleshooting

### Common Issues

1. **Bundle Too Large**: Increase `MAX_BUNDLE_SIZE_MB` or optimize bundles
2. **Sequence Mismatch**: Check client sequence tracking or use `force=true`
3. **Indexing Failures**: Verify Qdrant connectivity and collection exists
4. **Storage Issues**: Check PVC status and CephFS connectivity

### Debug Mode

Enable debug logging:
```bash
export UPLOAD_SERVICE_LOG_LEVEL=debug
python scripts/upload_service.py
```

### Health Check

Verify service status:
```bash
curl http://localhost:8002/health
```

## Architecture

The upload service follows the delta upload architecture defined in:
- `delta_upload_design.md` - Format specification
- `delta_upload_architecture.md` - System design

Key components:
- **FastAPI HTTP Server**: Handles incoming requests
- **Bundle Processor**: Extracts and validates delta bundles
- **File Operations**: Applies create/update/delete/move operations
- **Indexing Integration**: Calls existing indexing pipeline
- **State Management**: Tracks sequences and workspace state