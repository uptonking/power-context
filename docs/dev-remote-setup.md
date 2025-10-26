# Development Remote Upload System Setup

This guide covers setting up and using the development environment for testing the Context-Engine remote upload system with shared volumes that simulate the Kubernetes CephFS RWX PVC behavior.

## Overview

The `docker-compose.dev-remote.yml` file provides a complete local development environment that simulates the Kubernetes deployment with:

- **Shared Volumes**: Simulates CephFS ReadWriteMany (RWX) PVC behavior
- **Upload Service**: HTTP service for receiving delta bundles
- **All Existing Services**: Qdrant, MCP servers, indexer, watcher, etc.
- **Service Discovery**: Proper networking between all services
- **Development Tools**: Easy testing and debugging capabilities

## Quick Start

### 1. Initial Setup

```bash
# Run the development setup script
./scripts/dev-setup.sh

# Or manually:
mkdir -p dev-workspace/.codebase
cp .env.example .env  # if not exists
```

### 2. Start the System

```bash
# Bootstrap the complete system (recommended)
make dev-remote-bootstrap

# Or start services step by step:
make dev-remote-up
```

### 3. Test Your Repository

```bash
# 1. Copy your repository to the workspace
cp -r /path/to/your/repo dev-workspace/your-repo

# 2. Test the upload service
make dev-remote-test

# 3. Check service health
curl http://localhost:8004/health
```

## Architecture

### Shared Volume Structure

The development environment uses shared volumes to simulate Kubernetes CephFS behavior:

```
dev-workspace/                    # Main workspace (simulates CephFS RWX)
├── your-repo/                   # Your repository code
├── .codebase/                   # Indexing metadata and cache
└── ...                          # Other repositories
```

### Service Configuration

| Service | Port | Purpose | Volumes |
|---------|------|---------|---------|
| upload_service | 8004 | Delta upload HTTP API | shared_workspace, shared_codebase |
| qdrant | 6333/6334 | Vector database | qdrant_storage_dev_remote |
| mcp | 8000 | MCP search server (SSE) | shared_workspace (ro) |
| mcp_indexer | 8001 | MCP indexer server (SSE) | shared_workspace, shared_codebase |
| mcp_http | 8002 | MCP search server (HTTP) | shared_workspace (ro) |
| mcp_indexer_http | 8003 | MCP indexer server (HTTP) | shared_workspace, shared_codebase |
| llamacpp | 8080 | LLM decoder service | ./models (ro) |

### Network Configuration

All services communicate via the `dev-remote-network` bridge network (172.20.0.0/16), ensuring proper service discovery and isolation.

## Available Commands

### Development Environment Commands

```bash
# Environment setup
make dev-remote-up          # Start all services
make dev-remote-down        # Stop all services
make dev-remote-restart     # Restart with rebuild
make dev-remote-logs        # Follow service logs
make dev-remote-clean       # Clean up volumes and containers

# Bootstrap and testing
make dev-remote-bootstrap   # Complete system setup
make dev-remote-test        # Test upload workflow
make dev-remote-client      # Start remote upload client

# Individual service management
docker compose -f docker-compose.dev-remote.yml ps
docker compose -f docker-compose.dev-remote.yml logs upload_service
```

### Remote Upload Testing

```bash
# Test upload service health
curl http://localhost:8004/health

# Check workspace status
curl 'http://localhost:8004/api/v1/delta/status?workspace_path=/work/your-repo'

# Test file upload (requires delta bundle)
curl -X POST \
  -F 'bundle=@test-bundle.tar.gz' \
  -F 'workspace_path=/work/your-repo' \
  http://localhost:8004/api/v1/delta/upload
```

## Workflow Examples

### 1. Local Development Workflow

```bash
# 1. Setup environment
./scripts/dev-setup.sh

# 2. Add your repository
cp -r ~/my-project dev-workspace/my-project

# 3. Start the system
make dev-remote-bootstrap

# 4. Test indexing
docker compose -f docker-compose.dev-remote.yml run --rm indexer --root /work/my-project

# 5. Start watcher for live updates
docker compose -f docker-compose.dev-remote.yml run --rm watcher
```

### 2. Remote Upload Testing Workflow

```bash
# 1. Start upload service
make dev-remote-up

# 2. Test remote upload from another directory
cd ~/my-project
make watch-remote REMOTE_UPLOAD_ENDPOINT=http://localhost:8004

# 3. Make changes to your code
# Files will be automatically uploaded and indexed
```

### 3. Multiple Repository Testing

```bash
# 1. Setup multiple repositories
mkdir -p dev-workspace/{repo1,repo2,repo3}
cp -r ~/project1/* dev-workspace/repo1/
cp -r ~/project2/* dev-workspace/repo2/
cp -r ~/project3/* dev-workspace/repo3/

# 2. Start system
make dev-remote-bootstrap

# 3. Index each repository
docker compose -f docker-compose.dev-remote.yml run --rm indexer --root /work/repo1 --collection repo1
docker compose -f docker-compose.dev-remote.yml run --rm indexer --root /work/repo2 --collection repo2
docker compose -f docker-compose.dev-remote.yml run --rm indexer --root /work/repo3 --collection repo3
```

## Environment Variables

### Development-Specific Variables

```bash
# Workspace configuration
HOST_INDEX_PATH=./dev-workspace          # Local workspace path
DEV_REMOTE_MODE=1                        # Enable dev-remote mode
DEV_REMOTE_DEBUG=1                       # Enable debug logging

# Upload service configuration
UPLOAD_SERVICE_HOST=0.0.0.0             # Service bind address
UPLOAD_SERVICE_PORT=8002                  # Service port (internal)
UPLOAD_SERVICE_DEBUG=1                    # Enable debug mode

# Remote upload client configuration
REMOTE_UPLOAD_ENABLED=1                   # Enable remote upload
REMOTE_UPLOAD_ENDPOINT=http://upload_service:8002  # Upload endpoint
REMOTE_UPLOAD_MAX_RETRIES=3               # Max retry attempts
REMOTE_UPLOAD_TIMEOUT=30                  # Request timeout (seconds)
REMOTE_UPLOAD_DEBUG=1                     # Enable debug logging
```

### Standard Variables (from .env.example)

All standard Context-Engine variables are supported and can be overridden for development:

```bash
QDRANT_URL=http://qdrant:6333
COLLECTION_NAME=my-collection
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_PROVIDER=fastembed
# ... other standard variables
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check what's using ports
   netstat -tulpn | grep :8004
   # Stop conflicting services
   make dev-remote-down
   ```

2. **Volume Permission Issues**
   ```bash
   # Fix workspace permissions
   sudo chown -R $USER:$USER dev-workspace
   chmod -R 755 dev-workspace
   ```

3. **Service Not Ready**
   ```bash
   # Check service status
   make dev-remote-logs
   docker compose -f docker-compose.dev-remote.yml ps
   
   # Restart specific service
   docker compose -f docker-compose.dev-remote.yml restart upload_service
   ```

4. **Upload Failures**
   ```bash
   # Check upload service logs
   docker compose -f docker-compose.dev-remote.yml logs upload_service
   
   # Test upload service directly
   curl -v http://localhost:8004/health
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Add to .env
DEV_REMOTE_DEBUG=1
UPLOAD_SERVICE_DEBUG=1
REMOTE_UPLOAD_DEBUG=1

# Restart services
make dev-remote-restart
```

### Clean Reset

For a complete reset:

```bash
# Clean everything
make dev-remote-clean

# Remove workspace
rm -rf dev-workspace

# Start fresh
./scripts/dev-setup.sh
make dev-remote-bootstrap
```

## Integration with Existing Workflows

### Using with Existing Make Targets

The dev-remote environment integrates with existing Make targets:

```bash
# Use dev-remote environment with existing targets
HOST_INDEX_PATH=./dev-workspace docker compose -f docker-compose.dev-remote.yml run --rm indexer --root /work/my-repo

# Test with dev-remote stack
make health  # Uses dev-remote stack if running
make hybrid   # Uses dev-remote Qdrant instance
```

### MCP Client Configuration

Configure your MCP clients (Cursor, Windsurf, etc.):

```json
{
  "mcpServers": {
    "qdrant": { 
      "type": "sse", 
      "url": "http://localhost:8000/sse", 
      "disabled": false 
    },
    "qdrant-indexer": { 
      "type": "sse", 
      "url": "http://localhost:8001/sse", 
      "disabled": false 
    }
  }
}
```

## Performance Considerations

### Resource Allocation

The dev-remote environment is configured for development:

- **Memory**: Moderate allocation suitable for development
- **CPU**: Shared allocation with reasonable limits
- **Storage**: Local volumes for fast I/O

### Optimization Tips

1. **Use SSD Storage**: Place `dev-workspace` on SSD for better performance
2. **Limit Repository Size**: Test with smaller repositories first
3. **Adjust Batch Sizes**: Tune `INDEX_UPSERT_BATCH` for your hardware
4. **Monitor Resources**: Use `docker stats` to monitor resource usage

## Next Steps

1. **Test Your Repository**: Add your code to `dev-workspace` and test the workflow
2. **Experiment with Remote Upload**: Try the remote upload client with your changes
3. **Integrate with IDE**: Configure your MCP client for the development environment
4. **Contribute**: Report issues and contribute improvements to the dev-remote setup

## Support

For issues with the dev-remote environment:

1. Check the troubleshooting section above
2. Review service logs: `make dev-remote-logs`
3. Check the main documentation: `docs/remote_upload.md`
4. Open an issue with details about your setup and the problem