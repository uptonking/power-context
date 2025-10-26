# Remote Upload Client for Context-Engine

This document describes the remote upload client functionality that extends the existing watch_index.py for remote delta uploads.

## Overview

The remote upload client enables real-time code synchronization by uploading delta bundles to a remote server instead of processing files locally. This is useful for distributed development environments where multiple instances need to stay synchronized.

## Architecture

The system consists of:

1. **RemoteUploadClient** - Handles delta bundle creation and HTTP uploads
2. **Extended ChangeQueue** - Integrates with remote client for delta processing
3. **Enhanced watch_index.py** - Supports both local and remote modes
4. **Delta Bundle Format** - Standardized tarball format with metadata

## Configuration

The remote upload client is configured via environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `REMOTE_UPLOAD_ENABLED` | Enable remote mode | `false` | `1` |
| `REMOTE_UPLOAD_ENDPOINT` | Upload server URL | `http://localhost:8080` | `https://api.example.com` |
| `REMOTE_UPLOAD_MAX_RETRIES` | Max upload retries | `3` | `5` |
| `REMOTE_UPLOAD_TIMEOUT` | Request timeout (seconds) | `30` | `60` |

## Usage

### Local Mode (Default)
```bash
make watch
```

### Remote Mode
```bash
# Set environment variables
export REMOTE_UPLOAD_ENABLED=1
export REMOTE_UPLOAD_ENDPOINT=https://your-server.com:8080
export REMOTE_UPLOAD_MAX_RETRIES=5
export REMOTE_UPLOAD_TIMEOUT=60

# Or use the convenience target
make watch-remote REMOTE_UPLOAD_ENDPOINT=https://your-server.com:8080
```

## Delta Bundle Format

Delta bundles are tarballs (`.tar.gz`) containing:

```
delta-bundle.tar.gz
├── manifest.json          # Bundle metadata and file operations
├── files/                 # Directory containing file content
│   ├── created/           # New files
│   ├── updated/           # Modified files
│   └── moved/            # Moved files (at destination)
└── metadata/             # File metadata and hashes
    ├── hashes.json        # Content hashes for all files
    └── operations.json   # Detailed operation metadata
```

### Manifest Format
```json
{
  "version": "1.0",
  "bundle_id": "uuid-v4",
  "workspace_path": "/absolute/path/to/workspace",
  "collection_name": "workspace-collection",
  "created_at": "2025-01-26T01:55:00.000Z",
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

## Features

### Change Detection
- **Hash-based detection** - Uses SHA1 hashes to detect file changes
- **Move detection** - Identifies file moves by matching content hashes
- **Efficient caching** - Leverages existing workspace state cache
- **Debouncing** - Integrates with existing ChangeQueue debouncing

### Error Handling
- **Automatic retry** - Exponential backoff for network failures
- **Sequence recovery** - Handles sequence number mismatches
- **Fallback mode** - Falls back to local processing on upload failures
- **Bundle persistence** - Stores bundles locally for recovery

### Integration
- **Backward compatible** - Existing local mode unchanged
- **Same logging** - Uses existing logging patterns
- **Same filtering** - Leverages existing file exclusion logic
- **Same debouncing** - Integrates with existing ChangeQueue

## API Endpoints

### Upload Endpoint
```
POST /api/v1/delta/upload
Content-Type: multipart/form-data

Parameters:
- bundle: Delta bundle tarball
- workspace_path: Absolute workspace path
- collection_name: Override collection name
- sequence_number: Expected sequence number
- force: Force upload even if sequence mismatch
```

### Status Endpoint
```
GET /api/v1/delta/status?workspace_path=/workspace

Response:
{
  "workspace_path": "/workspace",
  "collection_name": "workspace-collection",
  "last_sequence": 41,
  "last_upload": "2025-01-26T01:50:00.000Z",
  "pending_operations": 0,
  "status": "ready",
  "server_info": {
    "version": "1.0.0",
    "max_bundle_size_mb": 100,
    "supported_formats": ["tar.gz"]
  }
}
```

## Testing

Run the basic tests to verify functionality:

```bash
python scripts/test_remote_basic.py
```

This tests:
- Remote configuration detection
- Delta bundle structure creation
- Sequence number tracking

## Implementation Notes

### File Structure
- `scripts/remote_upload_client.py` - Main remote upload client
- `scripts/watch_index.py` - Extended with remote mode support
- `Makefile` - Added `watch-remote` target

### Key Classes
- `RemoteUploadClient` - Core upload functionality
- `ChangeQueue` - Extended with remote client support
- `IndexHandler` - Updated for optional client (remote mode)

### Integration Points
- Uses existing `get_cached_file_hash()` for change detection
- Leverages existing file filtering from `IndexHandler._maybe_enqueue()`
- Integrates with existing debouncing in `ChangeQueue`
- Maintains same logging and progress reporting patterns

## Troubleshooting

### Common Issues

1. **"No module named 'qdrant_client'"**
   - Install dependencies: `pip install qdrant-client fastembed watchdog requests`

2. **"Remote mode not enabled"**
   - Set `REMOTE_UPLOAD_ENABLED=1` in environment

3. **"Upload failed"**
   - Check `REMOTE_UPLOAD_ENDPOINT` is accessible
   - Verify server supports delta upload API
   - Check network connectivity

4. **"Sequence mismatch"**
   - Server will attempt automatic recovery
   - Can force upload with `force=true` parameter

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from scripts.remote_upload_client import RemoteUploadClient
# ... your debug code
"
```

## Security Considerations

For this PoC implementation:
- No authentication is required (development mode)
- No encryption is applied to bundles
- Server endpoint validation is basic
- Production deployment should add proper authentication

## Future Enhancements

1. **Authentication** - Add API key or token-based auth
2. **Compression** - Add support for different compression algorithms
3. **Incremental uploads** - Support for large file incremental sync
4. **Conflict resolution** - Handle concurrent modifications
5. **Batch optimization** - Bundle multiple changes together