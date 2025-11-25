#!/usr/bin/env python3
"""
HTTP Upload Service for Delta Bundles in Context-Engine.

This FastAPI service receives delta bundles from remote upload clients,
processes them, and integrates with the existing indexing pipeline.
"""

import os
import json
import tarfile
import tempfile
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import existing workspace state and indexing functions
try:
    from scripts.workspace_state import (
        log_activity,
        get_collection_name,
        get_cached_file_hash,
        set_cached_file_hash,
        _extract_repo_name_from_path,
        update_repo_origin,
        get_collection_mappings,
    )
except ImportError:
    # Fallback for testing without full environment
    log_activity = None
    get_collection_name = None
    get_cached_file_hash = None
    set_cached_file_hash = None
    _extract_repo_name_from_path = None
    update_repo_origin = None
    get_collection_mappings = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "my-collection")
WORK_DIR = os.environ.get("WORK_DIR", "/work")
MAX_BUNDLE_SIZE_MB = int(os.environ.get("MAX_BUNDLE_SIZE_MB", "100"))
UPLOAD_TIMEOUT_SECS = int(os.environ.get("UPLOAD_TIMEOUT_SECS", "300"))

# FastAPI app
app = FastAPI(
    title="Context-Engine Delta Upload Service",
    description="HTTP service for receiving and processing delta bundles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sequence tracking (in production, use persistent storage)
_sequence_tracker: Dict[str, int] = {}

class UploadResponse(BaseModel):
    success: bool
    bundle_id: Optional[str] = None
    sequence_number: Optional[int] = None
    processed_operations: Optional[Dict[str, int]] = None
    processing_time_ms: Optional[int] = None
    next_sequence: Optional[int] = None
    error: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    workspace_path: str
    collection_name: str
    last_sequence: int
    last_upload: Optional[str] = None
    pending_operations: int
    status: str
    server_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    qdrant_url: str
    work_dir: str

def get_workspace_key(workspace_path: str) -> str:
    """Generate 16-char hash for collision avoidance in remote uploads.

    Remote uploads may have identical folder names from different users,
    so uses longer hash than local indexing (8-chars) to ensure uniqueness.

    Both host paths (/home/user/project/repo) and container paths (/work/repo)
    should generate the same key for the same repository.
    """
    repo_name = Path(workspace_path).name
    return hashlib.sha256(repo_name.encode('utf-8')).hexdigest()[:16]

def get_next_sequence(workspace_path: str) -> int:
    """Get next sequence number for workspace."""
    key = get_workspace_key(workspace_path)
    current = _sequence_tracker.get(key, 0)
    next_seq = current + 1
    _sequence_tracker[key] = next_seq
    return next_seq

def get_last_sequence(workspace_path: str) -> int:
    """Get last sequence number for workspace."""
    key = get_workspace_key(workspace_path)
    return _sequence_tracker.get(key, 0)

def validate_bundle_format(bundle_path: Path) -> Dict[str, Any]:
    """Validate delta bundle format and return manifest."""
    try:
        with tarfile.open(bundle_path, "r:gz") as tar:
            # Check for required files
            required_files = ["manifest.json", "metadata/operations.json", "metadata/hashes.json"]
            members = tar.getnames()

            for req_file in required_files:
                if not any(req_file in member for member in members):
                    raise ValueError(f"Missing required file: {req_file}")

            # Extract and validate manifest
            manifest_member = None
            for member in members:
                if member.endswith("manifest.json"):
                    manifest_member = member
                    break

            if not manifest_member:
                raise ValueError("manifest.json not found in bundle")

            manifest_file = tar.extractfile(manifest_member)
            if not manifest_file:
                raise ValueError("Cannot extract manifest.json")

            manifest = json.loads(manifest_file.read().decode('utf-8'))

            # Validate manifest structure
            required_fields = ["version", "bundle_id", "workspace_path", "created_at", "sequence_number"]
            for field in required_fields:
                if field not in manifest:
                    raise ValueError(f"Missing required field in manifest: {field}")

            return manifest

    except Exception as e:
        raise ValueError(f"Invalid bundle format: {str(e)}")

def process_delta_bundle(workspace_path: str, bundle_path: Path, manifest: Dict[str, Any]) -> Dict[str, int]:
    """Process delta bundle and return operation counts."""
    operations_count = {
        "created": 0,
        "updated": 0,
        "deleted": 0,
        "moved": 0,
        "skipped": 0,
        "failed": 0
    }

    try:
        # CRITICAL FIX: Extract repo name and create workspace under WORK_DIR
        # Previous bug: used source workspace_path directly, extracting files outside /work
        # This caused watcher service to never see uploaded files
        if _extract_repo_name_from_path:
            repo_name = _extract_repo_name_from_path(workspace_path)
            # Fallback to directory name if repo detection fails
            if not repo_name:
                repo_name = Path(workspace_path).name
        else:
            # Fallback: use directory name
            repo_name = Path(workspace_path).name

        # Generate workspace under WORK_DIR using repo name hash
        workspace_key = get_workspace_key(workspace_path)
        workspace = Path(WORK_DIR) / f"{repo_name}-{workspace_key}"
        workspace.mkdir(parents=True, exist_ok=True)

        with tarfile.open(bundle_path, "r:gz") as tar:
            # Extract operations metadata
            ops_member = None
            for member in tar.getnames():
                if member.endswith("metadata/operations.json"):
                    ops_member = member
                    break

            if not ops_member:
                raise ValueError("operations.json not found in bundle")

            ops_file = tar.extractfile(ops_member)
            if not ops_file:
                raise ValueError("Cannot extract operations.json")

            operations_data = json.loads(ops_file.read().decode('utf-8'))
            operations = operations_data.get("operations", [])

            # Process each operation
            for operation in operations:
                op_type = operation.get("operation")
                rel_path = operation.get("path")

                if not rel_path:
                    operations_count["skipped"] += 1
                    continue

                target_path = workspace / rel_path

                try:
                    if op_type == "created":
                        # Extract file from bundle
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/created/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                operations_count["created"] += 1
                            else:
                                operations_count["failed"] += 1
                        else:
                            operations_count["failed"] += 1

                    elif op_type == "updated":
                        # Extract updated file
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/updated/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                operations_count["updated"] += 1
                            else:
                                operations_count["failed"] += 1
                        else:
                            operations_count["failed"] += 1

                    elif op_type == "moved":
                        # Extract moved file to destination
                        file_member = None
                        for member in tar.getnames():
                            if member.endswith(f"files/moved/{rel_path}"):
                                file_member = member
                                break

                        if file_member:
                            file_content = tar.extractfile(file_member)
                            if file_content:
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                target_path.write_bytes(file_content.read())
                                operations_count["moved"] += 1
                            else:
                                operations_count["failed"] += 1
                        else:
                            operations_count["failed"] += 1

                    elif op_type == "deleted":
                        # Delete file
                        if target_path.exists():
                            target_path.unlink()
                            operations_count["deleted"] += 1
                        else:
                            operations_count["skipped"] += 1

                    else:
                        operations_count["skipped"] += 1

                except Exception as e:
                    logger.error(f"Error processing operation {op_type} for {rel_path}: {e}")
                    operations_count["failed"] += 1

        return operations_count

    except Exception as e:
        logger.error(f"Error processing delta bundle: {e}")
        raise


async def _process_bundle_background(
    workspace_path: str,
    bundle_path: Path,
    manifest: Dict[str, Any],
    sequence_number: Optional[int],
    bundle_id: Optional[str],
) -> None:
    try:
        start_time = datetime.now()
        operations_count = await asyncio.to_thread(
            process_delta_bundle, workspace_path, bundle_path, manifest
        )
        if sequence_number is not None:
            key = get_workspace_key(workspace_path)
            _sequence_tracker[key] = sequence_number
        if log_activity:
            try:
                repo = _extract_repo_name_from_path(workspace_path) if _extract_repo_name_from_path else None
                log_activity(
                    repo_name=repo,
                    action="uploaded",
                    file_path=bundle_id,
                    details={
                        "bundle_id": bundle_id,
                        "operations": operations_count,
                        "source": "delta_upload",
                    },
                )
            except Exception as activity_err:
                logger.debug(f"[upload_service] Failed to log activity for bundle {bundle_id}: {activity_err}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(
            f"[upload_service] Finished processing bundle {bundle_id} seq {sequence_number} in {int(processing_time)}ms"
        )
    except Exception as e:
        logger.error(f"[upload_service] Error in background processing for bundle {bundle_id}: {e}")
    finally:
        try:
            bundle_path.unlink()
        except Exception:
            pass


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        qdrant_url=QDRANT_URL,
        work_dir=WORK_DIR
    )

@app.get("/api/v1/delta/status", response_model=StatusResponse)
async def get_status(workspace_path: str):
    """Get upload status for workspace."""
    try:
        # Get collection name
        if get_collection_name:
            repo_name = _extract_repo_name_from_path(workspace_path) if _extract_repo_name_from_path else None
            collection_name = get_collection_name(repo_name)
        else:
            collection_name = DEFAULT_COLLECTION

        # Get last sequence
        last_sequence = get_last_sequence(workspace_path)

        last_upload = None

        return StatusResponse(
            workspace_path=workspace_path,
            collection_name=collection_name,
            last_sequence=last_sequence,
            last_upload=last_upload,
            pending_operations=0,
            status="ready",
            server_info={
                "version": "1.0.0",
                "max_bundle_size_mb": MAX_BUNDLE_SIZE_MB,
                "supported_formats": ["tar.gz"]
            }
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/delta/upload", response_model=UploadResponse)
async def upload_delta_bundle(
    request: Request,
    bundle: UploadFile = File(...),
    workspace_path: str = Form(...),
    collection_name: Optional[str] = Form(None),
    sequence_number: Optional[int] = Form(None),
    force: Optional[bool] = Form(False),
    source_path: Optional[str] = Form(None),
):
    """Upload and process delta bundle."""
    start_time = datetime.now()
    client_host = request.client.host if hasattr(request, 'client') and request.client else 'unknown'

    try:
        logger.info(f"[upload_service] Begin processing upload for workspace={workspace_path} from {client_host}")
        # Validate workspace path
        workspace = Path(workspace_path)
        if not workspace.is_absolute():
            workspace = Path(WORK_DIR) / workspace

        workspace_path = str(workspace.resolve())

        # Always derive repo_name from workspace_path for origin tracking
        repo_name = _extract_repo_name_from_path(workspace_path) if _extract_repo_name_from_path else None
        if not repo_name:
            repo_name = Path(workspace_path).name

        # Get collection name (respect client-supplied name when provided)
        if not collection_name:
            if get_collection_name and repo_name:
                collection_name = get_collection_name(repo_name)
            else:
                collection_name = DEFAULT_COLLECTION

        # Persist origin metadata for remote lookups (including client source_path)
        # Use slugged repo name (repo+16) for state so it matches ingest/watch_index usage
        try:
            if update_repo_origin and repo_name:
                workspace_key = get_workspace_key(workspace_path)
                slug_repo_name = f"{repo_name}-{workspace_key}"
                container_workspace = str(Path(WORK_DIR) / slug_repo_name)
                update_repo_origin(
                    workspace_path=container_workspace,
                    repo_name=slug_repo_name,
                    container_path=container_workspace,
                    source_path=source_path or workspace_path,
                    collection_name=collection_name,
                )
        except Exception as origin_err:
            logger.debug(f"[upload_service] Failed to persist origin info: {origin_err}")

        # Validate bundle size
        if bundle.size and bundle.size > MAX_BUNDLE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Bundle too large. Max size: {MAX_BUNDLE_SIZE_MB}MB"
            )

        # Save bundle to temporary file
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            bundle_path = Path(temp_file.name)

            max_bytes = MAX_BUNDLE_SIZE_MB * 1024 * 1024
            if bundle.size and bundle.size > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Bundle too large. Max size: {MAX_BUNDLE_SIZE_MB}MB"
                )

            # Stream upload to file while enforcing size
            total = 0
            chunk_size = 1024 * 1024
            while True:
                chunk = await bundle.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    try:
                        temp_file.close()
                        bundle_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"Bundle too large. Max size: {MAX_BUNDLE_SIZE_MB}MB"
                    )
                temp_file.write(chunk)

        handed_off = False

        try:
            # Validate bundle format
            manifest = validate_bundle_format(bundle_path)
            bundle_id = manifest.get("bundle_id")
            manifest_sequence = manifest.get("sequence_number")

            # Check sequence number
            last_sequence = get_last_sequence(workspace_path)
            if sequence_number is None:
                if manifest_sequence is not None:
                    sequence_number = manifest_sequence
                else:
                    sequence_number = last_sequence + 1

            if not force and sequence_number is not None:
                if sequence_number != last_sequence + 1:
                    return UploadResponse(
                        success=False,
                        error={
                            "code": "SEQUENCE_MISMATCH",
                            "message": f"Expected sequence {last_sequence + 1}, got {sequence_number}",
                            "expected_sequence": last_sequence + 1,
                            "received_sequence": sequence_number,
                            "retry_after": 5000
                        }
                    )

            handed_off = True

            asyncio.create_task(
                _process_bundle_background(
                    workspace_path=workspace_path,
                    bundle_path=bundle_path,
                    manifest=manifest,
                    sequence_number=sequence_number,
                    bundle_id=bundle_id,
                )
            )

            return UploadResponse(
                success=True,
                bundle_id=bundle_id,
                sequence_number=sequence_number,
                processed_operations=None,
                processing_time_ms=None,
                next_sequence=sequence_number + 1 if sequence_number else None
            )

        finally:
            if not handed_off:
                try:
                    bundle_path.unlink()
                except Exception:
                    pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        return UploadResponse(
            success=False,
            error={
                "code": "PROCESSING_ERROR",
                "message": f"Error processing bundle: {str(e)}"
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error"
            }
        }
    )

def main():
    """Main entry point for the upload service."""
    host = os.environ.get("UPLOAD_SERVICE_HOST", "0.0.0.0")
    port = int(os.environ.get("UPLOAD_SERVICE_PORT", "8002"))

    logger.info(f"Starting upload service on {host}:{port}")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Work directory: {WORK_DIR}")
    logger.info(f"Max bundle size: {MAX_BUNDLE_SIZE_MB}MB")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
