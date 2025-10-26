#!/usr/bin/env python3
"""
Remote upload client for delta bundles in Context-Engine.

This module provides functionality to create and upload delta bundles to a remote
server, enabling real-time code synchronization across distributed environments.
"""

import os
import json
import time
import uuid
import hashlib
import tarfile
import tempfile
import threading
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing workspace state functions
from scripts.workspace_state import (
    get_cached_file_hash,
    set_cached_file_hash,
    remove_cached_file,
)

# Import existing hash function
import scripts.ingest_code as idx


class RemoteUploadClient:
    """Client for uploading delta bundles to remote server."""
    
    def __init__(self,
                 upload_endpoint: str,
                 workspace_path: str,
                 collection_name: str,
                 max_retries: int = 3,
                 timeout: int = 30):
        """
        Initialize remote upload client.
        
        Args:
            upload_endpoint: HTTP endpoint for delta uploads
            workspace_path: Absolute path to workspace
            collection_name: Target collection name
            max_retries: Maximum number of upload retries
            timeout: Request timeout in seconds
        """
        self.upload_endpoint = upload_endpoint.rstrip('/')
        self.workspace_path = workspace_path
        self.collection_name = collection_name
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Bundle persistence directory (initialize before sequence tracking)
        self.bundle_dir = Path(workspace_path) / ".codebase" / "delta_bundles"
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # Sequence number tracking
        self._sequence_lock = threading.Lock()
        self._sequence_number = self._get_last_sequence()
        
        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _get_last_sequence(self) -> int:
        """Get the last sequence number from local state."""
        seq_file = self.bundle_dir / "last_sequence.txt"
        try:
            if seq_file.exists():
                return int(seq_file.read_text().strip())
        except (ValueError, IOError):
            pass
        return 0
    
    def _set_last_sequence(self, sequence: int) -> None:
        """Persist the last sequence number."""
        seq_file = self.bundle_dir / "last_sequence.txt"
        try:
            seq_file.write_text(str(sequence))
        except IOError:
            pass
    
    def _get_next_sequence(self) -> int:
        """Get the next sequence number atomically."""
        with self._sequence_lock:
            self._sequence_number += 1
            self._set_last_sequence(self._sequence_number)
            return self._sequence_number
    
    def detect_file_changes(self, changed_paths: List[Path]) -> Dict[str, List]:
        """
        Detect what type of changes occurred for each file path.
        
        Args:
            changed_paths: List of changed file paths
            
        Returns:
            Dictionary with change types: created, updated, deleted, moved, unchanged
        """
        changes = {
            "created": [],
            "updated": [],
            "deleted": [],
            "moved": [],
            "unchanged": []
        }
        
        for path in changed_paths:
            abs_path = str(path.resolve())
            cached_hash = get_cached_file_hash(self.workspace_path, abs_path)
            
            if not path.exists():
                # File was deleted
                if cached_hash:
                    changes["deleted"].append(path)
            else:
                # File exists - calculate current hash
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                    current_hash = hashlib.sha1(content).hexdigest()
                    
                    if not cached_hash:
                        # New file
                        changes["created"].append(path)
                    elif cached_hash != current_hash:
                        # Modified file
                        changes["updated"].append(path)
                    else:
                        # Unchanged (might be a move detection candidate)
                        changes["unchanged"].append(path)
                        
                    # Update cache
                    set_cached_file_hash(self.workspace_path, abs_path, current_hash)
                except Exception:
                    # Skip files that can't be read
                    continue
        
        # Detect moves by looking for files with same content hash
        # but different paths (requires additional tracking)
        changes["moved"] = self._detect_moves(changes["created"], changes["deleted"])
        
        return changes
    
    def _detect_moves(self, created_files: List[Path], deleted_files: List[Path]) -> List[Tuple[Path, Path]]:
        """
        Detect file moves by matching content hashes between created and deleted files.
        
        Args:
            created_files: List of newly created files
            deleted_files: List of deleted files
            
        Returns:
            List of (source, destination) path tuples for detected moves
        """
        moves = []
        deleted_hashes = {}
        
        # Build hash map for deleted files
        for deleted_path in deleted_files:
            try:
                # Try to get cached hash first, fallback to file content
                cached_hash = get_cached_file_hash(self.workspace_path, str(deleted_path))
                if cached_hash:
                    deleted_hashes[cached_hash] = deleted_path
                    continue
                    
                # If no cached hash, try to read from file if it still exists
                if deleted_path.exists():
                    with open(deleted_path, 'rb') as f:
                        content = f.read()
                    file_hash = hashlib.sha1(content).hexdigest()
                    deleted_hashes[file_hash] = deleted_path
            except Exception:
                continue
        
        # Match created files with deleted files by hash
        for created_path in created_files:
            try:
                with open(created_path, 'rb') as f:
                    content = f.read()
                file_hash = hashlib.sha1(content).hexdigest()
                
                if file_hash in deleted_hashes:
                    source_path = deleted_hashes[file_hash]
                    moves.append((source_path, created_path))
                    # Remove from consideration
                    del deleted_hashes[file_hash]
            except Exception:
                continue
        
        return moves
    
    def create_delta_bundle(self, changes: Dict[str, List]) -> Tuple[str, Dict[str, Any]]:
        """
        Create a delta bundle from detected changes.
        
        Args:
            changes: Dictionary of file changes by type
            
        Returns:
            Tuple of (bundle_path, manifest_metadata)
        """
        bundle_id = str(uuid.uuid4())
        sequence_number = self._get_next_sequence()
        created_at = datetime.now().isoformat()
        
        # Create temporary directory for bundle
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            files_dir = temp_path / "files"
            metadata_dir = temp_path / "metadata"
            files_dir.mkdir()
            metadata_dir.mkdir()
            
            # Create subdirectories
            (files_dir / "created").mkdir()
            (files_dir / "updated").mkdir()
            (files_dir / "moved").mkdir()
            
            operations = []
            total_size = 0
            file_hashes = {}
            
            # Process created files
            for path in changes["created"]:
                rel_path = str(path.relative_to(Path(self.workspace_path)))
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                    file_hash = hashlib.sha1(content).hexdigest()
                    content_hash = f"sha1:{file_hash}"
                    
                    # Write file to bundle
                    bundle_file_path = files_dir / "created" / rel_path
                    bundle_file_path.parent.mkdir(parents=True, exist_ok=True)
                    bundle_file_path.write_bytes(content)
                    
                    # Get file info
                    stat = path.stat()
                    language = idx.CODE_EXTS.get(path.suffix.lower(), "unknown")
                    
                    operation = {
                        "operation": "created",
                        "path": rel_path,
                        "relative_path": rel_path,
                        "absolute_path": str(path.resolve()),
                        "size_bytes": stat.st_size,
                        "content_hash": content_hash,
                        "file_hash": f"sha1:{idx.hash_id(content.decode('utf-8', errors='ignore'), rel_path, 1, len(content.splitlines()))}",
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "language": language
                    }
                    operations.append(operation)
                    file_hashes[rel_path] = f"sha1:{file_hash}"
                    total_size += stat.st_size
                    
                except Exception as e:
                    print(f"[bundle_create] Error processing created file {path}: {e}")
                    continue
            
            # Process updated files
            for path in changes["updated"]:
                rel_path = str(path.relative_to(Path(self.workspace_path)))
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                    file_hash = hashlib.sha1(content).hexdigest()
                    content_hash = f"sha1:{file_hash}"
                    previous_hash = get_cached_file_hash(self.workspace_path, str(path.resolve()))
                    
                    # Write file to bundle
                    bundle_file_path = files_dir / "updated" / rel_path
                    bundle_file_path.parent.mkdir(parents=True, exist_ok=True)
                    bundle_file_path.write_bytes(content)
                    
                    # Get file info
                    stat = path.stat()
                    language = idx.CODE_EXTS.get(path.suffix.lower(), "unknown")
                    
                    operation = {
                        "operation": "updated",
                        "path": rel_path,
                        "relative_path": rel_path,
                        "absolute_path": str(path.resolve()),
                        "size_bytes": stat.st_size,
                        "content_hash": content_hash,
                        "previous_hash": f"sha1:{previous_hash}" if previous_hash else None,
                        "file_hash": f"sha1:{idx.hash_id(content.decode('utf-8', errors='ignore'), rel_path, 1, len(content.splitlines()))}",
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "language": language
                    }
                    operations.append(operation)
                    file_hashes[rel_path] = f"sha1:{file_hash}"
                    total_size += stat.st_size
                    
                except Exception as e:
                    print(f"[bundle_create] Error processing updated file {path}: {e}")
                    continue
            
            # Process moved files
            for source_path, dest_path in changes["moved"]:
                dest_rel_path = str(dest_path.relative_to(Path(self.workspace_path)))
                source_rel_path = str(source_path.relative_to(Path(self.workspace_path)))
                try:
                    with open(dest_path, 'rb') as f:
                        content = f.read()
                    file_hash = hashlib.sha1(content).hexdigest()
                    content_hash = f"sha1:{file_hash}"
                    
                    # Write file to bundle
                    bundle_file_path = files_dir / "moved" / dest_rel_path
                    bundle_file_path.parent.mkdir(parents=True, exist_ok=True)
                    bundle_file_path.write_bytes(content)
                    
                    # Get file info
                    stat = dest_path.stat()
                    language = idx.CODE_EXTS.get(dest_path.suffix.lower(), "unknown")
                    
                    operation = {
                        "operation": "moved",
                        "path": dest_rel_path,
                        "relative_path": dest_rel_path,
                        "absolute_path": str(dest_path.resolve()),
                        "source_path": source_rel_path,
                        "source_relative_path": source_rel_path,
                        "source_absolute_path": str(source_path.resolve()),
                        "size_bytes": stat.st_size,
                        "content_hash": content_hash,
                        "file_hash": f"sha1:{idx.hash_id(content.decode('utf-8', errors='ignore'), dest_rel_path, 1, len(content.splitlines()))}",
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "language": language
                    }
                    operations.append(operation)
                    file_hashes[dest_rel_path] = f"sha1:{file_hash}"
                    total_size += stat.st_size
                    
                except Exception as e:
                    print(f"[bundle_create] Error processing moved file {source_path} -> {dest_path}: {e}")
                    continue
            
            # Process deleted files
            for path in changes["deleted"]:
                rel_path = str(path.relative_to(Path(self.workspace_path)))
                try:
                    previous_hash = get_cached_file_hash(self.workspace_path, str(path.resolve()))
                    
                    operation = {
                        "operation": "deleted",
                        "path": rel_path,
                        "relative_path": rel_path,
                        "absolute_path": str(path.resolve()),
                        "previous_hash": f"sha1:{previous_hash}" if previous_hash else None,
                        "file_hash": None,
                        "modified_time": datetime.now().isoformat(),
                        "language": idx.CODE_EXTS.get(path.suffix.lower(), "unknown")
                    }
                    operations.append(operation)
                    
                except Exception as e:
                    print(f"[bundle_create] Error processing deleted file {path}: {e}")
                    continue
            
            # Create manifest
            manifest = {
                "version": "1.0",
                "bundle_id": bundle_id,
                "workspace_path": self.workspace_path,
                "collection_name": self.collection_name,
                "created_at": created_at,
                "sequence_number": sequence_number,
                "parent_sequence": sequence_number - 1,
                "operations": {
                    "created": len(changes["created"]),
                    "updated": len(changes["updated"]),
                    "deleted": len(changes["deleted"]),
                    "moved": len(changes["moved"])
                },
                "total_files": len(operations),
                "total_size_bytes": total_size,
                "compression": "gzip",
                "encoding": "utf-8"
            }
            
            # Write manifest
            (temp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
            
            # Write operations metadata
            operations_metadata = {
                "operations": operations
            }
            (metadata_dir / "operations.json").write_text(json.dumps(operations_metadata, indent=2))
            
            # Write hashes
            hashes_metadata = {
                "workspace_path": self.workspace_path,
                "updated_at": created_at,
                "file_hashes": file_hashes
            }
            (metadata_dir / "hashes.json").write_text(json.dumps(hashes_metadata, indent=2))
            
            # Create tarball
            bundle_path = self.bundle_dir / f"{bundle_id}.tar.gz"
            with tarfile.open(bundle_path, "w:gz") as tar:
                tar.add(temp_path, arcname=f"{bundle_id}")
            
            return str(bundle_path), manifest
    
    def upload_bundle(self, bundle_path: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upload delta bundle to remote server with exponential backoff retry.
        
        Args:
            bundle_path: Path to the bundle tarball
            manifest: Bundle manifest metadata
            
        Returns:
            Server response dictionary
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Calculate backoff delay (exponential with jitter)
                if attempt > 0:
                    base_delay = 2 ** (attempt - 1)  # 1, 2, 4, 8...
                    jitter = base_delay * 0.1 * (0.5 + (hash(str(time.time())) % 100) / 100)
                    delay = min(base_delay + jitter, 30)  # Cap at 30 seconds
                    logger.info(f"[remote_upload] Retry attempt {attempt + 1}/{self.max_retries + 1} after {delay:.2f}s delay")
                    time.sleep(delay)
                
                # Verify bundle exists before attempting upload
                if not os.path.exists(bundle_path):
                    return {
                        "success": False,
                        "error": {
                            "code": "BUNDLE_NOT_FOUND",
                            "message": f"Bundle file not found: {bundle_path}"
                        }
                    }
                
                # Check bundle size
                bundle_size = os.path.getsize(bundle_path)
                max_size_mb = 100  # Default max size
                max_size_bytes = max_size_mb * 1024 * 1024
                
                if bundle_size > max_size_bytes:
                    return {
                        "success": False,
                        "error": {
                            "code": "BUNDLE_TOO_LARGE",
                            "message": f"Bundle size {bundle_size} bytes exceeds maximum {max_size_bytes} bytes"
                        }
                    }
                
                with open(bundle_path, 'rb') as bundle_file:
                    files = {
                        'bundle': (f"{manifest['bundle_id']}.tar.gz", bundle_file, 'application/gzip')
                    }
                    
                    data = {
                        'workspace_path': self.workspace_path,
                        'collection_name': self.collection_name,
                        'sequence_number': str(manifest['sequence_number']),
                        'force': 'false'
                    }
                    
                    logger.info(f"[remote_upload] Uploading bundle {manifest['bundle_id']} (size: {bundle_size} bytes)")
                    
                    response = self.session.post(
                        f"{self.upload_endpoint}/api/v1/delta/upload",
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        logger.info(f"[remote_upload] Successfully uploaded bundle {manifest['bundle_id']}")
                        return result
                    else:
                        error_msg = f"Upload failed with status {response.status_code}"
                        try:
                            error_detail = response.json()
                            error_detail_msg = error_detail.get('error', {}).get('message', 'Unknown error')
                            error_msg += f": {error_detail_msg}"
                            error_code = error_detail.get('error', {}).get('code', 'HTTP_ERROR')
                        except:
                            error_msg += f": {response.text[:200]}"  # Truncate long responses
                            error_code = "HTTP_ERROR"
                        
                        last_error = {
                            "success": False,
                            "error": {
                                "code": error_code,
                                "message": error_msg,
                                "status_code": response.status_code
                            }
                        }
                        
                        # Don't retry on client errors (4xx)
                        if 400 <= response.status_code < 500 and response.status_code != 429:
                            logger.warning(f"[remote_upload] Client error {response.status_code}, not retrying: {error_msg}")
                            return last_error
                        
                        logger.warning(f"[remote_upload] Upload attempt {attempt + 1} failed: {error_msg}")
                        
            except requests.exceptions.Timeout as e:
                last_error = {
                    "success": False,
                    "error": {
                        "code": "TIMEOUT_ERROR",
                        "message": f"Upload timeout after {self.timeout}s: {str(e)}"
                    }
                }
                logger.warning(f"[remote_upload] Upload timeout on attempt {attempt + 1}: {e}")
                
            except requests.exceptions.ConnectionError as e:
                last_error = {
                    "success": False,
                    "error": {
                        "code": "CONNECTION_ERROR",
                        "message": f"Connection error during upload: {str(e)}"
                    }
                }
                logger.warning(f"[remote_upload] Connection error on attempt {attempt + 1}: {e}")
                
            except requests.exceptions.RequestException as e:
                last_error = {
                    "success": False,
                    "error": {
                        "code": "NETWORK_ERROR",
                        "message": f"Network error during upload: {str(e)}"
                    }
                }
                logger.warning(f"[remote_upload] Network error on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_error = {
                    "success": False,
                    "error": {
                        "code": "UPLOAD_ERROR",
                        "message": f"Unexpected error during upload: {str(e)}"
                    }
                }
                logger.error(f"[remote_upload] Unexpected error on attempt {attempt + 1}: {e}")
        
        # All retries exhausted
        logger.error(f"[remote_upload] All {self.max_retries + 1} upload attempts failed for bundle {manifest.get('bundle_id', 'unknown')}")
        return last_error or {
            "success": False,
            "error": {
                "code": "MAX_RETRIES_EXCEEDED",
                "message": f"Upload failed after {self.max_retries + 1} attempts"
            }
        }
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get server status and last sequence number with enhanced error handling."""
        try:
            logger.debug(f"[remote_upload] Checking server status at {self.upload_endpoint}")
            
            response = self.session.get(
                f"{self.upload_endpoint}/api/v1/delta/status",
                params={'workspace_path': self.workspace_path},
                timeout=min(self.timeout, 10)  # Use shorter timeout for status checks
            )
            
            if response.status_code == 200:
                status_data = response.json()
                logger.debug(f"[remote_upload] Server status: {status_data}")
                return status_data
            else:
                error_msg = f"Status check failed with HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    error_detail_msg = error_detail.get('error', {}).get('message', 'Unknown error')
                    error_msg += f": {error_detail_msg}"
                except:
                    error_msg += f": {response.text[:100]}"
                
                logger.warning(f"[remote_upload] {error_msg}")
                return {
                    "success": False,
                    "error": {
                        "code": "STATUS_ERROR",
                        "message": error_msg,
                        "status_code": response.status_code
                    }
                }
                
        except requests.exceptions.Timeout as e:
            error_msg = f"Status check timeout after {min(self.timeout, 10)}s"
            logger.warning(f"[remote_upload] {error_msg}: {e}")
            return {
                "success": False,
                "error": {
                    "code": "STATUS_TIMEOUT",
                    "message": error_msg
                }
            }
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Cannot connect to server at {self.upload_endpoint}"
            logger.warning(f"[remote_upload] {error_msg}: {e}")
            return {
                "success": False,
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": error_msg
                }
            }
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during status check: {str(e)}"
            logger.warning(f"[remote_upload] {error_msg}")
            return {
                "success": False,
                "error": {
                    "code": "NETWORK_ERROR",
                    "message": error_msg
                }
            }
        except Exception as e:
            error_msg = f"Unexpected error during status check: {str(e)}"
            logger.error(f"[remote_upload] {error_msg}")
            return {
                "success": False,
                "error": {
                    "code": "STATUS_CHECK_ERROR",
                    "message": error_msg
                }
            }
    
    def has_meaningful_changes(self, changes: Dict[str, List]) -> bool:
        """Check if changes warrant a delta upload."""
        total_changes = sum(len(files) for op, files in changes.items() if op != "unchanged")
        return total_changes > 0
    
    def process_and_upload_changes(self, changed_paths: List[Path]) -> bool:
        """
        Process changed paths and upload delta bundle if meaningful changes exist.
        Includes comprehensive error handling and graceful fallback.
        
        Args:
            changed_paths: List of changed file paths
            
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            logger.info(f"[remote_upload] Processing {len(changed_paths)} changed paths")
            
            # Validate input
            if not changed_paths:
                logger.info("[remote_upload] No changed paths provided")
                return True
            
            # Detect changes
            try:
                changes = self.detect_file_changes(changed_paths)
            except Exception as e:
                logger.error(f"[remote_upload] Error detecting file changes: {e}")
                return False
            
            if not self.has_meaningful_changes(changes):
                logger.info("[remote_upload] No meaningful changes detected, skipping upload")
                return True
            
            # Log change summary
            total_changes = sum(len(files) for op, files in changes.items() if op != "unchanged")
            logger.info(f"[remote_upload] Detected {total_changes} meaningful changes: "
                       f"{len(changes['created'])} created, {len(changes['updated'])} updated, "
                       f"{len(changes['deleted'])} deleted, {len(changes['moved'])} moved")
            
            # Create delta bundle
            bundle_path = None
            try:
                bundle_path, manifest = self.create_delta_bundle(changes)
                logger.info(f"[remote_upload] Created delta bundle: {manifest['bundle_id']} "
                           f"(seq: {manifest['sequence_number']}, size: {manifest['total_size_bytes']} bytes)")
                
                # Validate bundle was created successfully
                if not bundle_path or not os.path.exists(bundle_path):
                    raise RuntimeError(f"Failed to create bundle at {bundle_path}")
                
            except Exception as e:
                logger.error(f"[remote_upload] Error creating delta bundle: {e}")
                return False
            
            # Upload bundle with retry logic
            try:
                response = self.upload_bundle(bundle_path, manifest)
                
                if response.get("success", False):
                    processed_ops = response.get('processed_operations', {})
                    logger.info(f"[remote_upload] Successfully uploaded bundle {manifest['bundle_id']}")
                    logger.info(f"[remote_upload] Processed operations: {processed_ops}")
                    
                    # Clean up local bundle after successful upload
                    try:
                        if os.path.exists(bundle_path):
                            os.remove(bundle_path)
                            logger.debug(f"[remote_upload] Cleaned up local bundle: {bundle_path}")
                    except Exception as cleanup_error:
                        logger.warning(f"[remote_upload] Failed to cleanup bundle {bundle_path}: {cleanup_error}")
                    
                    return True
                else:
                    error = response.get("error", {})
                    error_code = error.get("code", "UNKNOWN")
                    error_msg = error.get("message", "Unknown error")
                    
                    logger.error(f"[remote_upload] Upload failed: {error_msg}")
                    
                    # Handle specific error types
                    if error_code == "SEQUENCE_MISMATCH":
                        logger.info("[remote_upload] Attempting to handle sequence mismatch")
                        return self._handle_sequence_mismatch(response, manifest)
                    elif error_code in ["BUNDLE_TOO_LARGE", "BUNDLE_NOT_FOUND"]:
                        # These are unrecoverable errors
                        logger.error(f"[remote_upload] Unrecoverable error ({error_code}): {error_msg}")
                        return False
                    elif error_code in ["TIMEOUT_ERROR", "CONNECTION_ERROR", "NETWORK_ERROR"]:
                        # These might be temporary, suggest fallback
                        logger.warning(f"[remote_upload] Network-related error ({error_code}): {error_msg}")
                        logger.warning("[remote_upload] Consider falling back to local mode if this persists")
                        return False
                    else:
                        # Other errors
                        logger.error(f"[remote_upload] Upload error ({error_code}): {error_msg}")
                        return False
                        
            except Exception as e:
                logger.error(f"[remote_upload] Unexpected error during upload: {e}")
                return False
                
        except Exception as e:
            logger.error(f"[remote_upload] Critical error in process_and_upload_changes: {e}")
            logger.exception("[remote_upload] Full traceback:")
            return False
    
    def _handle_sequence_mismatch(self, error_response: Dict[str, Any], manifest: Dict[str, Any]) -> bool:
        """Handle sequence number mismatch by recovering missing bundles."""
        try:
            expected_seq = error_response["error"]["expected_sequence"]
            current_seq = manifest["sequence_number"]
            
            print(f"[remote_upload] Sequence mismatch: expected {expected_seq}, got {current_seq}")
            
            # For PoC, we'll just force upload with the expected sequence
            # In a production system, we would implement proper recovery
            print(f"[remote_upload] Forcing upload with sequence {expected_seq}")
            
            # Update our sequence number
            with self._sequence_lock:
                self._sequence_number = expected_seq
                self._set_last_sequence(expected_seq)
            
            # Retry upload with force=true
            bundle_path = self.bundle_dir / f"{manifest['bundle_id']}.tar.gz"
            if bundle_path.exists():
                data = {
                    'workspace_path': self.workspace_path,
                    'collection_name': self.collection_name,
                    'sequence_number': str(expected_seq),
                    'force': 'true'
                }
                
                with open(bundle_path, 'rb') as bundle_file:
                    files = {
                        'bundle': (f"{manifest['bundle_id']}.tar.gz", bundle_file, 'application/gzip')
                    }
                    
                    response = self.session.post(
                        f"{self.upload_endpoint}/api/v1/delta/upload",
                        files=files,
                        data=data,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success", False):
                            print(f"[remote_upload] Force upload successful for bundle {manifest['bundle_id']}")
                            return True
            
            print(f"[remote_upload] Force upload failed for bundle {manifest['bundle_id']}")
            return False
            
        except Exception as e:
            print(f"[remote_upload] Error handling sequence mismatch: {e}")
            return False


def is_remote_mode_enabled() -> bool:
    """Check if remote upload mode is enabled via environment variables."""
    return os.environ.get("REMOTE_UPLOAD_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def get_remote_config() -> Dict[str, str]:
    """Get remote upload configuration from environment variables."""
    return {
        "upload_endpoint": os.environ.get("REMOTE_UPLOAD_ENDPOINT", "http://localhost:8080"),
        "workspace_path": os.environ.get("WATCH_ROOT", os.environ.get("WORKSPACE_PATH", "/work")),
        "collection_name": os.environ.get("COLLECTION_NAME", "my-collection"),
        "max_retries": int(os.environ.get("REMOTE_UPLOAD_MAX_RETRIES", "3")),
        "timeout": int(os.environ.get("REMOTE_UPLOAD_TIMEOUT", "30"))
    }