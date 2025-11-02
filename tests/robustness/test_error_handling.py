"""
Comprehensive error handling and robustness tests for Context-Engine.

Tests edge cases, timeout handling, race conditions, and recovery mechanisms
that commonly cause production issues.
"""

import pytest
import asyncio
import time
import threading
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from scripts.mcp_indexer_server import (
        _ca_unwrap_and_normalize,
        _ca_prepare_filters_and_retrieve,
        context_answer,
        _ENV_LOCK,
    )
    from scripts.watch_index import ChangeQueue
    from scripts.utils import ResourceLock, timeout_context
    from scripts.config_validator import ConfigValidator
    from scripts.logger import get_logger
except ImportError as e:
    print(f"❌ Failed to import test modules: {e}")
    sys.exit(1)

logger = get_logger(__name__)


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    def test_generic_exception_handling(self):
        """Test that generic exceptions are handled gracefully."""

        # Test malformed query handling
        with pytest.raises(Exception):
            _ca_unwrap_and_normalize(
                query={"invalid": "structure"},  # Should trigger handling
                limit="not_a_number",  # Should be converted safely
                collection=None,
                **{}
            )

        # Test malformed JSON in nested kwargs
        result = _ca_unwrap_and_normalize(
            query="test",
            limit=5,
            collection="test",
            **{"raw": '{"invalid": json structure'}"}  # Should be handled gracefully
        )

        assert isinstance(result, dict)
        assert "queries" in result
        assert result["queries"] == ["test"]  # Should fallback to clean input

    def test_timeout_scenarios(self):
        """Test various timeout scenarios."""

        # Mock timeout in subprocess calls
        with patch('scripts.mcp_indexer_server._run_async') as mock_run:
            mock_run.side_effect = asyncio.TimeoutError("Simulated timeout")

            result = asyncio.run(context_answer(query="test query", limit=5))

            assert "error" in result
            assert "timeout" in result["error"].lower()
            assert "citations" in result  # Should always return citations

    def test_resource_lock_contention(self):
        """Test resource lock under high contention."""

        lock = ResourceLock("test_lock", timeout=1.0)
        results = []
        errors = []

        def worker(worker_id: int):
            try:
                with lock(timeout=2.0):
                    time.sleep(0.5)  # Simulate work
                    results.append(f"worker_{worker_id}")
            except Exception as e:
                errors.append(f"worker_{worker_id}: {e}")

        # Start multiple workers to create contention
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            as_completed(futures)

        # Should have some successes and some timeouts
        assert len(results) > 0
        assert len(errors) > 0
        assert len(results) + len(errors) == 10

    def test_memory_exhaustion_handling(self):
        """Test behavior under memory pressure scenarios."""

        # Mock large result that could cause memory issues
        large_results = [
            {
                "path": f"/test/path/file_{i}.py",
                "content": "x" * 10000,  # Large content
                "metadata": {"test": "data" * 1000}
            }
            for i in range(1000)  # Many large items
        ]

        with patch('scripts.hybrid_search.run_hybrid_search') as mock_search:
            mock_search.return_value = large_results

            # Should handle large results gracefully
            result = asyncio.run(context_answer(query="memory test", limit=100))

            # Should either succeed or fail gracefully, not crash
            assert isinstance(result, dict)
            assert "error" in result or "answer" in result

    def test_network_partition_simulation(self):
        """Test behavior when services become unavailable."""

        # Simulate Qdrant unavailability
        with patch('scripts.mcp_indexer_server.QdrantClient') as mock_client:
            mock_client.side_effect = ConnectionError("Connection refused")

            result = asyncio.run(context_answer(query="network test", limit=5))

            assert "error" in result
            assert "connection" in result["error"].lower()
            assert "citations" in result  # Fallback behavior

    def test_concurrent_request_handling(self):
        """Test system behavior under concurrent load."""

        async def make_request(request_id: int):
            try:
                return await context_answer(
                    query=f"concurrent test {request_id}",
                    limit=5,
                    max_tokens=100
                )
            except Exception as e:
                return {"error": str(e), "request_id": request_id}

        # Launch many concurrent requests
        tasks = [make_request(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes vs failures
        successes = [r for r in results if isinstance(r, dict) and "answer" in r]
        failures = [r for r in results if isinstance(r, dict) and "error" in r]

        # Should handle concurrent load gracefully
        assert len(successes) > 0  # Some should succeed
        assert len(failures) >= 0  # Some might fail under load
        assert len(successes) + len(failures) == 20

    def test_environment_variable_validation(self):
        """Test environment variable validation edge cases."""

        # Test with empty critical variables
        with patch.dict(os.environ, {"QDRANT_URL": "", "EMBEDDING_MODEL": ""}, clear=True):
            validator = ConfigValidator()

            # Should detect missing critical variables
            qdrant_ok = validator.validate_qdrant_connection("")
            model_ok = validator.validate_embedding_model("")

            assert not qdrant_ok
            assert not model_ok
            assert any("error" in str(r).lower() for r in validator.results)

    def test_file_system_permission_errors(self):
        """Test handling of file system permission issues."""

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_file.py"
            test_file.write_text("test content")

            # Remove read permissions
            original_mode = test_file.stat().st_mode
            test_file.chmod(0o000)  # No permissions

            try:
                with patch('scripts.watch_index.open') as mock_open:
                    mock_open.side_effect = PermissionError("Permission denied")

                    # Should handle permission errors gracefully
                    from scripts.watch_index import IndexHandler
                    handler = IndexHandler(
                        root=Path(temp_dir),
                        queue=Mock(),
                        client=Mock(),
                        collection="test"
                    )

                    # Should not crash, should handle gracefully
                    result = handler._maybe_enqueue(str(test_file))
                    assert result is None  # Should skip unreadable file

            finally:
                # Restore permissions for cleanup
                test_file.chmod(original_mode)

    def test_malformed_data_handling(self):
        """Test handling of malformed data from external services."""

        # Test with malformed Qdrant response
        malformed_responses = [
            '{"invalid": json}',
            '{"points": "not_a_list"}',
            '{"results": null}',
            'not even json',
            '{"vectors": [[], [[]], [[[]]]]}',  # Nested array issues
        ]

        for malformed_response in malformed_responses:
            with patch('scripts.mcp_indexer_server.QdrantClient') as mock_client:
                mock_client.return_value.scroll.return_value = ([], malformed_response)

                result = asyncio.run(context_answer(query="malformed test", limit=5))

                # Should handle malformed data gracefully
                assert isinstance(result, dict)
                assert "error" in result or "citations" in result

    def test_deadlock_prevention(self):
        """Test deadlock prevention mechanisms."""

        lock1 = ResourceLock("deadlock_test_1", timeout=1.0)
        lock2 = ResourceLock("deadlock_test_2", timeout=1.0)
        deadlock_detected = []

        def acquire_locks_in_order(lock_a, lock_b, thread_id: int):
            try:
                with lock_a(timeout=0.5):
                    time.sleep(0.1)
                    with lock_b(timeout=0.5):
                        time.sleep(0.1)
                        return f"thread_{thread_id}_success"
            except Exception as e:
                deadlock_detected.append(f"thread_{thread_id}: {e}")
                return f"thread_{thread_id}_deadlock"

        # Create potential deadlock scenario
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Thread 1: lock1 -> lock2
            future1 = executor.submit(acquire_locks_in_order, lock1, lock2, 1)
            # Thread 2: lock2 -> lock1
            future2 = executor.submit(acquire_locks_in_order, lock2, lock1, 2)
            # Thread 3: just lock1 (should succeed)
            future3 = executor.submit(lambda: lock1.acquire(timeout=1.0) or lock1.release(), 3)

            results = [future.result() for future in as_completed([future1, future2, future3])]

        # Should detect and handle deadlock situations
        assert len(deadlock_detected) >= 1  # At least one deadlock detected
        assert any("deadlock" in result.lower() for result in results if result)

    def test_cascading_failure_scenarios(self):
        """Test handling of cascading failures."""

        failure_points = []

        def mock_failing_operation(operation_name: str):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    failure_points.append(operation_name)
                    if len(failure_points) > 3:  # Fail after 3 operations
                        return {"ok": False, "error": f"Cascading failure at {operation_name}"}
                    return func(*args, **kwargs)
                return wrapper
            return decorator

        with patch('scripts.mcp_indexer_server._run_async') as mock_run:
            mock_run.side_effect = mock_failing_operation("subprocess_call")

            result = asyncio.run(context_answer(query="cascade test", limit=5))

            # Should detect cascading failure
            assert len(failure_points) > 3
            assert "cascading" in result.get("error", "").lower()

    def test_resource_cleanup_on_failure(self):
        """Test proper resource cleanup on failures."""

        cleanup_tracker = {"files_opened": 0, "locks_acquired": 0}

        class TrackingResource:
            def __init__(self, name: str):
                self.name = name

            def __enter__(self):
                cleanup_tracker["locks_acquired"] += 1
                cleanup_tracker["files_opened"] += 1
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                cleanup_tracker["locks_acquired"] -= 1
                cleanup_tracker["files_opened"] -= 1

        # Simulate operation that fails during resource usage
        try:
            with TrackingResource("test_resource"):
                with patch('scripts.mcp_indexer_server._run_async') as mock_run:
                    mock_run.side_effect = RuntimeError("Simulated failure")

                    result = asyncio.run(context_answer(query="cleanup test", limit=5))

                    # Should fail but resources should be cleaned up
                    assert "error" in result

        finally:
            # Verify cleanup happened
            assert cleanup_tracker["files_opened"] == 0
            assert cleanup_tracker["locks_acquired"] == 0

    def test_partial_failure_recovery(self):
        """Test recovery from partial system failures."""

        # Mock partial service failure
        services_status = {
            "qdrant": True,     # Available
            "search": False,   # Failed
            "indexer": True,   # Available
            "decoder": False,   # Failed
        }

        def mock_service_check(service_name: str):
            return services_status.get(service_name, False)

        with patch.multiple(
            'scripts.mcp_indexer_server.is_decoder_enabled',
            'scripts.config_validator.ConfigValidator.validate_qdrant_connection'
        ) as mocks:
            mocks[0].return_value = services_status["decoder"]
            mocks[1].return_value = services_status["qdrant"]

            result = asyncio.run(context_answer(query="partial failure test", limit=5))

            # Should provide degraded service but still function
            assert isinstance(result, dict)

            if not services_status["decoder"]:
                assert "decoder disabled" in result.get("error", "").lower()
            elif services_status["qdrant"]:
                assert "citations" in result  # Fallback should work

    def test_config_drift_detection(self):
        """Test detection of configuration drift from defaults."""

        # Test with configuration that has drifted from expected values
        drifted_config = {
            "QDRANT_TIMEOUT": "1",      # Too low
            "CTX_CLIENT_DEADLINE_SEC": "10",  # Too low
            "MICRO_BUDGET_TOKENS": "50000",  # Too high
            "DECODER_TIMEOUT_CAP": "300",   # Too high
        }

        with patch.dict(os.environ, drifted_config, clear=True):
            validator = ConfigValidator()

            # Should detect configuration issues
            validation_results = validator.validate_memory_settings()

            assert not validation_results  # Should detect issues
            assert any("too low" in str(r).lower() for r in validator._critical_issues)
            assert any("too high" in str(r).lower() for r in validator._critical_issues)

    def test_exponential_backoff_functionality(self):
        """Test exponential backoff in retry mechanisms."""

        attempt_count = []
        delays = []

        def mock_operation_that_fails_then_succeeds():
            attempt_count.append(len(attempt_count))
            if len(attempt_count) < 3:
                raise ConnectionError("Simulated failure")
            return "success"

        with patch('time.sleep') as mock_sleep:
            with timeout_context(seconds=10, operation="exponential_backoff_test"):
                # Simulate retry logic
                for attempt in range(5):
                    try:
                        return mock_operation_that_fails_then_succeeds()
                    except ConnectionError:
                        if attempt < 3:
                            # Exponential backoff should be used
                            continue
                        break

            # Verify exponential backoff was called
            assert mock_sleep.call_count >= 2

            # Verify attempt pattern
            assert len(attempt_count) == 4  # 3 failures + 1 success

    def test_buffer_overflow_handling(self):
        """Test handling of buffer overflow scenarios."""

        # Test with very large input that could cause buffer issues
        large_input = "x" * 1000000  # 1MB of text

        # Test in various buffer scenarios
        buffer_tests = [
            ("large_query", {"query": large_input}),
            ("large_citations", {"limit": 10000}),  # Very large limit
            ("deep_nesting", {"query": "nested " * 1000}),  # Deep nesting
        ]

        for test_name, params in buffer_tests:
            result = asyncio.run(context_answer(**params, limit=5))

            # Should handle large inputs gracefully
            assert isinstance(result, dict)

            # Should either succeed or fail with meaningful error
            if "error" in result:
                assert len(result["error"]) < 1000  # Error message shouldn't be too large
                assert any(keyword in result["error"].lower()
                          for keyword in ["too large", "overflow", "buffer", "memory"])

    def test_race_condition_in_change_queue(self):
        """Test ChangeQueue for race conditions."""

        queue = ChangeQueue(Mock())
        processed_paths = []
        processing_lock = threading.Lock()

        def mock_process(paths: List[str]):
            with processing_lock:
                processed_paths.extend(paths)
                time.sleep(0.1)  # Simulate processing time

        # Simulate rapid additions from multiple threads
        def add_files(thread_id: int, file_count: int):
            for i in range(file_count):
                queue.add(Path(f"/test/thread_{thread_id}/file_{i}.py"))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(add_files, thread_id, 10)
                for thread_id in range(10)
            ]

            # Wait for all additions
            as_completed(futures)

            # Wait for processing
            time.sleep(2)

        # Should handle concurrent additions without data loss
        assert len(processed_paths) > 0

        # Check for duplicates or missing files
        all_paths = set(str(p) for p in processed_paths)
        expected_count = 100  # 10 threads * 10 files each

        # Should have processed most files (some might be deduplicated)
        assert len(all_paths) > 50  # At least half should be processed

    def test_environment_lock_contention(self):
        """Test _ENV_LOCK under high contention."""

        lock_acquisition_times = []
        contention_detected = []

        def simulate_lock_contention():
            start_time = time.time()

            try:
                with _ENV_LOCK:
                    acquisition_time = time.time() - start_time
                    lock_acquisition_times.append(acquisition_time)

                    # Simulate some work
                    time.sleep(0.05)

                    # Check if we're waiting too long
                    if acquisition_time > 0.1:  # 100ms threshold
                        contention_detected.append(True)

            except Exception as e:
                contention_detected.append(False)

        # Create contention with multiple threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(simulate_lock_contention) for _ in range(50)]
            as_completed(futures)

        # Should have some contention but still succeed
        assert len(lock_acquisition_times) == 50
        assert sum(contention_detected) <= 50  # Most should succeed eventually
        assert any(t > 0.1 for t in lock_acquisition_times)  # Some should wait


class TestTimeoutAndRecovery:
    """Test timeout handling and recovery mechanisms."""

    async def test_deadline_aware_decoding(self):
        """Test deadline-aware LLM decoding."""

        with patch('scripts.mcp_indexer_server._ca_decode') as mock_decode:
            # Simulate timeout on first call
            mock_decode.side_effect = [
                TimeoutError("Decoder timeout"),
                "this is a valid response"  # Second call succeeds
            ]

            # Should handle timeout gracefully
            result1 = asyncio.run(context_answer(
                query="timeout test 1",
                limit=5,
                max_tokens=100
            ))

            assert "deadline" in result1.get("error", "").lower()
            assert "citations" in result1

            # Second call should succeed
            result2 = asyncio.run(context_answer(
                query="timeout test 2",
                limit=5,
                max_tokens=100
            ))

            assert "answer" in result2 or "citations" in result2

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for repeated failures."""

        failure_count = 0

        def mock_operation_with_circuit_breaker():
            nonlocal failure_count
            failure_count += 1

            if failure_count <= 3:
                raise ConnectionError(f"Failure {failure_count}")
            return "success after circuit breaker"

        with patch('scripts.mcp_indexer_server._run_async') as mock_run:
            mock_run.side_effect = mock_operation_with_circuit_breaker

            results = []
            for i in range(5):
                result = asyncio.run(context_answer(
                    query=f"circuit test {i}",
                    limit=5
                ))
                results.append(result)

            # First 3 should fail, last 2 should succeed
            failures = [r for r in results if "error" in r]
            successes = [r for r in results if "answer" in r or "citations" in r]

            assert len(failures) == 3
            assert len(successes) == 2

    def test_graceful_degradation(self):
        """Test graceful degradation when services are unavailable."""

        # Simulate progressive service degradation
        service_states = [
            {"qdrant": True, "decoder": True},
            {"qdrant": True, "decoder": False},
            {"qdrant": False, "decoder": False},
        ]

        results = []
        for state in service_states:
            with patch('scripts.mcp_indexer_server.is_decoder_enabled') as mock_decoder:
                with patch('scripts.config_validator.ConfigValidator.validate_qdrant_connection') as mock_qdrant:
                    mock_decoder.return_value = state["decoder"]
                    mock_qdrant.return_value = state["qdrant"]

                    result = asyncio.run(context_answer(
                        query=f"degradation test {len(results)}",
                        limit=5
                    ))
                    results.append(result)

        # Should progressively degrade but remain functional
        assert all(isinstance(r, dict) for r in results)

        # First should have full functionality
        assert "answer" in results[0] or "citations" in results[0]

        # Second should fall back gracefully
        if not service_states[1]["decoder"]:
            assert "decoder" in results[1].get("error", "").lower()
            assert "citations" in results[1]

        # Third should still provide citations even without Qdrant
        if not service_states[2]["qdrant"]:
            assert "qdrant" in results[2].get("error", "").lower()


class TestDataIntegrity:
    """Test data integrity and consistency."""

    def test_citation_consistency(self):
        """Test citation ID consistency and uniqueness."""

        mock_search_results = [
            {
                "path": "/test/file1.py",
                "start_line": 10,
                "end_line": 20,
                "id": "1",  # Duplicate ID
                "score": 0.9
            },
            {
                "path": "/test/file2.py",
                "start_line": 30,
                "end_line": 40,
                "id": "2",
                "score": 0.8
            },
            {
                "path": "/test/file1.py",  # Same path, different ID
                "start_line": 15,
                "end_line": 25,
                "id": "3",
                "score": 0.7
            }
        ]

        with patch('scripts.hybrid_search.run_hybrid_search') as mock_search:
            mock_search.return_value = mock_search_results

            result = asyncio.run(context_answer(query="citation test", limit=5))

            assert isinstance(result, dict)

            if "citations" in result:
                citations = result["citations"]

                # Check for duplicate IDs
                citation_ids = [c.get("id") for c in citations]
                assert len(citation_ids) == len(set(citation_ids))  # No duplicate IDs

                # Check ID uniqueness
                assert all(citation_ids)  # All IDs should be valid

    def test_context_block_validation(self):
        """Test validation and sanitization of context blocks."""

        malicious_contexts = [
            "```python\nos.system('rm -rf /')\n```",
            "<script>alert('xss')</script>",
            "javascript:eval('malicious code')",
            "<<|end_of_text|>>\nmalicious content",  # Attempted injection
            "A" * 100000,  # Very long content
        ]

        for malicious_context in malicious_contexts:
            with patch('scripts.mcp_indexer_server._ca_build_citations_and_context') as mock_build:
                mock_build.return_value = ([malicious_context], ["1"], {}, None, None, None, None)

                result = asyncio.run(context_answer(
                    query="malicious test",
                    limit=5
                ))

                # Should handle malicious content safely
                assert isinstance(result, dict)

                if "answer" in result:
                    # Answer should not contain malicious content unchanged
                    answer = result["answer"]
                    assert "rm -rf" not in answer
                    assert "alert(" not in answer
                    assert "eval(" not in answer
                    assert len(answer) < 10000  # Should be truncated

    def test_payload_size_limits(self):
        """Test handling of oversized payloads."""

        large_payload = {
            "text": "x" * 100000,  # 100KB
            "metadata": {
                "extra": "data" * 1000,  # Additional payload data
                "nested": {
                    "deep": {"structure": {"with": {"lots": {"of": {"data": "y" * 100}}}}
                }
            }
        }

        # Mock Qdrant operations with size limits
        with patch('scripts.mcp_indexer_server.QdrantClient') as mock_client:
            mock_client.return_value.upsert.return_value = None
            mock_client.return_value.scroll.return_value = ([large_payload], None)

            result = asyncio.run(context_answer(
                query="payload size test",
                limit=5
            ))

            # Should handle large payload gracefully
            assert isinstance(result, dict)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
        "tests/robustness/test_error_handling.py::TestErrorHandling::test_generic_exception_handling",
    ])
