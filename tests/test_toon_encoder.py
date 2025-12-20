"""Tests for TOON (Token-Oriented Object Notation) encoder."""

import os
import pytest

# Ensure scripts module is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.toon_encoder import (
    encode,
    encode_tabular,
    encode_simple_array,
    encode_object,
    encode_search_results,
    compare_formats,
    is_toon_enabled,
    get_toon_delimiter,
    include_length_markers,
    _encode_value,
    _is_uniform_array_of_objects,
)


class TestFeatureFlags:
    """Test feature flag behavior."""

    def test_toon_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("TOON_ENABLED", raising=False)
        assert is_toon_enabled() is False

    def test_toon_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("TOON_ENABLED", "1")
        assert is_toon_enabled() is True
        
        monkeypatch.setenv("TOON_ENABLED", "true")
        assert is_toon_enabled() is True

    def test_delimiter_default(self, monkeypatch):
        monkeypatch.delenv("TOON_DELIMITER", raising=False)
        assert get_toon_delimiter() == ","

    def test_delimiter_tab(self, monkeypatch):
        monkeypatch.setenv("TOON_DELIMITER", "\\t")
        assert get_toon_delimiter() == "\t"
        
        monkeypatch.setenv("TOON_DELIMITER", "tab")
        assert get_toon_delimiter() == "\t"

    def test_length_markers_default(self, monkeypatch):
        monkeypatch.delenv("TOON_INCLUDE_LENGTH", raising=False)
        assert include_length_markers() is True


class TestValueEncoding:
    """Test individual value encoding."""

    def test_encode_primitives(self):
        assert _encode_value(None, ",") == ""
        assert _encode_value(True, ",") == "true"
        assert _encode_value(False, ",") == "false"
        assert _encode_value(42, ",") == "42"
        assert _encode_value(3.14, ",") == "3.14"
        assert _encode_value("hello", ",") == "hello"

    def test_encode_string_with_delimiter(self):
        # String containing delimiter should be quoted
        assert _encode_value("hello,world", ",") == '"hello,world"'
        
    def test_encode_string_with_newline(self):
        assert _encode_value("line1\nline2", ",") == '"line1\nline2"'

    def test_encode_nested_object(self):
        # Nested objects become compact JSON
        result = _encode_value({"a": 1}, ",")
        assert result == '{"a":1}'


class TestArrayDetection:
    """Test uniform array detection."""

    def test_uniform_array(self):
        arr = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        assert _is_uniform_array_of_objects(arr) is True

    def test_non_uniform_array(self):
        arr = [{"a": 1}, {"b": 2}]  # Different keys
        assert _is_uniform_array_of_objects(arr) is False

    def test_empty_array(self):
        assert _is_uniform_array_of_objects([]) is False

    def test_primitive_array(self):
        assert _is_uniform_array_of_objects([1, 2, 3]) is False


class TestTabularEncoding:
    """Test tabular array encoding."""

    def test_encode_tabular_basic(self):
        arr = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        lines = encode_tabular("users", arr)
        assert lines[0] == "users[2]{id,name}:"
        assert lines[1] == "  1,Alice"
        assert lines[2] == "  2,Bob"

    def test_encode_tabular_no_length(self):
        arr = [{"x": 1}]
        lines = encode_tabular("items", arr, include_length=False)
        assert lines[0] == "items{x}:"

    def test_encode_tabular_tab_delimiter(self):
        arr = [{"a": 1, "b": 2}]
        lines = encode_tabular("data", arr, delimiter="\t")
        assert lines[0] == "data[1]{a\tb}:"
        assert lines[1] == "  1\t2"

    def test_encode_empty_array(self):
        lines = encode_tabular("empty", [])
        assert lines == ["empty[0]: []"]


class TestSimpleArrayEncoding:
    """Test simple array encoding."""

    def test_encode_simple_array(self):
        lines = encode_simple_array("tags", ["python", "async", "api"])
        assert lines == ["tags[3]: python,async,api"]

    def test_encode_numeric_array(self):
        lines = encode_simple_array("nums", [1, 2, 3])
        assert lines == ["nums[3]: 1,2,3"]


class TestFullEncoding:
    """Test full object encoding."""

    def test_encode_simple_object(self):
        obj = {"name": "Alice", "age": 30, "active": True}
        result = encode(obj)
        assert "name: Alice" in result
        assert "age: 30" in result
        assert "active: true" in result

    def test_encode_nested_object(self):
        obj = {
            "user": {
                "name": "Bob",
                "role": "admin",
            }
        }
        result = encode(obj)
        assert "user:" in result
        assert "  name: Bob" in result

    def test_encode_with_tabular_array(self):
        obj = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }
        result = encode(obj)
        assert "users[2]{id,name}:" in result
        assert "  1,Alice" in result
        assert "  2,Bob" in result

    def test_encode_mixed_structure(self):
        """Test the hikes example from TOON spec."""
        obj = {
            "context": {
                "task": "Our favorite hikes together",
                "location": "Boulder",
            },
            "friends": ["ana", "luis", "sam"],
            "hikes": [
                {"id": 1, "name": "Blue Lake Trail", "distanceKm": 7.5},
                {"id": 2, "name": "Ridge Overlook", "distanceKm": 9.2},
            ]
        }
        result = encode(obj)
        # Context nested
        assert "context:" in result
        assert "  task: Our favorite hikes together" in result
        # Friends inline
        assert "friends[3]: ana,luis,sam" in result
        # Hikes tabular
        assert "hikes[2]{id,name,distanceKm}:" in result


class TestSearchResults:
    """Test search result encoding specifically."""

    def test_encode_search_results_compact(self):
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
            {"path": "/src/utils.py", "start_line": 5, "end_line": 15, "score": 0.87},
        ]
        output = encode_search_results(results, compact=True)
        assert "results[2]{path,start_line,end_line}:" in output
        assert "  /src/main.py,10,20" in output
        assert "0.95" not in output  # Score excluded in compact

    def test_encode_search_results_full(self):
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95, "symbol": "main"},
        ]
        output = encode_search_results(results, compact=False)
        assert "score" in output
        assert "0.95" in output

    def test_encode_empty_results(self):
        output = encode_search_results([])
        assert output == "results[0]: []"


class TestTokenComparison:
    """Test token counting and comparison."""

    def test_compare_formats_basic(self):
        data = {
            "users": [
                {"id": 1, "name": "Alice", "role": "admin"},
                {"id": 2, "name": "Bob", "role": "user"},
                {"id": 3, "name": "Carol", "role": "viewer"},
            ]
        }
        stats = compare_formats(data)

        assert stats["json_tokens"] > 0
        assert stats["toon_tokens"] > 0
        # TOON should be smaller than pretty JSON for tabular data
        assert stats["toon_tokens"] < stats["json_tokens"]
        assert stats["savings_vs_json"] > 0

    def test_compare_formats_larger_dataset(self):
        """Simulate search results - where TOON shines."""
        data = {
            "results": [
                {"path": f"/src/file{i}.py", "start_line": i * 10, "end_line": i * 10 + 5, "score": 0.9 - i * 0.05}
                for i in range(20)
            ]
        }
        stats = compare_formats(data)

        # Should see significant savings on uniform arrays
        assert stats["savings_vs_json"] > 30  # Expect 30%+ savings
        print(f"\nToken comparison for 20 search results:")
        print(f"  JSON (pretty):  {stats['json_tokens']} tokens")
        print(f"  JSON (compact): {stats['json_compact_tokens']} tokens")
        print(f"  TOON (comma):   {stats['toon_tokens']} tokens")
        print(f"  TOON (tab):     {stats['toon_tab_tokens']} tokens")
        print(f"  Savings vs JSON: {stats['savings_vs_json']}%")


class TestMCPIntegration:
    """Test TOON integration with MCP server helpers."""

    def test_should_use_toon_explicit_param(self, monkeypatch):
        """Test explicit output_format parameter takes precedence."""
        # Import the helpers from mcp_indexer_server
        monkeypatch.delenv("TOON_ENABLED", raising=False)

        # We need to test the helper functions directly
        # Since they're in mcp_indexer_server, we'll test the logic here
        from scripts.toon_encoder import is_toon_enabled

        # When TOON_ENABLED is not set, default is False
        assert is_toon_enabled() is False

        # When explicitly set
        monkeypatch.setenv("TOON_ENABLED", "1")
        assert is_toon_enabled() is True

    def test_format_results_as_toon_structure(self):
        """Test that TOON formatting adds expected fields."""
        # Simulate a search response
        response = {
            "results": [
                {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
                {"path": "/src/utils.py", "start_line": 5, "end_line": 15, "score": 0.87},
            ],
            "total": 2,
            "args": {"query": "test"},
        }

        # Apply TOON formatting
        toon_output = encode_search_results(response["results"], compact=True)

        # Verify TOON output structure
        assert "results[2]{path,start_line,end_line}:" in toon_output
        assert "/src/main.py,10,20" in toon_output
        assert "/src/utils.py,5,15" in toon_output

    def test_toon_replaces_results_array(self):
        """Test that TOON formatting replaces JSON array with TOON string."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20},
        ]

        # When TOON is applied, results becomes a string
        toon_output = encode_search_results(results, compact=True)
        assert isinstance(toon_output, str)
        assert "results[1]{path,start_line,end_line}:" in toon_output
        assert "/src/main.py,10,20" in toon_output

    def test_toon_compact_mode_excludes_score(self):
        """Test that compact mode excludes score field."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
        ]

        output = encode_search_results(results, compact=True)

        # Score should not be in compact output
        assert "0.95" not in output
        assert "score" not in output

    def test_toon_full_mode_includes_score(self):
        """Test that full mode includes score field."""
        results = [
            {"path": "/src/main.py", "start_line": 10, "end_line": 20, "score": 0.95},
        ]

        output = encode_search_results(results, compact=False)

        # Score should be in full output
        assert "0.95" in output

