"""
TOON (Token-Oriented Object Notation) encoder for Context-Engine.

A compact, token-efficient format for LLM input that combines:
- YAML-like indentation for nested objects
- CSV-style tabular layout for uniform arrays

Feature flags:
- TOON_ENABLED=1          Enable TOON encoding (default: 0)
- TOON_DELIMITER=","      Field delimiter (default: ",", use "\t" for even fewer tokens)
- TOON_INCLUDE_LENGTH=1   Include [N] length markers (default: 1)

Reference: https://github.com/toon-format/toon
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union, Callable

# -----------------------------------------------------------------------------
# Feature Flags
# -----------------------------------------------------------------------------

def is_toon_enabled() -> bool:
    """Check if TOON encoding is enabled via environment variable."""
    return os.environ.get("TOON_ENABLED", "0").lower() in ("1", "true", "yes")


def get_toon_delimiter() -> str:
    """Get the field delimiter for TOON encoding."""
    delim = os.environ.get("TOON_DELIMITER", ",")
    # Handle escaped tab
    if delim in ("\\t", "tab", "TAB"):
        return "\t"
    return delim


def include_length_markers() -> bool:
    """Check if [N] length markers should be included."""
    return os.environ.get("TOON_INCLUDE_LENGTH", "1").lower() in ("1", "true", "yes")


# -----------------------------------------------------------------------------
# Value Encoding
# -----------------------------------------------------------------------------

def _needs_quoting(value: str, delimiter: str) -> bool:
    """Check if a string value needs quoting."""
    if not value:
        return False
    # Quote if contains delimiter, newline, or starts/ends with whitespace
    return (
        delimiter in value
        or "\n" in value
        or "\r" in value
        or value[0].isspace()
        or value[-1].isspace()
        or value.startswith('"')
    )


def _encode_value(value: Any, delimiter: str) -> str:
    """Encode a single value to TOON format."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if _needs_quoting(value, delimiter):
            # Escape quotes and wrap
            escaped = value.replace('"', '""')
            return f'"{escaped}"'
        return value
    # For nested objects/arrays in a table row, use compact JSON
    import json
    return json.dumps(value, separators=(",", ":"))


# -----------------------------------------------------------------------------
# Array Detection
# -----------------------------------------------------------------------------

def _is_uniform_array_of_objects(arr: List[Any]) -> bool:
    """Check if array is uniform objects (same keys) - suitable for tabular format."""
    if not arr or len(arr) < 1:
        return False
    if not all(isinstance(item, dict) for item in arr):
        return False
    # Check all have same keys
    first_keys = set(arr[0].keys())
    return all(set(item.keys()) == first_keys for item in arr)


def _get_field_order(arr: List[Dict[str, Any]]) -> List[str]:
    """Get consistent field order from first object."""
    if not arr:
        return []
    return list(arr[0].keys())


# -----------------------------------------------------------------------------
# TOON Encoding
# -----------------------------------------------------------------------------

def encode_tabular(
    key: str,
    arr: List[Dict[str, Any]],
    delimiter: str = ",",
    indent: int = 0,
    include_length: bool = True,
) -> List[str]:
    """Encode a uniform array of objects as TOON tabular format.
    
    Example output:
        users[3]{id,name,role}:
          1,Alice,admin
          2,Bob,user
          3,Carol,viewer
    """
    if not arr:
        length_part = "[0]" if include_length else ""
        return [f"{'  ' * indent}{key}{length_part}: []"]
    
    fields = _get_field_order(arr)
    length_part = f"[{len(arr)}]" if include_length else ""
    fields_part = "{" + delimiter.join(fields) + "}"
    
    lines = [f"{'  ' * indent}{key}{length_part}{fields_part}:"]
    
    row_indent = "  " * (indent + 1)
    for item in arr:
        values = [_encode_value(item.get(f), delimiter) for f in fields]
        lines.append(f"{row_indent}{delimiter.join(values)}")
    
    return lines


def encode_simple_array(
    key: str,
    arr: List[Any],
    delimiter: str = ",",
    indent: int = 0,
    include_length: bool = True,
) -> List[str]:
    """Encode a simple (non-object) array inline.

    Example output:
        tags[3]: python,async,api
        empty[0]: []
    """
    length_part = f"[{len(arr)}]" if include_length else ""
    prefix = "  " * indent
    # Handle empty arrays with explicit [] marker
    if not arr:
        return [f"{prefix}{key}{length_part}: []"]
    values = [_encode_value(v, delimiter) for v in arr]
    return [f"{prefix}{key}{length_part}: {delimiter.join(values)}"]


def encode_object(
    obj: Dict[str, Any],
    delimiter: str = ",",
    indent: int = 0,
    include_length: bool = True,
) -> List[str]:
    """Encode an object to TOON format.

    - Simple key-value pairs become YAML-like lines
    - Uniform arrays of objects become tabular
    - Other arrays become inline
    """
    lines: List[str] = []
    prefix = "  " * indent

    for key, value in obj.items():
        if value is None:
            lines.append(f"{prefix}{key}:")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{prefix}{key}: {value}")
        elif isinstance(value, str):
            # Multi-line strings get block format
            if "\n" in value:
                lines.append(f"{prefix}{key}: |")
                for line in value.split("\n"):
                    lines.append(f"{prefix}  {line}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        elif isinstance(value, list):
            if _is_uniform_array_of_objects(value):
                lines.extend(encode_tabular(key, value, delimiter, indent, include_length))
            elif value and all(isinstance(v, dict) for v in value):
                # Non-uniform objects - encode each separately
                length_part = f"[{len(value)}]" if include_length else ""
                lines.append(f"{prefix}{key}{length_part}:")
                for item in value:
                    lines.append(f"{prefix}  -")
                    lines.extend(encode_object(item, delimiter, indent + 2, include_length))
            else:
                lines.extend(encode_simple_array(key, value, delimiter, indent, include_length))
        elif isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(encode_object(value, delimiter, indent + 1, include_length))
        else:
            # Fallback: stringify
            lines.append(f"{prefix}{key}: {value}")

    return lines


def encode(
    data: Any,
    delimiter: Optional[str] = None,
    include_length: Optional[bool] = None,
) -> str:
    """Encode any JSON-compatible data to TOON format.

    Args:
        data: JSON-compatible data (dict, list, or primitive)
        delimiter: Field delimiter (default from env or ",")
        include_length: Include [N] markers (default from env or True)

    Returns:
        TOON-formatted string
    """
    if delimiter is None:
        delimiter = get_toon_delimiter()
    if include_length is None:
        include_length = include_length_markers()

    if isinstance(data, dict):
        lines = encode_object(data, delimiter, 0, include_length)
    elif isinstance(data, list):
        if _is_uniform_array_of_objects(data):
            lines = encode_tabular("data", data, delimiter, 0, include_length)
        else:
            lines = encode_simple_array("data", data, delimiter, 0, include_length)
    else:
        # Primitive
        return str(data)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Search Results Formatting (Context-Engine specific)
# -----------------------------------------------------------------------------

def encode_search_results(
    results: List[Dict[str, Any]],
    delimiter: Optional[str] = None,
    include_length: Optional[bool] = None,
    compact: bool = True,
) -> str:
    """Encode search results to TOON tabular format.

    Dynamically includes all fields present in results to avoid dropping data.
    Core fields are ordered first, then any additional fields alphabetically.

    Args:
        results: List of search result dicts
        delimiter: Field delimiter
        include_length: Include [N] markers
        compact: If True, only include core location fields (path/lines)

    Returns:
        TOON-formatted search results
    """
    if delimiter is None:
        delimiter = get_toon_delimiter()
    if include_length is None:
        include_length = include_length_markers()

    length_part = f"[{len(results)}]" if include_length else ""

    if not results:
        return f"results{length_part}: []"

    # Determine fields based on compact mode
    if compact:
        fields = ["path", "start_line", "end_line"]
    else:
        # Collect all unique fields from all results
        all_fields: set = set()
        for r in results:
            all_fields.update(r.keys())

        # Core fields ordered first, then remaining alphabetically
        core_order = ["path", "start_line", "end_line", "score", "symbol", "kind",
                      "snippet", "information", "relevance_score", "why"]
        fields = [f for f in core_order if f in all_fields]
        # Add remaining fields not in core_order
        extra_fields = sorted(all_fields - set(core_order))
        fields.extend(extra_fields)

    # Build tabular output
    fields_part = "{" + delimiter.join(fields) + "}"

    lines = [f"results{length_part}{fields_part}:"]
    for r in results:
        values = [_encode_value(r.get(f), delimiter) for f in fields]
        lines.append(f"  {delimiter.join(values)}")

    return "\n".join(lines)


def encode_context_results(
    results: List[Dict[str, Any]],
    delimiter: Optional[str] = None,
    include_length: Optional[bool] = None,
    compact: bool = True,
) -> str:
    """Encode context_search results (code + memory) to TOON format.

    Dynamically includes all fields present to avoid dropping data.
    Handles mixed result types with source-aware encoding.

    Args:
        results: List of mixed search result dicts
        delimiter: Field delimiter
        include_length: Include [N] markers
        compact: If True, use minimal core fields only

    Returns:
        TOON-formatted context results with source-aware encoding
    """
    if delimiter is None:
        delimiter = get_toon_delimiter()
    if include_length is None:
        include_length = include_length_markers()

    length_part = f"[{len(results)}]" if include_length else ""

    if not results:
        return f"results{length_part}: []"

    # Separate code and memory results
    code_results = [r for r in results if r.get("source") != "memory"]
    memory_results = [r for r in results if r.get("source") == "memory"]

    lines = []

    # Encode code results
    if code_results:
        code_len = f"[{len(code_results)}]" if include_length else ""
        if compact:
            code_fields = ["path", "start_line", "end_line"]
        else:
            # Collect all fields from code results
            all_code_fields: set = set()
            for r in code_results:
                all_code_fields.update(r.keys())
            # Core order, then extras
            core_order = ["path", "start_line", "end_line", "score", "symbol", "kind",
                          "snippet", "information", "relevance_score", "why"]
            code_fields = [f for f in core_order if f in all_code_fields]
            extra = sorted(all_code_fields - set(core_order) - {"source"})
            code_fields.extend(extra)
        code_fields_part = "{" + delimiter.join(code_fields) + "}"
        lines.append(f"code{code_len}{code_fields_part}:")
        for r in code_results:
            values = [_encode_value(r.get(f), delimiter) for f in code_fields]
            lines.append(f"  {delimiter.join(values)}")

    # Encode memory results
    if memory_results:
        mem_len = f"[{len(memory_results)}]" if include_length else ""
        if compact:
            mem_fields = ["content", "score"]
        else:
            # Collect all fields from memory results
            all_mem_fields: set = set()
            for r in memory_results:
                all_mem_fields.update(r.keys())
            core_order = ["content", "score", "id", "information"]
            mem_fields = [f for f in core_order if f in all_mem_fields]
            extra = sorted(all_mem_fields - set(core_order) - {"source"})
            mem_fields.extend(extra)
        mem_fields_part = "{" + delimiter.join(mem_fields) + "}"
        lines.append(f"memory{mem_len}{mem_fields_part}:")
        for r in memory_results:
            values = [_encode_value(r.get(f), delimiter) for f in mem_fields]
            lines.append(f"  {delimiter.join(values)}")

    # If no separation needed (all same type or empty), use unified format
    if not lines:
        # Fallback: use generic encode
        return encode_search_results(results, delimiter, include_length, compact)

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Token Estimation
# -----------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4 heuristic, good enough for comparison)."""
    return len(text) // 4


def compare_formats(data: Any) -> Dict[str, Any]:
    """Compare TOON vs JSON token counts for given data.

    Returns:
        {
            "json_tokens": int,
            "json_compact_tokens": int,
            "toon_tokens": int,
            "toon_tab_tokens": int,
            "savings_vs_json": float,  # percentage
            "savings_vs_compact": float,
        }
    """
    import json

    json_pretty = json.dumps(data, indent=2)
    json_compact = json.dumps(data, separators=(",", ":"))
    toon_comma = encode(data, delimiter=",")
    toon_tab = encode(data, delimiter="\t")

    json_tokens = estimate_tokens(json_pretty)
    json_compact_tokens = estimate_tokens(json_compact)
    toon_tokens = estimate_tokens(toon_comma)
    toon_tab_tokens = estimate_tokens(toon_tab)

    return {
        "json_tokens": json_tokens,
        "json_compact_tokens": json_compact_tokens,
        "toon_tokens": toon_tokens,
        "toon_tab_tokens": toon_tab_tokens,
        "savings_vs_json": round((1 - toon_tokens / json_tokens) * 100, 1) if json_tokens else 0,
        "savings_vs_compact": round((1 - toon_tokens / json_compact_tokens) * 100, 1) if json_compact_tokens else 0,
    }

