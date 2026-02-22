"""Pattern search command: structural code pattern matching."""
from __future__ import annotations

import argparse

from cli.core import output_json, resolve_collection, run_async


def cmd_pattern_search(args: argparse.Namespace) -> None:
    """Find structurally similar code patterns (code example or description)."""
    from scripts.mcp_impl.pattern_search import _pattern_search_impl

    result = run_async(_pattern_search_impl(
        query=args.query,
        language=getattr(args, "language", None),
        limit=args.limit,
        min_score=getattr(args, "min_score", 0.3),
        include_snippet=getattr(args, "include_snippet", False),
        context_lines=getattr(args, "context_lines", 2),
        query_mode=getattr(args, "query_mode", "auto"),
        collection=resolve_collection(getattr(args, "collection", None)),
    ))
    output_json(result)
