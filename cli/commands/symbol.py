"""Symbol graph command: callers, definition, importers navigation."""
from __future__ import annotations

import argparse

from cli.core import output_json, resolve_collection, run_async


def cmd_symbol_graph(args: argparse.Namespace) -> None:
    """Navigate symbol relationships (callers/definition/importers)."""
    from scripts.mcp_impl.symbol_graph import _symbol_graph_impl

    result = run_async(_symbol_graph_impl(
        symbol=args.symbol,
        query_type=args.query_type,
        limit=args.limit,
        language=getattr(args, "language", None),
        under=getattr(args, "under", None),
        collection=resolve_collection(getattr(args, "collection", None)),
    ))
    output_json(result)
