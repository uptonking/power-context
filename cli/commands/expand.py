"""Query expansion command."""
from __future__ import annotations

import argparse

from cli.core import output_json, run_async


def cmd_expand_query(args: argparse.Namespace) -> None:
    """Generate query variations for better recall."""
    from scripts.mcp_impl.query_expand import _expand_query_impl

    result = run_async(_expand_query_impl(
        query=args.query,
        max_new=getattr(args, "max_new", 3),
    ))
    output_json(result)
