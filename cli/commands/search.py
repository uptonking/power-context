"""Search commands: search, search-tests, search-config, search-callers, search-importers."""
from __future__ import annotations

import argparse
import os

from cli.core import (
    output_json,
    repo_search_async,
    resolve_collection,
    run_async,
)


def cmd_search(args: argparse.Namespace) -> None:
    """Hybrid code search (dense + lexical + rerank).

    Uses _repo_search_impl (same as MCP server) to get the full pipeline:
    reranking, code signal detection, filename boost, snippet inclusion, etc.
    """
    collection = resolve_collection(getattr(args, "collection", None))
    old_expand = None
    if getattr(args, "no_expand", False):
        old_expand = os.environ.get("HYBRID_EXPAND")
        os.environ["HYBRID_EXPAND"] = "0"
    try:
        result = run_async(repo_search_async(
            query=args.query,
            limit=args.limit,
            per_path=args.per_path,
            include_snippet=getattr(args, "include_snippet", False),
            context_lines=getattr(args, "context_lines", 2),
            language=getattr(args, "language", None),
            under=getattr(args, "under", None),
            kind=getattr(args, "kind", None),
            symbol=getattr(args, "symbol", None),
            ext=getattr(args, "ext", None),
            not_=getattr(args, "not_filter", None),
            case=getattr(args, "case", None),
            path_regex=getattr(args, "path_regex", None),
            path_glob=getattr(args, "path_glob", None),
            not_glob=getattr(args, "not_glob", None),
            compact=getattr(args, "compact", False),
            collection=collection,
            repo=getattr(args, "repo", None),
        ))
    finally:
        if getattr(args, "no_expand", False):
            if old_expand is None:
                os.environ.pop("HYBRID_EXPAND", None)
            else:
                os.environ["HYBRID_EXPAND"] = old_expand
    output_json(result)


def cmd_search_tests(args: argparse.Namespace) -> None:
    """Find test files related to a query."""
    from scripts.mcp_impl.search_specialized import _search_tests_for_impl

    collection = resolve_collection(getattr(args, "collection", None))
    result = run_async(_search_tests_for_impl(
        query=args.query,
        limit=args.limit,
        include_snippet=getattr(args, "include_snippet", False),
        context_lines=getattr(args, "context_lines", 2),
        under=getattr(args, "under", None),
        language=getattr(args, "language", None),
        compact=getattr(args, "compact", False),
        collection=collection,
        repo_search_fn=repo_search_async,
    ))
    output_json(result)


def cmd_search_config(args: argparse.Namespace) -> None:
    """Find config files related to a query."""
    from scripts.mcp_impl.search_specialized import _search_config_for_impl

    collection = resolve_collection(getattr(args, "collection", None))
    result = run_async(_search_config_for_impl(
        query=args.query,
        limit=args.limit,
        include_snippet=getattr(args, "include_snippet", False),
        context_lines=getattr(args, "context_lines", 2),
        under=getattr(args, "under", None),
        compact=getattr(args, "compact", False),
        collection=collection,
        repo_search_fn=repo_search_async,
    ))
    output_json(result)


def cmd_search_callers(args: argparse.Namespace) -> None:
    """Find callers of a symbol."""
    from scripts.mcp_impl.search_specialized import _search_callers_for_impl

    collection = resolve_collection(getattr(args, "collection", None))
    result = run_async(_search_callers_for_impl(
        query=args.query,
        limit=args.limit,
        language=getattr(args, "language", None),
        collection=collection,
        repo_search_fn=repo_search_async,
    ))
    output_json(result)


def cmd_search_importers(args: argparse.Namespace) -> None:
    """Find importers of a module."""
    from scripts.mcp_impl.search_specialized import _search_importers_for_impl

    collection = resolve_collection(getattr(args, "collection", None))
    result = run_async(_search_importers_for_impl(
        query=args.query,
        limit=args.limit,
        language=getattr(args, "language", None),
        collection=collection,
        repo_search_fn=repo_search_async,
    ))
    output_json(result)
