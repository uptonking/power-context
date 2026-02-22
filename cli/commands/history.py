"""Git history commands: search-commits, change-history."""
from __future__ import annotations

import argparse

from cli.core import output_json, resolve_collection, run_async


def cmd_search_commits(args: argparse.Namespace) -> None:
    """Search git commit history indexed in Qdrant."""
    from scripts.mcp_impl.search_history import _search_commits_for_impl
    from scripts.mcp_impl.admin_tools import _get_embedding_model
    from scripts.mcp_impl.workspace import _default_collection

    result = run_async(_search_commits_for_impl(
        query=args.query,
        path=getattr(args, "path", None),
        collection=resolve_collection(getattr(args, "collection", None)),
        limit=args.limit,
        default_collection_fn=_default_collection,
        get_embedding_model_fn=_get_embedding_model,
    ))
    output_json(result)


def cmd_change_history(args: argparse.Namespace) -> None:
    """Get change history for a file path."""
    from scripts.mcp_impl.search_history import (
        _change_history_for_path_impl,
        _search_commits_for_impl,
    )
    from scripts.mcp_impl.workspace import _default_collection

    result = run_async(_change_history_for_path_impl(
        path=args.path,
        collection=resolve_collection(getattr(args, "collection", None)),
        include_commits=getattr(args, "include_commits", False),
        default_collection_fn=_default_collection,
        search_commits_fn=_search_commits_for_impl,
    ))
    output_json(result)
