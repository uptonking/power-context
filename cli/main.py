"""CLI entry point — argparse dispatcher for all subcommands."""
from __future__ import annotations

import argparse
import json
import sys
import traceback


# ---------------------------------------------------------------------------
# Command registry: command name → (module_path, function_name)
# Lazy-imported at dispatch time to keep startup fast.
# ---------------------------------------------------------------------------
COMMANDS = {
    "search":           ("cli.commands.search",  "cmd_search"),
    "search-tests":     ("cli.commands.search",  "cmd_search_tests"),
    "search-config":    ("cli.commands.search",  "cmd_search_config"),
    "search-callers":   ("cli.commands.search",  "cmd_search_callers"),
    "search-importers": ("cli.commands.search",  "cmd_search_importers"),
    "symbol-graph":     ("cli.commands.symbol",  "cmd_symbol_graph"),
    "search-commits":   ("cli.commands.history", "cmd_search_commits"),
    "change-history":   ("cli.commands.history", "cmd_change_history"),
    "pattern-search":   ("cli.commands.pattern", "cmd_pattern_search"),
    "expand-query":     ("cli.commands.expand",  "cmd_expand_query"),
    "index":            ("cli.commands.index",   "cmd_index"),
    "prune":            ("cli.commands.index",   "cmd_prune"),
    "status":           ("cli.commands.index",   "cmd_status"),
    "list-collections": ("cli.commands.index",   "cmd_list_collections"),
    "watch":            ("cli.commands.watch",   "cmd_watch"),
}


# ---------------------------------------------------------------------------
# Shared argparse helpers
# ---------------------------------------------------------------------------
def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("query", help="Search query (text)")
    p.add_argument("-l", "--limit", type=int, default=10, help="Max results")
    p.add_argument("-c", "--collection", help="Qdrant collection name")


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--language", help="Filter by language")
    p.add_argument("--under", help="Path prefix filter")
    p.add_argument("--kind", help="AST node type filter")
    p.add_argument("--symbol", help="Symbol name filter")
    p.add_argument("--ext", help="File extension filter")
    p.add_argument("--path-regex", help="Path regex filter")
    p.add_argument("--path-glob", nargs="+", help="Include path globs")
    p.add_argument("--not-glob", nargs="+", help="Exclude path globs")
    p.add_argument("--not-filter", help="Negative text filter")
    p.add_argument("--case", help="Case sensitivity")
    p.add_argument("--repo", nargs="+", help="Repo name filter")
    p.add_argument("--per-path", type=int, default=2, help="Max results per file")
    p.add_argument("--no-expand", action="store_true", help="Disable query expansion")


def _add_snippet_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--include-snippet", action="store_true", help="Include code snippets")
    p.add_argument("--context-lines", type=int, default=2, help="Context lines around matches")
    p.add_argument("--compact", action="store_true", help="Compact output")


# ---------------------------------------------------------------------------
# Parser builder
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    from cli._version import __version__

    parser = argparse.ArgumentParser(
        prog="power-context",
        description="Context Engine CLI — programmatic code search and indexing",
    )
    parser.add_argument("--debug", action="store_true", help="Show stack traces on error")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # search
    p = sub.add_parser("search", help="Hybrid code search (dense + lexical + rerank)")
    _add_common_args(p)
    _add_filter_args(p)
    _add_snippet_args(p)

    # search-tests
    p = sub.add_parser("search-tests", help="Find test files")
    _add_common_args(p)
    _add_snippet_args(p)
    p.add_argument("--language", help="Filter by language")
    p.add_argument("--under", help="Path prefix filter")

    # search-config
    p = sub.add_parser("search-config", help="Find config files")
    _add_common_args(p)
    _add_snippet_args(p)
    p.add_argument("--under", help="Path prefix filter")

    # search-callers
    p = sub.add_parser("search-callers", help="Find callers of a symbol")
    _add_common_args(p)
    p.add_argument("--language", help="Filter by language")

    # search-importers
    p = sub.add_parser("search-importers", help="Find importers of a module")
    _add_common_args(p)
    p.add_argument("--language", help="Filter by language")

    # symbol-graph
    p = sub.add_parser("symbol-graph", help="Symbol graph navigation")
    p.add_argument("symbol", help="Symbol name to look up")
    p.add_argument("-t", "--query-type", default="callers",
                   choices=["callers", "definition", "importers", "called_by"],
                   help="Relationship type")
    p.add_argument("-l", "--limit", type=int, default=10)
    p.add_argument("-c", "--collection", help="Qdrant collection")
    p.add_argument("--language", help="Filter by language")
    p.add_argument("--under", help="Path prefix filter")

    # search-commits
    p = sub.add_parser("search-commits", help="Search git commit history")
    _add_common_args(p)
    p.add_argument("--path", help="Filter by file path")

    # change-history
    p = sub.add_parser("change-history", help="File change history")
    p.add_argument("path", help="File path to inspect")
    p.add_argument("-c", "--collection", help="Qdrant collection")
    p.add_argument("--include-commits", action="store_true", help="Include commit details")

    # pattern-search
    p = sub.add_parser("pattern-search", help="Structural pattern matching")
    _add_common_args(p)
    p.add_argument("--language", help="Language hint")
    p.add_argument("--min-score", type=float, default=0.3, help="Min similarity")
    p.add_argument("--query-mode", choices=["code", "description", "auto"], default="auto")
    _add_snippet_args(p)

    # expand-query
    p = sub.add_parser("expand-query", help="Generate query variations")
    p.add_argument("query", help="Query to expand")
    p.add_argument("--max-new", type=int, default=3, help="Max new variations")

    # index
    p = sub.add_parser("index", help="Index a directory into Qdrant")
    p.add_argument("path", nargs="?", default=".", help="Root path to index")
    p.add_argument("--subdir", help="Only index a subdirectory")
    p.add_argument("-c", "--collection", help="Qdrant collection")
    p.add_argument("--recreate", action="store_true", help="Drop and recreate collection")

    # prune
    p = sub.add_parser("prune", help="Remove stale points from index")
    p.add_argument("path", nargs="?", default=".", help="Root path for file existence checks")
    p.add_argument("-c", "--collection", help="Qdrant collection")

    # status
    p = sub.add_parser("status", help="Collection status and health")
    p.add_argument("-c", "--collection", help="Qdrant collection")

    # list-collections
    sub.add_parser("list-collections", help="List all Qdrant collections")

    # watch
    p = sub.add_parser("watch", help="Auto-reindex on file changes (daemon)")
    p.add_argument("path", nargs="?", default=".", help="Root path to watch")
    p.add_argument("-c", "--collection", help="Qdrant collection")

    return parser


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    debug = False
    if "--" in argv:
        sep = argv.index("--")
        before = argv[:sep]
        after = argv[sep + 1 :]
    else:
        before = argv
        after = []
    if "--debug" in before:
        debug = True
        before = [arg for arg in before if arg != "--debug"]
    argv = before + (["--"] + after if after else [])
    args = parser.parse_args(argv)
    args.debug = bool(debug or getattr(args, "debug", False))

    entry = COMMANDS.get(args.command)
    if not entry:
        parser.print_help()
        sys.exit(1)

    mod_path, fn_name = entry
    try:
        import importlib
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, fn_name)
        fn(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        json.dump({"ok": False, "error": str(exc)}, sys.stdout, default=str)
        sys.stdout.write("\n")
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
