#!/usr/bin/env python3
"""
Test CLI commands correctness - verify CLI properly delegates to MCP implementations.

These tests verify that:
1. CLI commands correctly call the underlying _impl functions
2. Parameters are passed correctly
3. Collection resolution works properly
4. Dependency injection is wired up correctly
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path


class TestCLICommandDelegation:
    """Test that CLI commands correctly delegate to MCP implementation."""

    def test_search_command_calls_repo_search_impl(self):
        """Verify cmd_search calls _repo_search_impl with correct params."""
        from cli.commands.search import cmd_search

        # Mock args
        args = MagicMock()
        args.query = "test query"
        args.limit = 10
        args.per_path = 2
        args.collection = None
        args.no_expand = False
        args.include_snippet = False
        args.context_lines = 2
        args.language = "python"
        args.under = "src/"
        args.kind = None
        args.symbol = None
        args.ext = None
        args.not_filter = None
        args.case = None
        args.path_regex = None
        args.path_glob = None
        args.not_glob = None
        args.compact = False
        args.repo = None

        # Mock the implementation
        with patch('cli.commands.search.repo_search_async') as mock_search:
            mock_search.return_value = {"results": []}

            with patch('cli.commands.search.run_async') as mock_run:
                mock_run.return_value = {"results": []}

                cmd_search(args)

                # Verify repo_search_async was called
                assert mock_search.called

                # Get the call kwargs
                call_kwargs = mock_search.call_args[1]

                # Verify key parameters passed correctly
                assert call_kwargs['query'] == "test query"
                assert call_kwargs['limit'] == 10
                assert call_kwargs['per_path'] == 2
                assert call_kwargs['language'] == "python"
                assert call_kwargs['under'] == "src/"

    def test_search_tests_delegates_to_impl(self):
        """Verify cmd_search_tests calls _search_tests_for_impl."""
        from cli.commands.search import cmd_search_tests

        args = MagicMock()
        args.query = "UserService"
        args.limit = 5
        args.collection = None
        args.language = "python"
        args.under = "tests/"
        args.include_snippet = False
        args.context_lines = 2
        args.compact = False

        # Patch at the source module, not where it's imported
        with patch('scripts.mcp_impl.search_specialized._search_tests_for_impl') as mock_impl:
            mock_impl.return_value = {"results": []}

            cmd_search_tests(args)

            # Verify the implementation was called
            assert mock_impl.called

            call_kwargs = mock_impl.call_args[1]
            assert call_kwargs['query'] == "UserService"
            assert call_kwargs['limit'] == 5
            assert call_kwargs['language'] == "python"
            assert call_kwargs['under'] == "tests/"

            # Verify repo_search_fn was passed
            assert 'repo_search_fn' in call_kwargs
            assert callable(call_kwargs['repo_search_fn'])

    def test_search_config_delegates_to_impl(self):
        """Verify cmd_search_config calls _search_config_for_impl."""
        from cli.commands.search import cmd_search_config

        args = MagicMock()
        args.query = "database"
        args.limit = 10
        args.collection = None
        args.under = "config/"
        args.include_snippet = False
        args.context_lines = 2
        args.compact = False

        with patch('scripts.mcp_impl.search_specialized._search_config_for_impl') as mock_impl:
            mock_impl.return_value = {"results": []}

            cmd_search_config(args)

            assert mock_impl.called
            call_kwargs = mock_impl.call_args[1]
            assert call_kwargs['query'] == "database"
            assert 'repo_search_fn' in call_kwargs

    def test_search_callers_delegates_to_impl(self):
        """Verify cmd_search_callers calls _search_callers_for_impl."""
        from cli.commands.search import cmd_search_callers

        args = MagicMock()
        args.query = "processPayment"
        args.limit = 10
        args.collection = None
        args.language = "typescript"

        with patch('scripts.mcp_impl.search_specialized._search_callers_for_impl') as mock_impl:
            mock_impl.return_value = {"results": []}

            cmd_search_callers(args)

            assert mock_impl.called
            call_kwargs = mock_impl.call_args[1]
            assert call_kwargs['query'] == "processPayment"
            assert call_kwargs['language'] == "typescript"
            assert 'repo_search_fn' in call_kwargs

    def test_symbol_graph_delegates_to_impl(self):
        """Verify cmd_symbol_graph calls _symbol_graph_impl."""
        from cli.commands.symbol import cmd_symbol_graph

        args = MagicMock()
        args.symbol = "ASTAnalyzer"
        args.query_type = "definition"
        args.limit = 10
        args.collection = None
        args.language = "python"
        args.under = None

        with patch('scripts.mcp_impl.symbol_graph._symbol_graph_impl') as mock_impl:
            mock_impl.return_value = {"results": []}

            cmd_symbol_graph(args)

            assert mock_impl.called
            call_kwargs = mock_impl.call_args[1]
            assert call_kwargs['symbol'] == "ASTAnalyzer"
            assert call_kwargs['query_type'] == "definition"

    def test_pattern_search_delegates_to_impl(self):
        """Verify cmd_pattern_search calls _pattern_search_impl."""
        from cli.commands.pattern import cmd_pattern_search

        args = MagicMock()
        args.query = "if err != nil { return err }"
        args.language = "go"
        args.limit = 10
        args.min_score = 0.3
        args.include_snippet = False
        args.context_lines = 2
        args.query_mode = "auto"
        args.collection = None

        with patch('scripts.mcp_impl.pattern_search._pattern_search_impl') as mock_impl:
            mock_impl.return_value = {"results": []}

            cmd_pattern_search(args)

            assert mock_impl.called
            call_kwargs = mock_impl.call_args[1]
            assert call_kwargs['query'] == "if err != nil { return err }"
            assert call_kwargs['language'] == "go"
            assert call_kwargs['min_score'] == 0.3


class TestCollectionResolution:
    """Test collection resolution logic."""

    def test_resolve_collection_with_override(self):
        """Explicit collection arg takes precedence."""
        from cli.core import resolve_collection

        result = resolve_collection(override="my-collection")
        assert result == "my-collection"

    def test_resolve_collection_with_env(self):
        """Env var takes precedence over default."""
        import os
        from cli.core import DEFAULT_COLLECTION

        # Save current value
        old_val = os.environ.get('DEFAULT_COLLECTION')

        try:
            os.environ['DEFAULT_COLLECTION'] = 'env-collection'
            # Need to reimport to get updated value
            import importlib
            import cli.core
            importlib.reload(cli.core)
            from cli.core import resolve_collection

            result = resolve_collection()
            assert result == 'env-collection'
        finally:
            # Restore
            if old_val is None:
                os.environ.pop('DEFAULT_COLLECTION', None)
            else:
                os.environ['DEFAULT_COLLECTION'] = old_val
            # Reload to restore
            import importlib
            import cli.core
            importlib.reload(cli.core)

    def test_resolve_collection_default(self):
        """Returns default when no override or env."""
        import os
        from cli.core import resolve_collection

        # Clear env vars
        old_default = os.environ.get('DEFAULT_COLLECTION')
        old_coll = os.environ.get('COLLECTION_NAME')

        try:
            os.environ.pop('DEFAULT_COLLECTION', None)
            os.environ.pop('COLLECTION_NAME', None)

            result = resolve_collection()
            assert result == 'codebase'
        finally:
            if old_default:
                os.environ['DEFAULT_COLLECTION'] = old_default
            if old_coll:
                os.environ['COLLECTION_NAME'] = old_coll


class TestDependencyInjection:
    """Test that CLI correctly wires up dependencies."""

    def test_repo_search_async_provides_callbacks(self):
        """Verify repo_search_async provides all required callbacks."""
        from cli.core import repo_search_async

        # These are the callback parameters that should be provided
        callback_params = [
            'get_embedding_model_fn',
            'require_auth_session_fn',
            'do_highlight_snippet_fn',
            'run_async_fn',
        ]

        # Test that repo_search_async can be called
        # (it will fail without Qdrant, but we can check it tries to call _repo_search_impl)
        with patch('scripts.mcp_impl.search._repo_search_impl') as mock_impl:
            mock_impl.return_value = {"results": []}

            # Call the function
            import asyncio
            asyncio.run(repo_search_async(query="test", limit=5))

            # Verify it was called
            assert mock_impl.called

            # Get the kwargs passed to _repo_search_impl
            call_kwargs = mock_impl.call_args[1]

            # Verify all callback params were provided
            for param in callback_params:
                assert param in call_kwargs, f"Missing callback: {param}"
                if param != 'do_highlight_snippet_fn':  # This one can be None
                    assert call_kwargs[param] is not None, f"Callback {param} is None"


class TestIndexCommand:
    """Test index command integration."""

    def test_index_calls_index_repo(self):
        """Verify cmd_index calls index_repo with correct params."""
        from cli.commands.index import cmd_index

        args = MagicMock()
        args.path = "."
        args.subdir = None
        args.collection = None
        args.recreate = False

        with patch('scripts.ingest.pipeline.index_repo') as mock_index:
            with patch('cli.commands.index.get_client') as mock_client:
                mock_client.return_value = MagicMock()

                cmd_index(args)

                assert mock_index.called

                call_kwargs = mock_index.call_args[1]
                # Verify it was called with a Path object
                assert 'root' in call_kwargs
                assert 'qdrant_url' in call_kwargs
                assert 'collection' in call_kwargs
                assert 'model_name' in call_kwargs
                assert 'recreate' in call_kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
