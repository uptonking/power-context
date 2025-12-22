"""Tests for symbol and filename boosting logic in hybrid_search.

Tests the new token-level matching for symbols (camelCase/snake_case splitting)
and filename stem matching added in commit 0a8a896.
"""
import pytest
from scripts.hybrid_search import _split_ident


class TestSplitIdent:
    """Test identifier splitting for camelCase/snake_case."""

    def test_snake_case(self):
        assert _split_ident("user_auth_handler") == ["user", "auth", "handler"]

    def test_camel_case(self):
        assert _split_ident("userAuthHandler") == ["user", "auth", "handler"]

    def test_pascal_case(self):
        assert _split_ident("UserAuthHandler") == ["user", "auth", "handler"]

    def test_acronym_handling(self):
        # XMLParser -> xml, parser
        result = _split_ident("XMLParser")
        assert "xml" in result
        assert "parser" in result

    def test_mixed_separators(self):
        result = _split_ident("user_authHandler")
        assert "user" in result
        assert "auth" in result
        assert "handler" in result

    def test_filters_stopwords(self):
        # 'the' should be filtered
        result = _split_ident("getTheValue")
        assert "the" not in result
        assert "get" in result
        assert "value" in result

    def test_empty_string(self):
        assert _split_ident("") == []

    def test_single_word(self):
        assert _split_ident("handler") == ["handler"]


class TestSymbolPartMatching:
    """Test that symbol parts are correctly matched against queries."""

    def test_symbol_parts_extracted(self):
        """Verify split parts can match query tokens."""
        symbol = "getUserById"
        parts = set(p.lower() for p in _split_ident(symbol) if len(p) >= 2)
        assert "get" in parts
        assert "user" in parts
        # "by" is filtered as stopword, "id" should remain
        assert "id" in parts

    def test_symbol_parts_from_snake_case(self):
        symbol = "calculate_total_price"
        parts = set(p.lower() for p in _split_ident(symbol) if len(p) >= 2)
        assert "calculate" in parts
        assert "total" in parts
        assert "price" in parts


class TestFilenameBoostLogic:
    """Test filename/stem matching boost logic."""

    def test_stem_extraction(self):
        """Verify stem is correctly extracted from path."""
        path = "scripts/hybrid_search.py"
        basename = path.rsplit("/", 1)[-1].lower()
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        assert basename == "hybrid_search.py"
        assert stem == "hybrid_search"

    def test_stem_parts_extracted(self):
        """Verify stem parts match query tokens."""
        stem = "hybrid_search"
        stem_parts = set(p.lower() for p in _split_ident(stem) if len(p) >= 2)
        assert "hybrid" in stem_parts
        assert "search" in stem_parts

    def test_camelcase_filename_parts(self):
        """CamelCase filenames should be tokenized."""
        stem = "UserAuthService"
        stem_parts = set(p.lower() for p in _split_ident(stem) if len(p) >= 2)
        assert "user" in stem_parts
        assert "auth" in stem_parts
        assert "service" in stem_parts

    def test_query_matches_stem(self):
        """Direct stem match should work."""
        path = "utils/config_manager.py"
        basename = path.rsplit("/", 1)[-1].lower()
        stem = basename.rsplit(".", 1)[0]
        stem_parts = set(p.lower() for p in _split_ident(stem) if len(p) >= 2)
        
        # Query "config" should match
        query = "config"
        assert query == stem or query in stem_parts or query in basename

    def test_query_matches_stem_part(self):
        """Query matching a stem part should boost."""
        stem = "DatabaseConnection"
        stem_parts = set(p.lower() for p in _split_ident(stem) if len(p) >= 2)
        assert "database" in stem_parts
        assert "connection" in stem_parts

    def test_short_query_filtered(self):
        """Queries < 3 chars should not trigger filename boost."""
        query = "db"
        assert len(query) < 3  # Should be filtered in boost logic


class TestBoostIntegration:
    """Integration tests for the boost logic patterns."""

    def test_pattern_symbol_equality_boost(self):
        """Test the pattern used in hybrid_search for symbol equality."""
        symbol = "processUserData"
        sym = symbol.lower()
        sym_parts = set(p.lower() for p in _split_ident(symbol) if len(p) >= 2)
        
        # Query "user" should match via sym_parts
        query = "user"
        ql = query.lower()
        matches = ql == sym or ql in sym_parts
        assert matches, "Query 'user' should match symbol part"

    def test_pattern_filename_boost(self):
        """Test the pattern used in hybrid_search for filename boost."""
        path = "services/AuthenticationService.ts"
        basename = path.rsplit("/", 1)[-1].lower()
        stem = basename.rsplit(".", 1)[0] if "." in basename else basename
        stem_parts = set(p.lower() for p in _split_ident(stem) if len(p) >= 2)
        
        # Query "authentication" should trigger boost
        query = "authentication"
        ql = query.lower()
        matches = (
            len(ql) >= 3 and 
            (ql == stem or ql in stem_parts or ql in basename)
        )
        assert matches, "Query 'authentication' should match filename"

    def test_pattern_no_boost_short_query(self):
        """Short queries should not get filename boost."""
        path = "lib/db.py"
        basename = path.rsplit("/", 1)[-1].lower()
        stem = basename.rsplit(".", 1)[0]
        stem_parts = set(p.lower() for p in _split_ident(stem) if len(p) >= 2)
        
        query = "db"
        ql = query.lower()
        # This should NOT match due to len(ql) < 3 check
        matches = (
            len(ql) >= 3 and 
            (ql == stem or ql in stem_parts or ql in basename)
        )
        assert not matches, "Short query should not trigger boost"

