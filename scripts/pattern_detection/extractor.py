"""
Pattern Extractor - Language-agnostic AST-based structural feature extraction.

Extracts three types of features from code in ANY language:
1. AST Paths (code2vec style) - paths between terminals in AST
2. Structural Features (AROMA style) - node type sequences, depth patterns
3. Control Flow Features - loop/branch structure fingerprints

The key innovation: features are CONTENT-AGNOSTIC and LANGUAGE-NORMALIZED.
Variable names, string literals, and specific function names are abstracted
away. Language-specific AST nodes are normalized to universal concepts
(LOOP, BRANCH, TRY, FUNC, etc.) enabling cross-language pattern matching.

Supported languages: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++,
Ruby, PHP, C#, Kotlin, Swift, Scala, and more via Tree-sitter.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import Counter
import hashlib
import re


@dataclass
class PatternSignature:
    """Structural signature of a code snippet - language-independent."""

    # AST path features (code2vec style)
    # Each path: (start_type, path_nodes, end_type, count)
    # Types are NORMALIZED across languages
    ast_paths: List[Tuple[str, str, str, int]] = field(default_factory=list)

    # Structural features (AROMA style)
    # Normalized node type n-grams and their frequencies
    structural_ngrams: Counter = field(default_factory=Counter)

    # Control flow features - universal across all languages
    # Encoded as: (loop_depth, branch_count, try_depth, has_finally, ...)
    control_flow: Dict[str, Any] = field(default_factory=dict)

    # Original language (for reference, not used in matching)
    language: str = "unknown"

    # Hash for quick equality check
    _hash: Optional[str] = None

    def fingerprint(self) -> str:
        """Generate a compact fingerprint for this signature."""
        if self._hash:
            return self._hash

        # Combine top features into stable string
        top_paths = sorted(self.ast_paths, key=lambda x: -x[3])[:20]
        top_ngrams = self.structural_ngrams.most_common(20)

        parts = [
            # NOTE: Language NOT included - we want cross-language matching
            f"P:{','.join(f'{p[0]}>{p[1]}>{p[2]}' for p in top_paths)}",
            f"N:{','.join(f'{n}:{c}' for n, c in top_ngrams)}",
            f"C:{self.control_flow.get('signature', '')}",
        ]

        combined = "|".join(parts)
        self._hash = hashlib.md5(combined.encode()).hexdigest()[:16]
        return self._hash


# =============================================================================
# Language-specific node type mappings → Universal types
# =============================================================================

# Universal terminal types (content-bearing nodes)
TERMINAL_NORMALIZATION: Dict[str, str] = {
    # Identifiers (all languages)
    "identifier": "ID", "name": "ID", "property_identifier": "ID",
    "field_identifier": "ID", "type_identifier": "TYPE_ID",
    "shorthand_property_identifier": "ID", "statement_identifier": "ID",

    # Strings
    "string": "STR", "string_literal": "STR", "interpreted_string_literal": "STR",
    "raw_string_literal": "STR", "template_string": "STR", "string_content": "STR",
    "char_literal": "STR", "rune_literal": "STR",

    # Numbers
    "number": "NUM", "integer": "NUM", "integer_literal": "NUM", "int_literal": "NUM",
    "float": "NUM", "float_literal": "NUM", "decimal_integer_literal": "NUM",
    "hex_integer_literal": "NUM", "binary_integer_literal": "NUM",
    "octal_integer_literal": "NUM", "decimal_floating_point_literal": "NUM",

    # Booleans
    "true": "BOOL", "false": "BOOL", "boolean": "BOOL",

    # Null/None/Nil
    "none": "NIL", "null": "NIL", "nil": "NIL", "null_literal": "NIL",

    # Keywords become themselves (normalized case)
    "return": "RETURN", "break": "BREAK", "continue": "CONTINUE",
    "yield": "YIELD", "await": "AWAIT", "async": "ASYNC",
}

# Universal control flow types - maps language-specific AST nodes to universal concepts
CONTROL_FLOW_NORMALIZATION: Dict[str, Dict[str, str]] = {
    # Python
    "python": {
        "for_statement": "LOOP_FOR", "while_statement": "LOOP_WHILE",
        "if_statement": "BRANCH_IF", "elif_clause": "BRANCH_ELIF", "else_clause": "BRANCH_ELSE",
        "try_statement": "TRY", "except_clause": "CATCH", "finally_clause": "FINALLY",
        "with_statement": "RESOURCE_GUARD", "match_statement": "MATCH",
        "function_definition": "FUNC_DEF", "async_function_definition": "FUNC_DEF",
        "class_definition": "CLASS_DEF", "lambda": "LAMBDA",
        "list_comprehension": "COMPREHENSION", "dict_comprehension": "COMPREHENSION",
        "generator_expression": "GENERATOR",
    },
    # JavaScript / TypeScript
    "javascript": {
        "for_statement": "LOOP_FOR", "for_in_statement": "LOOP_FOR",
        "while_statement": "LOOP_WHILE", "do_statement": "LOOP_DO",
        "if_statement": "BRANCH_IF", "else_clause": "BRANCH_ELSE",
        "switch_statement": "MATCH", "case": "MATCH_CASE",
        "try_statement": "TRY", "catch_clause": "CATCH", "finally_clause": "FINALLY",
        "function_declaration": "FUNC_DEF", "function_expression": "FUNC_EXPR",
        "arrow_function": "LAMBDA", "class_declaration": "CLASS_DEF",
        "method_definition": "METHOD_DEF",
    },
    "typescript": {},  # Inherits from javascript
    # Go
    "go": {
        "for_statement": "LOOP_FOR", "range_clause": "LOOP_RANGE",
        "if_statement": "BRANCH_IF", "else_clause": "BRANCH_ELSE",
        "switch_statement": "MATCH", "type_switch_statement": "MATCH",
        "select_statement": "SELECT", "case_clause": "MATCH_CASE",
        "defer_statement": "DEFER", "go_statement": "GOROUTINE",
        "function_declaration": "FUNC_DEF", "method_declaration": "METHOD_DEF",
        "func_literal": "LAMBDA", "type_declaration": "TYPE_DEF",
    },
    # Rust
    "rust": {
        "for_expression": "LOOP_FOR", "while_expression": "LOOP_WHILE",
        "loop_expression": "LOOP_INFINITE",
        "if_expression": "BRANCH_IF", "else_clause": "BRANCH_ELSE",
        "match_expression": "MATCH", "match_arm": "MATCH_CASE",
        "function_item": "FUNC_DEF", "closure_expression": "LAMBDA",
        "impl_item": "IMPL", "trait_item": "TRAIT_DEF", "struct_item": "STRUCT_DEF",
        "enum_item": "ENUM_DEF", "macro_invocation": "MACRO",
    },
    # Java
    "java": {
        "for_statement": "LOOP_FOR", "enhanced_for_statement": "LOOP_FOR",
        "while_statement": "LOOP_WHILE", "do_statement": "LOOP_DO",
        "if_statement": "BRANCH_IF", "else": "BRANCH_ELSE",
        "switch_expression": "MATCH", "switch_block_statement_group": "MATCH_CASE",
        "try_statement": "TRY", "catch_clause": "CATCH", "finally_clause": "FINALLY",
        "try_with_resources_statement": "RESOURCE_GUARD",
        "method_declaration": "FUNC_DEF", "constructor_declaration": "CONSTRUCTOR",
        "class_declaration": "CLASS_DEF", "interface_declaration": "INTERFACE_DEF",
        "lambda_expression": "LAMBDA",
    },
    # C / C++
    "c": {
        "for_statement": "LOOP_FOR", "while_statement": "LOOP_WHILE",
        "do_statement": "LOOP_DO", "if_statement": "BRANCH_IF",
        "else_clause": "BRANCH_ELSE", "switch_statement": "MATCH",
        "case_statement": "MATCH_CASE", "function_definition": "FUNC_DEF",
        "struct_specifier": "STRUCT_DEF", "enum_specifier": "ENUM_DEF",
    },
    "cpp": {
        "for_statement": "LOOP_FOR", "for_range_loop": "LOOP_FOR",
        "while_statement": "LOOP_WHILE", "do_statement": "LOOP_DO",
        "if_statement": "BRANCH_IF", "else_clause": "BRANCH_ELSE",
        "switch_statement": "MATCH", "try_statement": "TRY",
        "catch_clause": "CATCH", "function_definition": "FUNC_DEF",
        "class_specifier": "CLASS_DEF", "lambda_expression": "LAMBDA",
    },
    # Ruby
    "ruby": {
        "for": "LOOP_FOR", "while": "LOOP_WHILE", "until": "LOOP_WHILE",
        "if": "BRANCH_IF", "unless": "BRANCH_IF", "elsif": "BRANCH_ELIF",
        "else": "BRANCH_ELSE", "case": "MATCH", "when": "MATCH_CASE",
        "begin": "TRY", "rescue": "CATCH", "ensure": "FINALLY",
        "method": "FUNC_DEF", "singleton_method": "FUNC_DEF",
        "class": "CLASS_DEF", "module": "MODULE_DEF",
        "lambda": "LAMBDA", "block": "BLOCK", "do_block": "BLOCK",
    },
    # C#
    "c_sharp": {
        "for_statement": "LOOP_FOR", "foreach_statement": "LOOP_FOR",
        "while_statement": "LOOP_WHILE", "do_statement": "LOOP_DO",
        "if_statement": "BRANCH_IF", "else_clause": "BRANCH_ELSE",
        "switch_statement": "MATCH", "switch_expression": "MATCH",
        "try_statement": "TRY", "catch_clause": "CATCH", "finally_clause": "FINALLY",
        "using_statement": "RESOURCE_GUARD",
        "method_declaration": "FUNC_DEF", "local_function_statement": "FUNC_DEF",
        "class_declaration": "CLASS_DEF", "interface_declaration": "INTERFACE_DEF",
        "lambda_expression": "LAMBDA",
    },
    # Bash / Shell
    "bash": {
        "for_statement": "LOOP_FOR", "while_statement": "LOOP_WHILE",
        "until_statement": "LOOP_WHILE",
        "if_statement": "BRANCH_IF", "elif_clause": "BRANCH_ELIF", "else_clause": "BRANCH_ELSE",
        "case_statement": "MATCH", "case_item": "MATCH_CASE",
        "function_definition": "FUNC_DEF",
        "subshell": "SUBSHELL", "command_substitution": "SUBSHELL",
        "pipeline": "PIPELINE",
    },
    # JSON (structural - no control flow, but patterns in structure)
    "json": {
        "object": "OBJECT", "array": "ARRAY", "pair": "KEY_VALUE",
        "string": "STRING", "number": "NUMBER", "true": "BOOL", "false": "BOOL", "null": "NULL",
    },
    # YAML (structural)
    "yaml": {
        "block_mapping": "OBJECT", "flow_mapping": "OBJECT",
        "block_sequence": "ARRAY", "flow_sequence": "ARRAY",
        "block_mapping_pair": "KEY_VALUE", "flow_pair": "KEY_VALUE",
        "anchor": "ANCHOR", "alias": "ALIAS",
    },
    # HTML
    "html": {
        "element": "ELEMENT", "self_closing_tag": "ELEMENT",
        "start_tag": "TAG_OPEN", "end_tag": "TAG_CLOSE",
        "script_element": "SCRIPT", "style_element": "STYLE",
        "attribute": "ATTRIBUTE",
        "doctype": "DOCTYPE", "comment": "COMMENT",
    },
    # CSS
    "css": {
        "rule_set": "RULE", "media_statement": "MEDIA_QUERY",
        "keyframes_statement": "KEYFRAMES", "supports_statement": "SUPPORTS",
        "declaration": "DECLARATION", "selector": "SELECTOR",
        "class_selector": "CLASS_SEL", "id_selector": "ID_SEL",
        "pseudo_class_selector": "PSEUDO_CLASS", "pseudo_element_selector": "PSEUDO_ELEM",
    },
    # Markdown (structural patterns in documents)
    "markdown": {
        "atx_heading": "HEADING", "setext_heading": "HEADING",
        "paragraph": "PARAGRAPH",
        "fenced_code_block": "CODE_BLOCK", "indented_code_block": "CODE_BLOCK",
        "list": "LIST", "list_item": "LIST_ITEM",
        "block_quote": "QUOTE", "thematic_break": "HR",
        "link": "LINK", "image": "IMAGE",
    },
}

# Build a unified lookup by flattening all language mappings
_UNIFIED_CF_MAP: Dict[str, str] = {}
for lang_map in CONTROL_FLOW_NORMALIZATION.values():
    _UNIFIED_CF_MAP.update(lang_map)

# =============================================================================
# Dynamic fallback patterns - work for ANY language even without explicit mapping
# =============================================================================
# These patterns match common AST node naming conventions across tree-sitter grammars.
# If a language isn't explicitly mapped, we try to match by common patterns.

_UNIVERSAL_PATTERNS: Dict[str, str] = {
    # Loops - most grammars use these patterns
    "for_statement": "LOOP_FOR", "for_expression": "LOOP_FOR",
    "for_loop": "LOOP_FOR", "for_in": "LOOP_FOR", "for_of": "LOOP_FOR",
    "while_statement": "LOOP_WHILE", "while_expression": "LOOP_WHILE",
    "while_loop": "LOOP_WHILE",
    "do_statement": "LOOP_DO", "do_while": "LOOP_DO",
    "loop": "LOOP_INFINITE", "loop_expression": "LOOP_INFINITE",

    # Conditionals
    "if_statement": "BRANCH_IF", "if_expression": "BRANCH_IF",
    "else_clause": "BRANCH_ELSE", "else": "BRANCH_ELSE",
    "elif_clause": "BRANCH_ELIF", "elsif": "BRANCH_ELIF", "else_if": "BRANCH_ELIF",
    "conditional_expression": "BRANCH_TERNARY", "ternary": "BRANCH_TERNARY",

    # Switch/Match
    "switch_statement": "MATCH", "switch_expression": "MATCH",
    "match_expression": "MATCH", "case_statement": "MATCH",
    "case": "MATCH_CASE", "switch_case": "MATCH_CASE", "match_arm": "MATCH_CASE",
    "when": "MATCH_CASE", "case_clause": "MATCH_CASE",

    # Error handling
    "try_statement": "TRY", "try_expression": "TRY", "try": "TRY",
    "catch_clause": "CATCH", "catch": "CATCH", "except": "CATCH",
    "except_clause": "CATCH", "rescue": "CATCH",
    "finally_clause": "FINALLY", "finally": "FINALLY", "ensure": "FINALLY",
    "throw_statement": "THROW", "raise_statement": "THROW", "throw": "THROW",

    # Functions
    "function_definition": "FUNC_DEF", "function_declaration": "FUNC_DEF",
    "function_item": "FUNC_DEF", "method_definition": "METHOD_DEF",
    "method_declaration": "METHOD_DEF", "method": "FUNC_DEF",
    "lambda": "LAMBDA", "lambda_expression": "LAMBDA", "arrow_function": "LAMBDA",
    "closure": "LAMBDA", "closure_expression": "LAMBDA",
    "func_literal": "LAMBDA", "anonymous_function": "LAMBDA",

    # Classes/Types
    "class_definition": "CLASS_DEF", "class_declaration": "CLASS_DEF",
    "class": "CLASS_DEF", "class_specifier": "CLASS_DEF",
    "struct_definition": "STRUCT_DEF", "struct_specifier": "STRUCT_DEF",
    "struct": "STRUCT_DEF", "struct_item": "STRUCT_DEF",
    "interface_declaration": "INTERFACE_DEF", "interface": "INTERFACE_DEF",
    "trait_item": "TRAIT_DEF", "trait": "TRAIT_DEF",
    "enum_declaration": "ENUM_DEF", "enum_item": "ENUM_DEF", "enum": "ENUM_DEF",
    "type_declaration": "TYPE_DEF", "type_alias": "TYPE_DEF",

    # Resource management
    "with_statement": "RESOURCE_GUARD", "using_statement": "RESOURCE_GUARD",
    "defer_statement": "DEFER", "defer": "DEFER",
}


class PatternExtractor:
    """Extract structural patterns from code using Tree-sitter - multi-language."""

    MAX_PATH_LENGTH = 8
    MAX_PATHS = 200
    NGRAM_SIZES = [2, 3, 4]

    def __init__(self):
        self._parsers: Dict[str, Any] = {}
        
    def _get_parser(self, language: str):
        """Get or create Tree-sitter parser for language."""
        # Normalize language name
        lang_key = self._normalize_language(language)

        if lang_key in self._parsers:
            return self._parsers[lang_key]

        try:
            from scripts.ingest.tree_sitter import _ts_parser
            parser = _ts_parser(lang_key)
            if parser:
                self._parsers[lang_key] = parser
                return parser
        except ImportError:
            pass
        return None

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language name to Tree-sitter key.

        Handles all common aliases, extensions, and variations.
        Returns lowercase normalized key that matches our normalization maps.
        """
        lang_map = {
            # Python
            "py": "python", "python3": "python", "python2": "python",
            # JavaScript
            "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
            "node": "javascript", "nodejs": "javascript",
            # TypeScript
            "ts": "typescript", "tsx": "typescript", "mts": "typescript",
            # Go
            "golang": "go",
            # Rust
            "rs": "rust",
            # Ruby
            "rb": "ruby", "rake": "ruby", "gemspec": "ruby",
            # C#
            "cs": "c_sharp", "csharp": "c_sharp",
            # C++
            "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hxx": "cpp",
            # Java
            "jav": "java",
            # Shell/Bash
            "bash": "bash", "sh": "bash", "zsh": "bash", "shell": "bash",
            "ksh": "bash", "fish": "bash",
            # Config languages
            "yml": "yaml",
            "htm": "html", "xhtml": "html",
            "scss": "css", "sass": "css", "less": "css",
            "md": "markdown", "mdx": "markdown",
            # Other common
            "kt": "kotlin", "kts": "kotlin",
            "swift": "swift",
            "php": "php",
            "scala": "scala", "sc": "scala",
            "pl": "perl", "pm": "perl",
            "lua": "lua",
            "r": "r",
            "jl": "julia",
            "ex": "elixir", "exs": "elixir",
            "erl": "erlang", "hrl": "erlang",
            "hs": "haskell", "lhs": "haskell",
            "clj": "clojure", "cljs": "clojure", "cljc": "clojure",
            "ml": "ocaml", "mli": "ocaml",
            "fs": "fsharp", "fsi": "fsharp", "fsx": "fsharp",
            "ps1": "powershell", "psm1": "powershell",
            "vue": "vue",
            "svelte": "svelte",
        }
        return lang_map.get(language.lower(), language.lower())

    def _get_cf_map(self, language: str) -> Dict[str, str]:
        """
        Get control flow normalization map for a language.

        Falls back gracefully:
        1. Exact language match
        2. TypeScript -> JavaScript inheritance
        3. Empty dict (will use _UNIVERSAL_PATTERNS fallback)
        """
        lang_key = self._normalize_language(language)

        # TypeScript inherits from JavaScript
        if lang_key == "typescript" and lang_key not in CONTROL_FLOW_NORMALIZATION:
            lang_key = "javascript"

        # Return the map or empty dict - _UNIVERSAL_PATTERNS will catch unknowns
        return CONTROL_FLOW_NORMALIZATION.get(lang_key, {})

    def extract(self, code: str, language: str = "python") -> PatternSignature:
        """Extract pattern signature from code - works for any supported language."""
        lang_key = self._normalize_language(language)
        sig = PatternSignature(language=lang_key)

        parser = self._get_parser(lang_key)
        if not parser:
            # Fallback to regex-based extraction
            return self._extract_regex_fallback(code, lang_key)

        try:
            tree = parser.parse(code.encode("utf-8"))
            if tree is None:
                return self._extract_regex_fallback(code, lang_key)
            root = tree.root_node
        except Exception:
            return self._extract_regex_fallback(code, lang_key)

        # Extract features with language-aware normalization
        sig.ast_paths = self._extract_ast_paths(root, code, lang_key)
        sig.structural_ngrams = self._extract_ngrams(root, lang_key)
        sig.control_flow = self._extract_control_flow(root, lang_key)

        return sig

    def _extract_ast_paths(self, root, code: str, language: str) -> List[Tuple[str, str, str, int]]:
        """Extract code2vec-style AST paths between terminals - normalized across languages."""
        terminals = []
        self._collect_terminals(root, terminals, depth=0)

        paths: Counter = Counter()
        cf_map = self._get_cf_map(language)

        # Generate paths between pairs of terminals
        for i, (node_i, depth_i) in enumerate(terminals):
            for j, (node_j, depth_j) in enumerate(terminals[i+1:min(i+10, len(terminals))]):
                path = self._compute_path(node_i, node_j, cf_map)
                if path and len(path) <= self.MAX_PATH_LENGTH:
                    start_type = self._normalize_node_type(node_i.type, cf_map)
                    end_type = self._normalize_node_type(node_j.type, cf_map)
                    path_str = "^".join(path)
                    paths[(start_type, path_str, end_type)] += 1

        return [(s, p, e, c) for (s, p, e), c in paths.most_common(self.MAX_PATHS)]

    def _normalize_node_type(self, node_type: str, cf_map: Dict[str, str]) -> str:
        """
        Normalize node type to universal type - enables cross-language matching.

        Priority order:
        1. Terminal normalization (literals, identifiers)
        2. Language-specific control flow map
        3. Unified map (all languages combined)
        4. Universal patterns (dynamic fallback for any language)
        5. Lowercase as-is (preserves unknown types consistently)
        """
        # First check terminal normalization
        if node_type in TERMINAL_NORMALIZATION:
            return TERMINAL_NORMALIZATION[node_type]
        # Then check control flow normalization (language-specific)
        if node_type in cf_map:
            return cf_map[node_type]
        # Then check unified map
        if node_type in _UNIFIED_CF_MAP:
            return _UNIFIED_CF_MAP[node_type]
        # Dynamic fallback: try universal patterns (works for ANY language)
        if node_type in _UNIVERSAL_PATTERNS:
            return _UNIVERSAL_PATTERNS[node_type]
        # Keep as-is but lowercase for consistency
        return node_type.lower()

    def _collect_terminals(self, node, terminals: List, depth: int):
        """Collect terminal (leaf) nodes."""
        if node.child_count == 0:
            terminals.append((node, depth))
        else:
            for child in node.children:
                self._collect_terminals(child, terminals, depth + 1)

    def _compute_path(self, node_a, node_b, cf_map: Dict[str, str]) -> Optional[List[str]]:
        """Compute AST path between two nodes (up to LCA, then down) - normalized."""
        # Get ancestors of both nodes
        ancestors_a = []
        n = node_a
        while n is not None:
            ancestors_a.append(n)
            n = n.parent

        ancestors_b = []
        n = node_b
        while n is not None:
            ancestors_b.append(n)
            n = n.parent

        # Find LCA
        set_a = set(id(n) for n in ancestors_a)
        lca = None
        lca_idx_b = 0
        for i, n in enumerate(ancestors_b):
            if id(n) in set_a:
                lca = n
                lca_idx_b = i
                break

        if lca is None:
            return None

        lca_idx_a = next(i for i, n in enumerate(ancestors_a) if id(n) == id(lca))

        # Build path: up from a to LCA, then down to b
        # Node types are normalized for cross-language matching
        path = []
        for n in ancestors_a[:lca_idx_a]:
            path.append(f"↑{self._normalize_node_type(n.type, cf_map)}")
        path.append(f"○{self._normalize_node_type(lca.type, cf_map)}")
        for n in reversed(ancestors_b[:lca_idx_b]):
            path.append(f"↓{self._normalize_node_type(n.type, cf_map)}")

        return path

    def _extract_ngrams(self, root, language: str) -> Counter:
        """Extract structural n-grams from AST traversal - normalized."""
        cf_map = self._get_cf_map(language)

        # Pre-order traversal of normalized node types
        node_types = []
        self._collect_node_types(root, node_types, cf_map)

        ngrams = Counter()
        for n in self.NGRAM_SIZES:
            for i in range(len(node_types) - n + 1):
                gram = tuple(node_types[i:i+n])
                ngrams[gram] += 1

        return ngrams

    def _collect_node_types(self, node, types: List[str], cf_map: Dict[str, str]):
        """Collect normalized node types in pre-order."""
        types.append(self._normalize_node_type(node.type, cf_map))
        for child in node.children:
            self._collect_node_types(child, types, cf_map)

    def _extract_control_flow(self, root, language: str) -> Dict[str, Any]:
        """Extract control flow structure features - language-agnostic."""
        cf_map = self._get_cf_map(language)

        cf = {
            "max_loop_depth": 0,
            "loop_count": 0,
            "branch_count": 0,
            "try_count": 0,
            "has_finally": False,
            "has_catch": False,  # Renamed from has_except for universality
            "has_resource_guard": False,  # with/using/defer/try-with-resources
            "func_count": 0,
            "class_count": 0,
            "lambda_count": 0,
            "match_count": 0,  # switch/match expressions
        }

        self._analyze_control_flow(root, cf, cf_map, loop_depth=0)

        # Generate compact signature - universal across all languages
        cf["signature"] = (
            f"L{cf['max_loop_depth']}_{cf['loop_count']}_"
            f"B{cf['branch_count']}_T{cf['try_count']}_"
            f"M{cf['match_count']}_"
            f"{'F' if cf['has_finally'] else '_'}"
            f"{'C' if cf['has_catch'] else '_'}"
            f"{'R' if cf['has_resource_guard'] else '_'}"
        )

        return cf

    def _analyze_control_flow(self, node, cf: Dict, cf_map: Dict[str, str], loop_depth: int):
        """Recursively analyze control flow - uses normalized types."""
        normalized = self._normalize_node_type(node.type, cf_map)

        # Loop detection (LOOP_FOR, LOOP_WHILE, LOOP_DO, LOOP_INFINITE, LOOP_RANGE)
        if normalized.startswith("LOOP_"):
            cf["loop_count"] += 1
            loop_depth += 1
            cf["max_loop_depth"] = max(cf["max_loop_depth"], loop_depth)
        # Branch detection
        elif normalized == "BRANCH_IF":
            cf["branch_count"] += 1
        # Try block detection
        elif normalized == "TRY":
            cf["try_count"] += 1
        # Catch/except detection
        elif normalized == "CATCH":
            cf["has_catch"] = True
        # Finally detection
        elif normalized == "FINALLY":
            cf["has_finally"] = True
        # Resource guard (with/using/defer)
        elif normalized in ("RESOURCE_GUARD", "DEFER"):
            cf["has_resource_guard"] = True
        # Function definition
        elif normalized in ("FUNC_DEF", "METHOD_DEF"):
            cf["func_count"] += 1
        # Class/struct definition
        elif normalized in ("CLASS_DEF", "STRUCT_DEF", "INTERFACE_DEF"):
            cf["class_count"] += 1
        # Lambda/closure
        elif normalized == "LAMBDA":
            cf["lambda_count"] += 1
        # Match/switch
        elif normalized == "MATCH":
            cf["match_count"] += 1

        for child in node.children:
            self._analyze_control_flow(child, cf, cf_map, loop_depth)

    def _extract_regex_fallback(self, code: str, language: str) -> PatternSignature:
        """Fallback pattern extraction using regex - multi-language aware."""
        sig = PatternSignature(language=language)

        # Language-aware regex patterns for common constructs
        loop_patterns = {
            "python": r'\b(for|while)\b',
            "javascript": r'\b(for|while|do)\b',
            "go": r'\bfor\b',
            "rust": r'\b(for|while|loop)\b',
            "java": r'\b(for|while|do)\b',
            "c": r'\b(for|while|do)\b',
            "cpp": r'\b(for|while|do)\b',
            "ruby": r'\b(for|while|until|each|loop)\b',
        }

        try_patterns = {
            "python": (r'\btry\b', r'\bexcept\b', r'\bfinally\b'),
            "javascript": (r'\btry\b', r'\bcatch\b', r'\bfinally\b'),
            "java": (r'\btry\b', r'\bcatch\b', r'\bfinally\b'),
            "cpp": (r'\btry\b', r'\bcatch\b', None),
            "ruby": (r'\bbegin\b', r'\brescue\b', r'\bensure\b'),
            "go": (None, None, r'\bdefer\b'),  # Go uses defer instead of try
        }

        # Get patterns for this language, fall back to Python-style
        loop_re = loop_patterns.get(language, r'\b(for|while)\b')
        try_re, catch_re, finally_re = try_patterns.get(language, (r'\btry\b', r'\bcatch\b', r'\bfinally\b'))

        sig.control_flow = {
            "loop_count": len(re.findall(loop_re, code)) if loop_re else 0,
            "branch_count": len(re.findall(r'\bif\b', code)),
            "try_count": len(re.findall(try_re, code)) if try_re else 0,
            "has_catch": bool(re.search(catch_re, code)) if catch_re else False,
            "has_finally": bool(re.search(finally_re, code)) if finally_re else False,
            "has_resource_guard": bool(re.search(r'\b(with|using|defer)\b', code)),
            "max_loop_depth": 1 if loop_re and re.search(loop_re, code) else 0,
            "signature": "FALLBACK",
        }

        # Simple structural tokens - abstracted
        tokens = re.findall(r'\b\w+\b', code)
        for n in self.NGRAM_SIZES:
            for i in range(len(tokens) - n + 1):
                sig.structural_ngrams[tuple(tokens[i:i+n])] += 1

        return sig

