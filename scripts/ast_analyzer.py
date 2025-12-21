#!/usr/bin/env python3
"""
Advanced AST-Based Code Understanding

Implements sophisticated code analysis using Abstract Syntax Trees (AST) for:
- Semantic-aware chunking (preserve function/class boundaries)
- Call graph extraction
- Import dependency analysis
- Type inference hints
- Cross-reference tracking
"""

import os
import re
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger("ast_analyzer")

# Optional tree-sitter support - tree-sitter 0.25+ API
_TS_LANGUAGES: Dict[str, Any] = {}
_TS_AVAILABLE = False
try:
    from tree_sitter import Parser, Language

    def _load_ts_language(mod: Any, *, preferred: list[str] | None = None) -> Any | None:
        """Return a tree-sitter Language instance from a per-language package.

        Different packages expose different entrypoints (e.g. language(),
        language_typescript(), language_tsx()).
        """
        preferred = preferred or []
        candidates: list[Any] = []
        if getattr(mod, "language", None) is not None and callable(getattr(mod, "language")):
            candidates.append(getattr(mod, "language"))
        for name in preferred:
            fn = getattr(mod, name, None)
            if fn is not None and callable(fn):
                candidates.append(fn)
        # Last resort: scan for any callable language* attribute
        for name in dir(mod):
            if not name.startswith("language"):
                continue
            fn = getattr(mod, name, None)
            if fn is not None and callable(fn):
                candidates.append(fn)

        for fn in candidates:
            try:
                raw_lang = fn()
                return raw_lang if isinstance(raw_lang, Language) else Language(raw_lang)
            except Exception:
                continue
        return None

    # Import all available language packages
    for lang_name, pkg_name in [
        ("python", "tree_sitter_python"),
        ("javascript", "tree_sitter_javascript"),
        ("typescript", "tree_sitter_typescript"),
        ("go", "tree_sitter_go"),
        ("rust", "tree_sitter_rust"),
        ("java", "tree_sitter_java"),
        ("c", "tree_sitter_c"),
        ("cpp", "tree_sitter_cpp"),
        ("ruby", "tree_sitter_ruby"),
        ("c_sharp", "tree_sitter_c_sharp"),
        ("bash", "tree_sitter_bash"),
        ("json", "tree_sitter_json"),
        ("yaml", "tree_sitter_yaml"),
        ("html", "tree_sitter_html"),
        ("css", "tree_sitter_css"),
        ("markdown", "tree_sitter_markdown"),
    ]:
        try:
            mod = __import__(pkg_name)
            preferred: list[str] = []
            if lang_name == "typescript":
                preferred = ["language_typescript"]
            elif lang_name == "c_sharp":
                preferred = ["language_c_sharp", "language_csharp"]
            lang = _load_ts_language(mod, preferred=preferred)
            if lang is not None:
                _TS_LANGUAGES[lang_name] = lang
                # Also load TSX if provided by the typescript package
                if lang_name == "typescript":
                    tsx_lang = _load_ts_language(mod, preferred=["language_tsx"])
                    if tsx_lang is not None:
                        _TS_LANGUAGES["tsx"] = tsx_lang
        except Exception:
            pass  # Language package not installed

    # Add aliases
    if "javascript" in _TS_LANGUAGES:
        _TS_LANGUAGES["jsx"] = _TS_LANGUAGES["javascript"]
    if "c_sharp" in _TS_LANGUAGES:
        _TS_LANGUAGES["csharp"] = _TS_LANGUAGES["c_sharp"]
    if "bash" in _TS_LANGUAGES:
        _TS_LANGUAGES["shell"] = _TS_LANGUAGES["bash"]
        _TS_LANGUAGES["sh"] = _TS_LANGUAGES["bash"]

    _TS_AVAILABLE = len(_TS_LANGUAGES) > 0
except Exception:
    Parser = None
    Language = None
    _TS_LANGUAGES = {}
    _TS_AVAILABLE = False


@dataclass
class CodeSymbol:
    """Represents a code symbol (function, class, method, etc)."""
    name: str
    kind: str  # function, class, method, interface, etc.
    start_line: int
    end_line: int
    path: Optional[str] = None  # Fully qualified path (e.g., "MyClass.method")
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent: Optional[str] = None  # Parent class/module
    complexity: int = 0  # Cyclomatic complexity estimate
    content_hash: Optional[str] = None


@dataclass
class CallReference:
    """Represents a function/method call."""
    caller: str  # Who is calling
    callee: str  # What is being called
    line: int
    context: str  # e.g., "function", "method", "module"


@dataclass
class ImportReference:
    """Represents an import statement."""
    module: str
    names: List[str]  # Specific imports (empty if import *)
    line: int
    alias: Optional[str] = None
    is_from: bool = False


@dataclass
class CodeContext:
    """Complete context for a code chunk."""
    chunk_text: str
    start_line: int
    end_line: int
    symbols: List[CodeSymbol]
    imports: List[ImportReference]
    calls: List[CallReference]
    dependencies: Set[str]  # Modules/files this depends on
    is_semantic_unit: bool = True  # True if chunk respects boundaries


class ASTAnalyzer:
    """
    Advanced AST-based code analyzer for semantic understanding.
    
    Features:
    - Language-aware symbol extraction
    - Call graph construction
    - Dependency tracking
    - Semantic chunking (preserve boundaries)
    - Cross-reference analysis
    """
    
    def __init__(self, use_tree_sitter: bool = True):
        """
        Initialize AST analyzer.
        
        Args:
            use_tree_sitter: Use tree-sitter when available (fallback to ast module)
        """
        self.use_tree_sitter = use_tree_sitter and _TS_AVAILABLE
        self._parsers: Dict[str, Any] = {}
        
        # Language support matrix
        self.supported_languages = {
            "python": {"ast": True, "tree_sitter": True},
            "javascript": {"ast": False, "tree_sitter": True},
            "typescript": {"ast": False, "tree_sitter": True},
            "java": {"ast": False, "tree_sitter": False},
            "go": {"ast": False, "tree_sitter": False},
            "rust": {"ast": False, "tree_sitter": False},
            "c": {"ast": False, "tree_sitter": False},
            "cpp": {"ast": False, "tree_sitter": False},
        }
        
        logger.info(f"ASTAnalyzer initialized: tree_sitter={self.use_tree_sitter}")
    
    def analyze_file(
        self, file_path: str, language: str, content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a source file and extract semantic information.
        
        Args:
            file_path: Path to the file
            language: Programming language
            content: Optional file content (if not provided, read from file)
        
        Returns:
            Dict with symbols, imports, calls, and dependencies
        """
        if content is None:
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return self._empty_analysis()
        
        # Route to appropriate analyzer
        if language == "python":
            return self._analyze_python(content, file_path)
        elif language in ("javascript", "typescript") and self.use_tree_sitter:
            return self._analyze_js_ts(content, file_path, language)
        else:
            # Fallback to regex-based analysis
            return self._analyze_generic(content, file_path, language)
    
    def extract_symbols_with_context(
        self, file_path: str, language: str, content: Optional[str] = None
    ) -> List[CodeSymbol]:
        """
        Extract code symbols with full context (docstrings, signatures, etc).
        
        Returns:
            List of CodeSymbol objects with rich metadata
        """
        analysis = self.analyze_file(file_path, language, content)
        return analysis.get("symbols", [])
    
    def chunk_semantic(
        self,
        content: str,
        language: str,
        max_lines: int = 120,
        overlap_lines: int = 20,
        preserve_boundaries: bool = True
    ) -> List[CodeContext]:
        """
        Chunk code semantically, respecting function/class boundaries.
        
        Args:
            content: Source code content
            language: Programming language
            max_lines: Maximum lines per chunk
            overlap_lines: Overlap between chunks
            preserve_boundaries: Try to keep complete functions/classes together
        
        Returns:
            List of CodeContext objects with semantic chunks
        """
        if not preserve_boundaries:
            # Fall back to line-based chunking
            return self._chunk_lines_simple(content, max_lines, overlap_lines)
        
        # Extract symbols
        analysis = self.analyze_file("", language, content)
        symbols = analysis.get("symbols", [])
        
        if not symbols:
            # No symbols found, use line-based
            return self._chunk_lines_simple(content, max_lines, overlap_lines)
        
        lines = content.splitlines()
        chunks = []
        
        # Sort symbols by start line
        symbols.sort(key=lambda s: s.start_line)
        
        i = 0
        while i < len(symbols):
            symbol = symbols[i]
            
            # Calculate chunk extent
            chunk_start = symbol.start_line
            chunk_end = symbol.end_line
            symbols_in_chunk = [symbol]
            
            # Try to include adjacent small symbols
            j = i + 1
            while j < len(symbols):
                next_symbol = symbols[j]
                potential_end = next_symbol.end_line
                
                # Check if adding next symbol exceeds max_lines
                if potential_end - chunk_start > max_lines:
                    break
                
                # Check if next symbol is close enough (within overlap)
                if next_symbol.start_line - chunk_end > overlap_lines:
                    break
                
                # Include this symbol
                chunk_end = potential_end
                symbols_in_chunk.append(next_symbol)
                j += 1
            
            # Create chunk
            chunk_lines = lines[chunk_start - 1:chunk_end]
            chunk_text = "\n".join(chunk_lines)
            
            # Extract chunk-specific imports and calls
            chunk_imports = [
                imp for imp in analysis.get("imports", [])
                if chunk_start <= imp.line <= chunk_end
            ]
            chunk_calls = [
                call for call in analysis.get("calls", [])
                if chunk_start <= call.line <= chunk_end
            ]
            
            context = CodeContext(
                chunk_text=chunk_text,
                start_line=chunk_start,
                end_line=chunk_end,
                symbols=symbols_in_chunk,
                imports=chunk_imports,
                calls=chunk_calls,
                dependencies=self._extract_dependencies(chunk_imports, chunk_calls),
                is_semantic_unit=True
            )
            
            chunks.append(context)
            i = j if j > i else i + 1
        
        # Handle code not covered by symbols (module-level code, etc)
        self._fill_gaps(chunks, lines, max_lines, overlap_lines, analysis)
        
        return chunks
    
    def build_call_graph(self, file_path: str, language: str) -> Dict[str, List[str]]:
        """
        Build call graph: mapping of caller -> list of callees.
        
        Returns:
            Dict mapping function names to list of functions they call
        """
        analysis = self.analyze_file(file_path, language)
        
        call_graph = defaultdict(list)
        for call in analysis.get("calls", []):
            call_graph[call.caller].append(call.callee)
        
        return dict(call_graph)
    
    def extract_dependencies(
        self, file_path: str, language: str
    ) -> Dict[str, List[str]]:
        """
        Extract file dependencies (imports, includes).
        
        Returns:
            Dict with 'modules' (external) and 'local' (same project) imports
        """
        analysis = self.analyze_file(file_path, language)
        imports = analysis.get("imports", [])
        
        modules = []
        local = []
        
        for imp in imports:
            # Simple heuristic: relative imports or without dots are likely local
            if imp.module.startswith(".") or "/" in imp.module:
                local.append(imp.module)
            else:
                modules.append(imp.module)
        
        return {
            "modules": list(set(modules)),
            "local": list(set(local))
        }
    
    # ---- Python-specific analysis (using ast module) ----
    
    def _analyze_python(self, content: str, file_path: str) -> Dict[str, Any]:
        """Analyze Python code using ast module."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Python syntax error in {file_path}: {e}")
            return self._empty_analysis()
        
        symbols = []
        imports = []
        calls = []
        
        # Extract symbols
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = self._extract_python_function(node, content)
                symbols.append(symbol)
            elif isinstance(node, ast.ClassDef):
                symbol = self._extract_python_class(node, content)
                symbols.append(symbol)
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportReference(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        line=node.lineno,
                        is_from=False
                    ))
            elif isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
                imports.append(ImportReference(
                    module=node.module or "",
                    names=names,
                    alias=None,
                    line=node.lineno,
                    is_from=True
                ))
        
        # Extract calls (simplified)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callee = self._get_call_name(node.func)
                if callee:
                    calls.append(CallReference(
                        caller="",  # Would need parent context
                        callee=callee,
                        line=node.lineno,
                        context="call"
                    ))
        
        return {
            "symbols": symbols,
            "imports": imports,
            "calls": calls,
            "language": "python"
        }
    
    def _extract_python_function(self, node: ast.FunctionDef, content: str) -> CodeSymbol:
        """Extract detailed function information from AST node."""
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Build signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"
        
        # Calculate complexity (simplified: count branches)
        complexity = sum(
            1 for n in ast.walk(node)
            if isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.With))
        )
        
        # Content hash
        lines = content.splitlines()
        if node.lineno <= len(lines) and node.end_lineno <= len(lines):
            func_content = "\n".join(lines[node.lineno - 1:node.end_lineno])
            content_hash = hashlib.md5(func_content.encode()).hexdigest()[:8]
        else:
            content_hash = None
        
        return CodeSymbol(
            name=node.name,
            kind="function",
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
            decorators=decorators,
            complexity=complexity,
            content_hash=content_hash
        )
    
    def _extract_python_class(self, node: ast.ClassDef, content: str) -> CodeSymbol:
        """Extract detailed class information from AST node."""
        docstring = ast.get_docstring(node)
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Get base classes
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        
        # Count methods
        methods = sum(
            1 for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        
        return CodeSymbol(
            name=node.name,
            kind="class",
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
            decorators=decorators,
            complexity=methods
        )
    
    def _get_decorator_name(self, node: ast.expr) -> str:
        """Extract decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""
    
    def _get_name(self, node: ast.expr) -> str:
        """Extract name from AST expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""
    
    def _get_call_name(self, node: ast.expr) -> str:
        """Extract function name from call node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # Return just the method name for simplicity
            return node.attr
        return ""
    
    # ---- JavaScript/TypeScript analysis (using tree-sitter) ----
    
    def _analyze_js_ts(
        self, content: str, file_path: str, language: str
    ) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript using tree-sitter."""
        ts_lang_key = language if language in _TS_LANGUAGES else "javascript"
        parser = self._get_ts_parser(ts_lang_key)
        if not parser:
            return self._empty_analysis()
        
        try:
            tree = parser.parse(content.encode("utf-8"))
            root = tree.root_node
        except Exception as e:
            logger.warning(f"Tree-sitter parse error in {file_path}: {e}")
            return self._empty_analysis()
        
        symbols = []
        imports = []
        calls = []
        
        def node_text(n):
            return content.encode("utf-8")[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")
        
        def walk(node, parent_class=None):
            node_type = node.type
            
            # Classes
            if node_type == "class_declaration":
                name_node = node.child_by_field_name("name")
                class_name = node_text(name_node) if name_node else ""
                
                symbols.append(CodeSymbol(
                    name=class_name,
                    kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1
                ))
                
                # Walk class body
                for child in node.children:
                    walk(child, parent_class=class_name)
                return
            
            # Functions
            if node_type in ("function_declaration", "arrow_function", "function_expression"):
                name_node = node.child_by_field_name("name")
                func_name = node_text(name_node) if name_node else "<anonymous>"
                
                symbols.append(CodeSymbol(
                    name=func_name,
                    kind="function",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent=parent_class
                ))
            
            # Methods
            if node_type == "method_definition":
                name_node = node.child_by_field_name("name")
                method_name = node_text(name_node) if name_node else ""
                
                symbols.append(CodeSymbol(
                    name=method_name,
                    kind="method",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent=parent_class,
                    path=f"{parent_class}.{method_name}" if parent_class else method_name
                ))
            
            # Imports
            if node_type == "import_statement":
                source = node.child_by_field_name("source")
                if source:
                    module = node_text(source).strip('"\'')
                    imports.append(ImportReference(
                        module=module,
                        names=[],
                        line=node.start_point[0] + 1,
                        is_from=True
                    ))
            
            # Recurse
            for child in node.children:
                walk(child, parent_class)
        
        walk(root)
        
        return {
            "symbols": symbols,
            "imports": imports,
            "calls": calls,
            "language": language
        }
    
    def _get_ts_parser(self, language: str):
        """Get or create tree-sitter parser for language.

        Uses tree-sitter 0.25+ API with pre-loaded Language objects.
        """
        if language in self._parsers:
            return self._parsers[language]

        if not _TS_AVAILABLE or language not in _TS_LANGUAGES:
            return None

        try:
            lang = _TS_LANGUAGES[language]
            parser = Parser(lang)
            self._parsers[language] = parser
            return parser
        except Exception as e:
            logger.warning(f"Failed to create tree-sitter parser for {language}: {e}")
            return None
    
    # ---- Generic/fallback analysis ----
    
    def _analyze_generic(
        self, content: str, file_path: str, language: str
    ) -> Dict[str, Any]:
        """Fallback regex-based analysis for unsupported languages."""
        symbols = []
        lines = content.splitlines()
        
        # Very basic heuristics
        for i, line in enumerate(lines, 1):
            # Try to find function-like patterns
            if re.match(r'^\s*(def|function|func|fn)\s+(\w+)', line):
                match = re.match(r'^\s*(?:def|function|func|fn)\s+(\w+)', line)
                if match:
                    symbols.append(CodeSymbol(
                        name=match.group(1),
                        kind="function",
                        start_line=i,
                        end_line=i  # Can't determine without parsing
                    ))
            
            # Try to find class-like patterns
            if re.match(r'^\s*class\s+(\w+)', line):
                match = re.match(r'^\s*class\s+(\w+)', line)
                if match:
                    symbols.append(CodeSymbol(
                        name=match.group(1),
                        kind="class",
                        start_line=i,
                        end_line=i
                    ))
        
        return {
            "symbols": symbols,
            "imports": [],
            "calls": [],
            "language": language
        }
    
    # ---- Helper methods ----
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result."""
        return {
            "symbols": [],
            "imports": [],
            "calls": [],
            "language": "unknown"
        }
    
    def _chunk_lines_simple(
        self, content: str, max_lines: int, overlap: int
    ) -> List[CodeContext]:
        """Simple line-based chunking fallback."""
        lines = content.splitlines()
        chunks = []
        
        i = 0
        while i < len(lines):
            chunk_end = min(i + max_lines, len(lines))
            chunk_lines = lines[i:chunk_end]
            
            chunks.append(CodeContext(
                chunk_text="\n".join(chunk_lines),
                start_line=i + 1,
                end_line=chunk_end,
                symbols=[],
                imports=[],
                calls=[],
                dependencies=set(),
                is_semantic_unit=False
            ))
            
            i = chunk_end - overlap if chunk_end < len(lines) else chunk_end
        
        return chunks
    
    def _fill_gaps(
        self,
        chunks: List[CodeContext],
        lines: List[str],
        max_lines: int,
        overlap: int,
        analysis: Dict[str, Any]
    ):
        """Fill gaps between symbol chunks with module-level code."""
        if not chunks:
            return
        
        # Find uncovered regions
        covered = set()
        for chunk in chunks:
            covered.update(range(chunk.start_line, chunk.end_line + 1))
        
        gaps = []
        gap_start = None
        for i in range(1, len(lines) + 1):
            if i not in covered:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gaps.append((gap_start, i - 1))
                    gap_start = None
        
        if gap_start is not None:
            gaps.append((gap_start, len(lines)))
        
        # Create chunks for gaps
        for start, end in gaps:
            if end - start + 1 < 3:  # Skip tiny gaps
                continue
            
            gap_lines = lines[start - 1:end]
            chunks.append(CodeContext(
                chunk_text="\n".join(gap_lines),
                start_line=start,
                end_line=end,
                symbols=[],
                imports=[imp for imp in analysis.get("imports", []) if start <= imp.line <= end],
                calls=[],
                dependencies=set(),
                is_semantic_unit=False
            ))
        
        # Re-sort chunks by start line
        chunks.sort(key=lambda c: c.start_line)
    
    def _extract_dependencies(
        self, imports: List[ImportReference], calls: List[CallReference]
    ) -> Set[str]:
        """Extract unique dependencies from imports and calls."""
        deps = set()
        
        for imp in imports:
            deps.add(imp.module)
            deps.update(imp.names)
        
        for call in calls:
            deps.add(call.callee)
        
        return deps


# Global analyzer instance
_analyzer: Optional[ASTAnalyzer] = None


def get_ast_analyzer(reset: bool = False) -> ASTAnalyzer:
    """Get or create global AST analyzer instance."""
    global _analyzer
    
    if _analyzer is None or reset:
        use_ts = os.environ.get("USE_TREE_SITTER", "1").lower() in {"1", "true", "yes", "on"}
        _analyzer = ASTAnalyzer(use_tree_sitter=use_ts)
    
    return _analyzer


# Convenience functions
def extract_symbols(file_path: str, language: str) -> List[CodeSymbol]:
    """Extract symbols from a file."""
    analyzer = get_ast_analyzer()
    return analyzer.extract_symbols_with_context(file_path, language)


def chunk_code_semantically(
    content: str,
    language: str,
    max_lines: int = 120,
    overlap: int = 20
) -> List[Dict[str, Any]]:
    """
    Chunk code semantically, returning simplified dicts for indexing.
    
    Returns list of dicts compatible with existing chunking interface.
    """
    analyzer = get_ast_analyzer()
    contexts = analyzer.chunk_semantic(content, language, max_lines, overlap)
    
    # Convert to simple dict format
    return [
        {
            "text": ctx.chunk_text,
            "start": ctx.start_line,
            "end": ctx.end_line,
            "is_semantic": ctx.is_semantic_unit,
            "symbols": [s.name for s in ctx.symbols],
            "symbol_types": [s.kind for s in ctx.symbols]
        }
        for ctx in contexts
    ]


if __name__ == "__main__":
    # Example usage
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    test_code = '''
import os
from typing import List, Dict

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def process(self, data: List[str]) -> List[str]:
        """Process the input data."""
        results = []
        for item in data:
            if item:
                results.append(self.transform(item))
        return results
    
    def transform(self, item: str) -> str:
        """Transform a single item."""
        return item.upper()

def main():
    """Main entry point."""
    processor = DataProcessor({})
    result = processor.process(["hello", "world"])
    print(result)

if __name__ == "__main__":
    main()
'''
    
    analyzer = get_ast_analyzer()
    
    print("=== Symbol Extraction ===")
    analysis = analyzer.analyze_file("test.py", "python", test_code)
    for symbol in analysis["symbols"]:
        print(f"{symbol.kind}: {symbol.name} (lines {symbol.start_line}-{symbol.end_line})")
        if symbol.docstring:
            print(f"  Docstring: {symbol.docstring[:50]}...")
    
    print("\n=== Imports ===")
    for imp in analysis["imports"]:
        print(f"Line {imp.line}: {imp.module} -> {imp.names}")
    
    print("\n=== Semantic Chunking ===")
    chunks = chunk_code_semantically(test_code, "python", max_lines=20)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (lines {chunk['start']}-{chunk['end']}):")
        print(f"  Symbols: {chunk['symbols']}")
        print(f"  Semantic unit: {chunk['is_semantic']}")
        print()
