#!/usr/bin/env python3
"""
Context-aware prompt enhancer CLI.

Retrieves relevant code context from the Context-Engine MCP server and enhances
your prompts with it. Perfect for piping to LLMs or getting quick context.

Usage:
  ctx "how does hybrid search work?"
  ctx --language python "explain the caching logic"
  ctx --under scripts/ "how is the watcher implemented?"
  ctx --limit 3 "authentication onboarding flow"

Examples:
  # Enhance prompt with context
  ctx "how does the indexer work?"

  # Pipe to LLM
  ctx "fix the bug in watcher.py" | llm

  # Filter by language and path
  ctx --language python --under scripts/ "caching implementation"

Environment:
  MCP_INDEXER_URL  - MCP indexer endpoint (default: http://localhost:8003/mcp)
  CTX_LIMIT        - Default result limit (default: 5)
  CTX_CONTEXT_LINES - Context lines for snippets (default: 3)
"""

import sys
import json
import os
import argparse
import subprocess
from urllib import request
from urllib.error import HTTPError, URLError
from typing import Dict, Any, List, Optional, Tuple

# Configuration from environment
MCP_URL = os.environ.get("MCP_INDEXER_URL", "http://localhost:8003/mcp")
DEFAULT_LIMIT = int(os.environ.get("CTX_LIMIT", "5"))
DEFAULT_CONTEXT_LINES = int(os.environ.get("CTX_CONTEXT_LINES", "3"))
DEFAULT_REWRITE_TOKENS = int(os.environ.get("CTX_REWRITE_MAX_TOKENS", "400"))

# Local decoder configuration (llama.cpp server)
def resolve_decoder_url() -> str:
    """Resolve decoder endpoint, honoring USE_GPU_DECODER + overrides."""
    override = os.environ.get("DECODER_URL", "").strip()
    if override:
        base = override
    else:
        use_gpu = str(os.environ.get("USE_GPU_DECODER", "0")).strip().lower()
        if use_gpu in {"1", "true", "yes", "on"}:
            host = "host.docker.internal" if os.path.exists("/.dockerenv") else "localhost"
            base = f"http://{host}:8081"
        else:
            base = os.environ.get("LLAMACPP_URL", "http://localhost:8080").strip()
    base = base or "http://localhost:8080"
    if base.endswith("/completion"):
        return base
    return base.rstrip("/") + "/completion"


DECODER_URL = resolve_decoder_url()
DECODER_TIMEOUT = int(os.environ.get("CTX_DECODER_TIMEOUT", "300"))


# Global session ID for MCP HTTP calls
_session_id: Optional[str] = None


def parse_sse_response(text: str) -> Dict[str, Any]:
    """Parse SSE format response (event: message\\ndata: {...})."""
    for line in text.strip().split('\n'):
        if line.startswith('data: '):
            return json.loads(line[6:])
    raise ValueError("No data line found in SSE response")


def get_session_id(timeout: int = 10) -> str:
    """Initialize MCP session and return session ID."""
    global _session_id
    if _session_id:
        return _session_id

    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "ctx-cli", "version": "1.0.0"}
        }
    }

    try:
        req = request.Request(
            MCP_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        )
        with request.urlopen(req, timeout=timeout) as resp:
            session_id = resp.headers.get("mcp-session-id")
            if not session_id:
                raise RuntimeError("Server did not return session ID")
            _session_id = session_id
            return session_id
    except Exception as e:
        raise RuntimeError(f"Failed to initialize MCP session: {e}")


def call_mcp_tool(tool_name: str, params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """Call MCP tool via HTTP JSON-RPC with session management."""
    session_id = get_session_id()

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": params}
    }

    try:
        req = request.Request(
            MCP_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id
            }
        )
        with request.urlopen(req, timeout=timeout) as resp:
            response_text = resp.read().decode('utf-8')
            return parse_sse_response(response_text)
    except HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def parse_mcp_response(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse MCP response and extract the actual result.

    Supports both text and json content items from FastMCP.
    """
    if "error" in result:
        return None

    # FastMCP typically wraps results in a content array
    res = result.get("result", {})
    content = res.get("content", [])

    # Some servers may return a dict directly (no content array)
    if isinstance(res, dict) and content == [] and any(k in res for k in ("results", "answer", "total")):
        return res

    if not content:
        return None

    item = content[0] or {}

    # Prefer typed JSON content
    if isinstance(item, dict) and "json" in item:
        return item.get("json")

    # Fallback: parse text as JSON or return raw text
    text = item.get("text", "") if isinstance(item, dict) else ""
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results with enough detail for the LLM rewrite."""
    lines: List[str] = []
    for idx, hit in enumerate(results, 1):
        path = hit.get("path", "unknown")
        start = hit.get("start_line", "?")
        end = hit.get("end_line", "?")
        score = hit.get("score", 0.0)
        language = hit.get("language") or ""
        symbol = hit.get("symbol") or ""
        snippet = (hit.get("snippet") or "").strip()

        header = f"### Ref {idx}: {path}:{start}-{end} (score {score:.3f})"
        if language:
            header += f" [{language}]"
        if symbol:
            header += f" symbol=`{symbol}`"
        lines.append(header)
        if snippet:
            lang_tag = language.lower() if language else ""
            lines.append(f"```{lang_tag}\n{snippet}\n```")
        lines.append("")  # spacer
    return "\n".join(lines).strip()


def enhance_prompt(query: str, **filters) -> str:
    """Retrieve context, invoke the LLM, and return a final enhanced prompt."""
    context_text, context_note = fetch_context(query, **filters)
    rewrite_opts = filters.get("rewrite_options") or {}
    include_context = bool(filters.get("include_context", False))
    rewritten = rewrite_prompt(
        query,
        context_text,
        context_note,
        max_tokens=rewrite_opts.get("max_tokens"),
    )
    return build_final_output(rewritten, context_text, context_note, include_context)


def fetch_context(query: str, **filters) -> Tuple[str, str]:
    """Fetch repository context text plus a note describing the status."""
    params = {
        "query": query,
        "limit": filters.get("limit", DEFAULT_LIMIT),
        "include_snippet": True,
        "context_lines": filters.get("context_lines", DEFAULT_CONTEXT_LINES),
    }
    for key in ["language", "under", "path_glob", "not_glob", "kind", "symbol", "ext", "per_path"]:
        if filters.get(key):
            params[key] = filters[key]

    result = call_mcp_tool("repo_search", params)
    if "error" in result:
        return "", f"Context retrieval failed: {result['error']}"

    data = parse_mcp_response(result)
    if not data:
        return "", "Context retrieval returned no data."

    hits = data.get("results") or []
    if not hits:
        return "", "No relevant context found for the prompt."

    return format_search_results(hits), ""


def rewrite_prompt(original_prompt: str, context: str, note: str, max_tokens: Optional[int]) -> str:
    """Use the local decoder (llama.cpp) to rewrite the prompt with repository context.

    Returns ONLY the improved prompt text. Raises exception if decoder fails.
    """
    effective_context = context.strip() if context.strip() else (note or "No context available.")

    meta_prompt = f"""Task: Rewrite the user's question to be more specific by adding references to relevant files, line numbers, and function names from the context.

DO NOT answer the question. Only rewrite it to be more precise.

Context from codebase:
{effective_context}

User's original question: {original_prompt.strip()}

Rewritten question (do not answer, only rewrite):"""

    decoder_url = DECODER_URL
    payload = {
        "prompt": meta_prompt,
        "n_predict": int(max_tokens or DEFAULT_REWRITE_TOKENS),
        "temperature": 0.7,
        "stream": False,
    }

    req = request.Request(
        decoder_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=DECODER_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)

        # Extract content from llama.cpp response
        enhanced = (
            (data.get("content") if isinstance(data, dict) else None)
            or ((data.get("choices") or [{}])[0].get("content") if isinstance(data, dict) else None)
            or ((data.get("choices") or [{}])[0].get("text") if isinstance(data, dict) else None)
            or (data.get("generated_text") if isinstance(data, dict) else None)
            or (data.get("text") if isinstance(data, dict) else None)
        )
        enhanced = (enhanced or "").strip()

        if not enhanced:
            raise ValueError(f"Decoder returned empty response (stop_type={data.get('stop_type')}, tokens={data.get('tokens_predicted')})")

        return enhanced





def build_final_output(
    rewritten_prompt: str, context: str, note: str, include_context: bool
) -> str:
    """Combine LLM rewrite with optional supporting context for downstream tools."""
    improved = rewritten_prompt.strip() or "No rewrite generated."
    if not include_context:
        return improved

    context_block = context.strip() if context.strip() else (note or "No supporting context.")

    return f"""# Improved Prompt
{improved}

---

# Supporting Context
{context_block}
"""


def main():
    parser = argparse.ArgumentParser(
        description="Context-aware prompt enhancer for Context-Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ctx "how does hybrid search work?"
  ctx --cmd llm "explain the caching logic"
  ctx --cmd pbcopy --language python "fix bug in watcher"
        """
    )

    parser.add_argument("query", help="Your question or prompt")

    # Command execution
    parser.add_argument("--cmd", "-c", help="Command to pipe enhanced prompt to (e.g., llm, pbcopy)")
    parser.add_argument("--with-context", action="store_true",
                        help="Append supporting context after the improved prompt")

    # Search filters
    parser.add_argument("--language", "-l", help="Filter by language (e.g., python, typescript)")
    parser.add_argument("--under", "-u", help="Filter by path prefix (e.g., scripts/)")
    parser.add_argument("--path-glob", help="Filter by path glob pattern")
    parser.add_argument("--not-glob", help="Exclude paths matching glob pattern")
    parser.add_argument("--kind", help="Filter by symbol kind (e.g., function, class)")
    parser.add_argument("--symbol", help="Filter by symbol name")
    parser.add_argument("--ext", help="Filter by file extension")

    # Output control
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                       help=f"Max results (default: {DEFAULT_LIMIT})")
    parser.add_argument("--context-lines", type=int, default=DEFAULT_CONTEXT_LINES,
                       help=f"Context lines for snippets (default: {DEFAULT_CONTEXT_LINES})")
    parser.add_argument("--per-path", type=int,
                       help="Limit results per file (default: server setting)")
    parser.add_argument("--rewrite-max-tokens", type=int, default=DEFAULT_REWRITE_TOKENS,
                       help=f"Max tokens for LLM rewrite (default: {DEFAULT_REWRITE_TOKENS})")

    args = parser.parse_args()

    # Build filter dict
    filters = {
        "limit": args.limit,
        "context_lines": args.context_lines,
        "language": args.language,
        "under": args.under,
        "path_glob": args.path_glob,
        "not_glob": args.not_glob,
        "kind": args.kind,
        "symbol": args.symbol,
        "ext": args.ext,
        "per_path": args.per_path,
        "rewrite_options": {
            "max_tokens": args.rewrite_max_tokens,
        },
    }

    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}

    try:
        # Fetch context and rewrite with LLM
        context_text, context_note = fetch_context(args.query, **filters)
        rewritten = rewrite_prompt(args.query, context_text, context_note, max_tokens=args.rewrite_max_tokens)
        output = rewritten.strip()

        if args.cmd:
            subprocess.run(args.cmd, input=output.encode("utf-8"), shell=True, check=False)
        else:
            print(output)

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
