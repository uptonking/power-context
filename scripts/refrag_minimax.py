"""
Feature-flagged adapter for decoder-side ReFRAG using MiniMax M2.

Safe defaults:
- Only used when REFRAG_DECODER=1 and REFRAG_RUNTIME=minimax
- Requires MINIMAX_API_KEY; optionally configure MINIMAX_MODEL

MiniMax M2 API is OpenAI-compatible at https://api.minimax.io/v1
"""
from __future__ import annotations
import os
from typing import Any, Optional


class MiniMaxRefragClient:
    """MiniMax M2 client exposing generate_with_soft_embeddings(prompt, ...).

    Notes:
    - soft_embeddings are ignored (MiniMax does not support KV/soft-embed injection)
    - prompt-mode only; mirrors llama.cpp adapter surface
    - Uses OpenAI SDK with custom base_url for MiniMax API
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("MINIMAX_API_KEY is required when using REFRAG_RUNTIME=minimax")
        self.base_url = base_url or os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io/v1")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_with_soft_embeddings(
        self,
        prompt: str,
        soft_embeddings: Optional[list[list[float]]] = None,  # unused
        max_tokens: Optional[int] = None,
        **gen_kwargs: Any,
    ) -> str:
        model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2")
        # Use DECODER_MAX_TOKENS env var; no hardcoded cap (MiniMax uses thinking tokens)
        if max_tokens is None:
            try:
                max_tokens = int(os.environ.get("DECODER_MAX_TOKENS", "4096"))
            except ValueError:
                max_tokens = 4096
        temperature = float(gen_kwargs.get("temperature", 0.2))
        top_p = float(gen_kwargs.get("top_p", 0.95))
        stop = gen_kwargs.get("stop")
        timeout = gen_kwargs.pop("timeout", None)
        # Optional hint from callers that they want strict JSON output.
        force_json = bool(gen_kwargs.pop("force_json", False))
        try:
            timeout_val = float(timeout) if timeout is not None else None
        except Exception:
            timeout_val = None

        try:
            # Prefer explicit gen_kwargs override, else use computed max_tokens
            final_max_tokens = int(gen_kwargs.get("max_tokens", max_tokens))
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": final_max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop if stop else None,
                "timeout": timeout_val,
            }
            # When explicitly requested and supported by the backend, ask for
            # JSON-only responses. If the provider rejects this parameter, the
            # API call will raise and the caller will handle the failure.
            if force_json:
                create_kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**create_kwargs)
            msg = response.choices[0].message
            content = msg.content or ""
            # MiniMax M2 may return <think>...</think> reasoning tokens; strip them
            content = self._strip_thinking_tokens(content)
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"MiniMax completion failed: {e}")

    def _strip_thinking_tokens(self, text: str) -> str:
        """Remove <think>...</think> blocks from MiniMax M2 responses."""
        import re
        # Remove <think>...</think> blocks (including partial/truncated ones)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Also handle unclosed <think> tags (truncated responses)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

