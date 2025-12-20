"""
Feature-flagged adapter for decoder-side ReFRAG using GLM (ZhipuAI).

Safe defaults:
- Only used when REFRAG_DECODER=1 and REFRAG_RUNTIME=glm
- Requires GLM_API_KEY; optionally configure GLM_MODEL and GLM_MODEL_FAST

Model selection:
- GLM_MODEL: Used for context_answer (default: glm-4.6)
- GLM_MODEL_FAST: Used for expand_query/simple tasks when disable_thinking=True (default: glm-4.5)
"""
from __future__ import annotations
import os
from typing import Any, Optional


class GLMRefragClient:
    """GLM client exposing generate_with_soft_embeddings(prompt, ...).

    Notes:
    - soft_embeddings are ignored (GLM does not support KV/soft-embed injection)
    - prompt-mode only; mirrors llama.cpp adapter surface
    - Uses OpenAI SDK with custom base_url for GLM API
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("GLM_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError("GLM_API_KEY is required when using REFRAG_RUNTIME=glm")
        self.base_url = base_url or os.environ.get("GLM_API_BASE", "https://api.z.ai/api/paas/v4/")
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_with_soft_embeddings(
        self,
        prompt: str,
        soft_embeddings: Optional[list[list[float]]] = None,  # unused
        max_tokens: int = 256,
        **gen_kwargs: Any,
    ) -> str:
        # Use fast model for simple tasks (expand_query), full model for context_answer
        disable_thinking = bool(gen_kwargs.pop("disable_thinking", False))
        if disable_thinking:
            model = os.environ.get("GLM_MODEL_FAST", "glm-4.5")
        else:
            model = os.environ.get("GLM_MODEL", "glm-4.6")
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
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(gen_kwargs.get("max_tokens", max_tokens)),
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
            # GLM-4.6 thinking models: disable deep thinking for simple JSON tasks
            if disable_thinking:
                create_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            response = self.client.chat.completions.create(**create_kwargs)
            msg = response.choices[0].message
            # GLM models may use either content or reasoning_content
            content = msg.content or ""
            # Fallback to reasoning_content if content is empty (some GLM models)
            if not content.strip():
                content = getattr(msg, 'reasoning_content', None) or ""
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"GLM completion failed: {e}")

    async def generate_batch_async(
        self,
        prompts: list[str],
        max_tokens: int = 96,
        concurrency: int = 4,
        **gen_kwargs: Any,
    ) -> list[str]:
        """Run multiple prompts concurrently using asyncio + ThreadPoolExecutor.

        Args:
            prompts: List of prompts to process
            max_tokens: Max tokens per response
            concurrency: Number of concurrent requests (default 4)
            **gen_kwargs: Additional args passed to generate_with_soft_embeddings

        Returns:
            List of responses in same order as prompts
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        if not prompts:
            return []

        # Always use fast model for batch operations (indexing)
        gen_kwargs["disable_thinking"] = True
        gen_kwargs["max_tokens"] = max_tokens

        async def run_one(prompt: str) -> str:
            loop = asyncio.get_event_loop()
            try:
                return await loop.run_in_executor(
                    executor,
                    lambda: self.generate_with_soft_embeddings(prompt, **gen_kwargs)
                )
            except Exception:
                return ""

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = await asyncio.gather(*[run_one(p) for p in prompts])

        return list(results)


def generate_pseudo_tags_batch(
    texts: list[str],
    concurrency: int = 4,
) -> list[tuple[str, list[str]]]:
    """Batch generate pseudo+tags for multiple code chunks concurrently.

    Args:
        texts: List of code snippets to process
        concurrency: Number of concurrent GLM calls (default 4)

    Returns:
        List of (pseudo, tags) tuples in same order as input texts
    """
    import asyncio
    import json as _json

    if not texts:
        return []

    # Build prompts
    prompts = []
    for text in texts:
        prompt = (
            "You are a JSON-only function that labels code spans for search enrichment.\n"
            "Respond with a single JSON object and nothing else (no prose, no markdown).\n"
            "Exact format: {\"pseudo\": string (<=20 tokens), \"tags\": [3-6 short strings]}.\n"
            "Code:\n" + text[:2000]
        )
        prompts.append(prompt)

    # Run batch
    client = GLMRefragClient()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    results = loop.run_until_complete(
        client.generate_batch_async(
            prompts,
            max_tokens=int(os.environ.get("PSEUDO_MAX_TOKENS", "96") or 96),
            concurrency=concurrency,
            temperature=float(os.environ.get("PSEUDO_TEMPERATURE", "0.10") or 0.10),
            top_p=float(os.environ.get("PSEUDO_TOP_P", "0.9") or 0.9),
            stop=["\n\n"],
            force_json=True,
        )
    )

    # Parse JSON responses
    parsed: list[tuple[str, list[str]]] = []
    for out in results:
        pseudo, tags = "", []
        try:
            obj = _json.loads(out)
            if isinstance(obj, dict):
                p = obj.get("pseudo")
                t = obj.get("tags")
                if isinstance(p, str):
                    pseudo = p.strip()[:256]
                if isinstance(t, list):
                    tags = [str(x).strip() for x in t if str(x).strip()][:6]
        except Exception:
            pass
        parsed.append((pseudo, tags))

    return parsed
