"""
Feature-flagged adapter for decoder-side ReFRAG using GLM (ZhipuAI).

Safe defaults:
- Only used when REFRAG_DECODER=1 and REFRAG_RUNTIME=glm
- Requires GLM_API_KEY; optionally configure GLM_MODEL
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

            response = self.client.chat.completions.create(**create_kwargs)
            msg = response.choices[0].message
            # GLM-4.6 uses reasoning_content for thinking models
            content = getattr(msg, 'reasoning_content', None) or msg.content or ""
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"GLM completion failed: {e}")

