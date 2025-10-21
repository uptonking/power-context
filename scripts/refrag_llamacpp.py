"""
Feature-flagged adapter for decoder-side ReFRAG using llama.cpp.

Notes:
- This file is intentionally lightweight and safe: it does nothing unless
  REFRAG_DECODER=1 (or true/yes/on).
- Actual soft-embedding / KV injection requires a patched llama.cpp build;
  wiring will be added behind this adapter once the runtime is available.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional


def _bool_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_decoder_enabled() -> bool:
    return _bool_env("REFRAG_DECODER", "0")


def get_runtime_kind() -> str:
    return str(os.environ.get("REFRAG_RUNTIME", "llamacpp")).strip().lower()


def get_phi_path() -> str:
    return str(os.environ.get("REFRAG_PHI_PATH", "")).strip()


def get_encoder_model() -> str:
    return str(os.environ.get("REFRAG_ENCODER_MODEL", "BAAI/bge-base-en-v1.5")).strip()


def get_sense_policy() -> str:
    return str(os.environ.get("REFRAG_SENSE", "heuristic")).strip().lower()


class LlamaCppRefragClient:
    """Feature-flagged client for llama.cpp decoder.

    Modes:
      - prompt (default): builds a compressed textual prompt from merged spans.
      - soft: reserved for patched server with soft-embedding support (NotImplemented).
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or os.environ.get(
            "LLAMACPP_URL", "http://localhost:8080"
        )
        if get_runtime_kind() != "llamacpp":
            raise ValueError(
                "REFRAG_RUNTIME must be 'llamacpp' for LlamaCppRefragClient"
            )

    def _post(self, path: str, json_payload: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        from urllib import request

        req = request.Request(self.base_url.rstrip("/") + path, method="POST")
        req.add_header("Content-Type", "application/json")
        data = _json.dumps(json_payload).encode("utf-8")
        import os as _os
        _timeout = float(_os.environ.get("LLAMACPP_TIMEOUT_SEC", "60") or 60)
        with request.urlopen(req, data=data, timeout=_timeout) as resp:
            body = resp.read()
        return _json.loads(body.decode("utf-8"))

    def generate_with_soft_embeddings(
        self,
        prompt: str,
        soft_embeddings: Optional[list[list[float]]] = None,
        max_tokens: int = 256,
        **gen_kwargs: Any,
    ) -> str:
        if not is_decoder_enabled():
            raise RuntimeError("Decoder path disabled: set REFRAG_DECODER=1 to enable")
        mode = os.environ.get("REFRAG_DECODER_MODE", "prompt").strip().lower()
        if mode == "soft":
            if not soft_embeddings:
                raise ValueError("soft mode requires soft_embeddings")
            payload = {
                "prompt": prompt,
                "soft_embeddings": soft_embeddings,
                "scale": float(os.environ.get("REFRAG_SOFT_SCALE", "1.0") or 1.0),
                "n_predict": int(gen_kwargs.get("max_tokens", max_tokens)),
                "temperature": float(gen_kwargs.get("temperature", 0.2)),
                "top_k": int(gen_kwargs.get("top_k", 40)),
                "top_p": float(gen_kwargs.get("top_p", 0.95)),
                "stop": gen_kwargs.get("stop") or [],
            }
            try:
                res = self._post("/soft_completion", payload)
            except Exception as e:
                raise RuntimeError(f"llama.cpp soft_completion failed: {e}")
            return (res.get("content") or res.get("generation") or "").strip()
        # Prompt mode: send a normal completion request using a compressed prompt
        payload = {
            "prompt": prompt,
            "n_predict": int(gen_kwargs.get("max_tokens", max_tokens)),
            # fast, deterministic-ish defaults; callers can override via gen_kwargs
            "temperature": float(gen_kwargs.get("temperature", 0.2)),
            "top_k": int(gen_kwargs.get("top_k", 40)),
            "top_p": float(gen_kwargs.get("top_p", 0.95)),
            "stop": gen_kwargs.get("stop") or [],
        }
        try:
            res = self._post("/completion", payload)
        except Exception as e:
            raise RuntimeError(f"llama.cpp completion failed: {e}")
        # llama.cpp server returns { 'content': '...' } or { 'token': ... } streams; we expect non-stream
        txt = (res.get("content") or res.get("generation") or "").strip()
        return txt
