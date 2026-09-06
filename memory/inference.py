"""Inference seam for text generation and embeddings.

This module is the seam between domain modules and model/runtime adapters.
It centralizes Ollama request handling and embedding calls so callers stop
open-coding model orchestration.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from memory.vectors import embed as _embed_text


_OLLAMA_URL = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = os.environ.get("MEMORY_OLLAMA_MODEL", "qwen2.5:7b")
_DEFAULT_TIMEOUT = 120


@dataclass(frozen=True)
class GenerationRequest:
    """A text-generation request for the local Ollama runtime.

    `temperature` was added so eval-like callers can pin deterministic model
    behavior instead of relying on Ollama defaults.
    """

    prompt: str
    model: str | None = None
    timeout_seconds: int = _DEFAULT_TIMEOUT
    temperature: float = 0.0


@dataclass(frozen=True)
class GenerationResult:
    """Structured text-generation response returned to callers."""

    text: str
    model: str


class InferenceError(RuntimeError):
    """Raised when a text-generation or embedding request cannot be completed."""


def generate_text(request: GenerationRequest) -> GenerationResult:
    """Generate text via the local Ollama HTTP API."""
    model = request.model or _DEFAULT_MODEL
    body = json.dumps({
        "model": model,
        "prompt": request.prompt,
        "stream": False,
        "options": {"temperature": request.temperature},
    }).encode("utf-8")

    req = urllib.request.Request(
        _OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=request.timeout_seconds) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise InferenceError(f"Ollama unavailable: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Invalid Ollama JSON response: {raw[:200]}") from exc

    if "response" not in data:
        raise InferenceError(f"Unexpected Ollama response: {raw[:200]}")

    return GenerationResult(text=data["response"], model=model)


def parse_json_payload(raw: str) -> list | dict:
    """Extract the first JSON object or array from model output.

    Local models often wrap valid JSON with short prose like
    "Here are the facts:". This helper tolerates that wrapper text while still
    returning only a parsed JSON object/array to callers.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(raw):
            # Scan forward until a plausible JSON payload starts.
            if char not in "[{":
                continue
            try:
                payload, _end = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, (list, dict)):
                return payload
    return []


def embed_text(text: str) -> list[float]:
    """Embed one text string using the configured embedding runtime."""
    try:
        return _embed_text(text)
    except Exception as exc:  # pragma: no cover - delegated behavior
        raise InferenceError(str(exc)) from exc
