"""LLM subpackage — Ollama client and inference seam."""

from memory.llm.inference import (
    GenerationRequest,
    GenerationResult,
    InferenceError,
    embed_text,
    generate_text,
    parse_json_payload,
)
from memory.llm.ollama import (
    call_ollama,
    is_ollama_running,
    start_ollama_if_needed,
    stop_ollama,
)

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "InferenceError",
    "embed_text",
    "generate_text",
    "parse_json_payload",
    "call_ollama",
    "is_ollama_running",
    "start_ollama_if_needed",
    "stop_ollama",
]
