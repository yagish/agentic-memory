"""Small client and process helpers for the local Ollama server."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


def _log(log_fn: Callable[[str], None] | None, message: str) -> None:
    if log_fn is not None:
        log_fn(message)


def is_ollama_running() -> bool:
    """Return whether the local Ollama API is responding."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_ollama_if_needed(
    log_fn: Callable[[str], None] | None = None,
) -> subprocess.Popen | None:
    """Start Ollama when needed and return its process if started here."""
    if is_ollama_running():
        _log(log_fn, "ollama already running")
        return None
    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        _log(log_fn, "ollama binary not found in PATH - skipping LLM processing")
        return None
    except Exception as exc:
        _log(log_fn, f"failed to launch ollama: {exc}")
        return None

    _log(log_fn, f"started ollama (PID {proc.pid})")
    for attempt in range(12):
        time.sleep(1)
        if is_ollama_running():
            _log(log_fn, f"ollama ready after {attempt + 1}s")
            return proc

    _log(log_fn, "ollama did not become ready in 12s - killing it")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass
    return None


def stop_ollama(
    proc: subprocess.Popen,
    log_fn: Callable[[str], None] | None = None,
) -> None:
    """Stop an Ollama process started by this daemon."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
        _log(log_fn, f"stopped ollama (PID {proc.pid})")
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except Exception as exc:
        _log(log_fn, f"error stopping ollama: {exc}")


def call_ollama(
    prompt_text: str,
    log_fn: Callable[[str], None] | None = None,
) -> str:
    """Generate one non-streaming response from the local Ollama server.
    
    Args:
        prompt_text: The prompt to send to Ollama.
        log_fn: Optional logging function to record the request and response.
    """
    model = os.environ.get("MEMORY_OLLAMA_MODEL", DEFAULT_MODEL)
    
    # Log the input prompt (truncated for readability).
    if log_fn is not None:
        prompt_preview = prompt_text[:200] + "…" if len(prompt_text) > 200 else prompt_text
        _log(log_fn, f"[OLLAMA_REQUEST] model={model} prompt_len={len(prompt_text)} preview={repr(prompt_preview)}")
    
    body = json.dumps({
        "model": model,
        "prompt": prompt_text,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    data = json.loads(raw)
    if "response" not in data:
        raise RuntimeError(f"Unexpected ollama response: {raw[:200]}")
    
    response = data["response"]
    
    # Log the output response (truncated for readability).
    if log_fn is not None:
        response_preview = response[:200] + "…" if len(response) > 200 else response
        _log(log_fn, f"[OLLAMA_RESPONSE] response_len={len(response)} preview={repr(response_preview)}")
    
    return response
