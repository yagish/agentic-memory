from __future__ import annotations

import json
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_loads(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _session_text_from_turns(turns: list[dict]) -> str:
    parts: list[str] = []
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def log_retrieval(conn, tool: str, query: str | None, result_size: int) -> None:
    """Backward-compatible no-op after retrieval logging removal."""
    del conn, tool, query, result_size
