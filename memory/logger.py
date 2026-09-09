"""Lightweight compatibility logger.

Older call sites still import ``activity_log`` / ``error_log``. Restore these
as small append-only file loggers so dashboard log views can surface wake-up and
save-hook activity again without reintroducing a heavyweight logging system.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any


_ACTIVITY_LOG_PATH = os.path.expanduser("~/.memory/activity.log")
_ERROR_LOG_PATH = os.path.expanduser("~/.memory/error.log")


def _should_write_log_file() -> bool:
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    return True



def _append_json_line(path: str, payload: dict) -> None:
    if not _should_write_log_file():
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass



def activity_log(component: str, action: str, **kwargs) -> None:
    _append_json_line(
        _ACTIVITY_LOG_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": "info",
            "component": component,
            "action": action,
            **kwargs,
        },
    )



def error_log(component: str, message: str, exc: BaseException | None = None) -> None:
    _append_json_line(
        _ERROR_LOG_PATH,
        {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": "error",
            "component": component,
            "message": message,
            "exception": repr(exc) if exc is not None else None,
        },
    )


def estimate_tokens(text: str | None) -> int:
    content = str(text or "").strip()
    if not content:
        return 0
    return max(1, (len(content) + 3) // 4)


def log_memory_answer(
    component: str,
    *,
    prompt: str,
    answer: str,
    session_id: str | None = None,
    agent: str | None = None,
    response: dict[str, Any] | None = None,
) -> None:
    response = response or {}
    activity_log(
        component,
        "memory_answer",
        session=session_id,
        agent=agent,
        prompt_tokens_estimate=estimate_tokens(prompt),
        answer_tokens_estimate=estimate_tokens(answer),
        tokens_saved_estimate=estimate_tokens(prompt) + estimate_tokens(answer),
        facts_count=int(response.get("facts_count") or 0),
        episodic_count=int(response.get("episodic_count") or 0),
        procedural_count=int(response.get("procedural_count") or 0),
        session_memory_count=int(response.get("session_memory_count") or 0),
        working_memory_count=int(response.get("working_memory_count") or 0),
    )


