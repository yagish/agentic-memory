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



def _normalize_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""



def _collect_fragments(item: Any, fields: tuple[str, ...]) -> list[str]:
    if not isinstance(item, dict):
        return []
    fragments: list[str] = []
    for field in fields:
        value = item.get(field)
        if isinstance(value, list):
            fragments.extend(fragment for fragment in (_normalize_fragment(v) for v in value) if fragment)
            continue
        fragment = _normalize_fragment(value)
        if fragment:
            fragments.append(fragment)
    return fragments



def estimate_recalled_context_tokens(response: dict[str, Any] | None) -> int:
    response = response or {}
    context = response.get("context") or {}
    fragments: list[str] = []

    fragments.extend(
        _collect_fragments(
            context.get("working_mem"),
            ("current_goal", "current_focus", "active_tasks", "constraints", "next_step", "status"),
        )
    )
    for item in context.get("session_memory") or []:
        fragments.extend(
            _collect_fragments(item, ("title", "summary", "what_was_tried", "outcomes", "left_off_at", "next_steps"))
        )
    for item in context.get("episodic") or []:
        fragments.extend(
            _collect_fragments(item, ("title", "abstract", "participants", "decisions", "outcomes", "follow_ups"))
        )
    for item in context.get("procedural") or []:
        fragments.extend(
            _collect_fragments(item, ("title", "summary", "steps", "trigger_phrases", "tools"))
        )
    for item in context.get("facts") or []:
        if not isinstance(item, dict):
            continue
        fact_text = _normalize_fragment(item.get("semantic_content") or item.get("content") or item.get("value"))
        if fact_text:
            fragments.append(fact_text)

    return estimate_tokens("\n".join(fragments))



def _response_memory_counts(response: dict[str, Any]) -> dict[str, int]:
    return {
        "facts_count": int(response.get("facts_count") or 0),
        "episodic_count": int(response.get("episodic_count") or 0),
        "procedural_count": int(response.get("procedural_count") or 0),
        "session_memory_count": int(response.get("session_memory_count") or 0),
        "working_memory_count": int(response.get("working_memory_count") or 0),
    }



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
        **_response_memory_counts(response),
    )



def log_memory_injection(
    component: str,
    *,
    prompt: str,
    injection: str,
    session_id: str | None = None,
    agent: str | None = None,
    response: dict[str, Any] | None = None,
) -> None:
    response = response or {}
    recalled_context_tokens = estimate_recalled_context_tokens(response)
    injection_tokens = estimate_tokens(injection)
    activity_log(
        component,
        "memory_injection",
        session=session_id,
        agent=agent,
        prompt_tokens_estimate=estimate_tokens(prompt),
        injection_tokens_estimate=injection_tokens,
        recalled_context_tokens_estimate=recalled_context_tokens,
        compression_gain_tokens_estimate=max(0, recalled_context_tokens - injection_tokens),
        **_response_memory_counts(response),
    )
