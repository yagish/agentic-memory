"""LLM-backed working-memory extraction helpers and eval utilities."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from memory.contracts import ExtractedWorkingMemory
from memory.llm.inference import (
    GenerationRequest,
    GenerationResult,
    InferenceError,
    generate_text,
    parse_json_payload,
)


_WORKING_PROMPT_VERSION = "working-memory-v1"
_WORKING_LOG_PATH = os.path.expanduser("~/.memory/working_memory.log")

GenerationFn = Callable[[GenerationRequest], GenerationResult]


def _should_write_log_file() -> bool:
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    return True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_working_memory_event(event: str, **payload) -> None:
    try:
        if not _should_write_log_file():
            return
        os.makedirs(os.path.dirname(_WORKING_LOG_PATH), exist_ok=True)
        record = {"timestamp": _utc_now(), "event": event, **payload}
        with open(_WORKING_LOG_PATH, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _with_strict_json_retry(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Reminder: return only one JSON object or {}. Do not use markdown fences. "
        "Preserve concrete branch names, env vars, flags, commands, file paths, job names, URLs, and stage names when central. "
        "If there is no active temporary context that would help resume the current work, return {}."
    )


def build_working_memory_extraction_prompt(session_text: str) -> str:
    transcript = session_text.strip()
    return f'''You extract one working-memory snapshot from a conversation transcript.

Prompt version: {_WORKING_PROMPT_VERSION}

A valid working memory is the current temporary active context that would help resume the work in the same session or the next immediate handoff.

It is for:
- the current goal
- the current focus
- active unfinished tasks
- active constraints or blockers
- the next concrete step
- the current status

It is not for:
- durable user/profile facts
- durable project procedures or runbooks
- a past event summary with no active work left
- fully completed conversations with no next step

Return exactly one JSON object with this shape when active working context exists:
{{
  "current_goal": "short statement of the current objective",
  "current_focus": "what part is currently being worked on",
  "active_tasks": ["unfinished task 1", "unfinished task 2"],
  "constraints": ["active blocker or constraint"],
  "next_step": "the next concrete thing to do",
  "status": "in_progress",
  "confidence": 0.0,
  "source_quote": "optional supporting quote"
}}

If the transcript does not contain useful active working context, return exactly:
{{}}

Allowed status values:
- in_progress
- blocked
- ready_to_resume
- done

Rules:
- Use the full transcript, including user and assistant turns.
- Focus on what is still active right now.
- Preserve exact literals when central: env vars, flags, branch names, commands, file names, URLs, stage names, model names, rollout percentages, and dates.
- current_goal should be the overall task, not a generic summary.
- current_focus should be the specific slice currently in motion.
- active_tasks should list only unfinished work items.
- constraints should list active blockers, dependencies, deadlines, or guardrails that still matter.
- next_step must be concrete and actionable.
- Use done only when the transcript clearly says the work is wrapped up and no active continuation is needed; otherwise prefer ready_to_resume or in_progress.
- Return empty arrays when a list has no items.
- Never invent blockers, tasks, or next steps.
- Never copy from examples unless the transcript contains those details.
- Return raw JSON only.

Example 1:
Transcript:
User: We moved token validation into shared auth middleware.
Assistant: The redirect loop stopped reproducing locally.
User: Next I need regression tests for refresh-token and expired-session flows.

Output:
{{
  "current_goal": "Finish the auth middleware refactor",
  "current_focus": "Regression coverage for refresh-token and expired-session flows after moving token validation into shared auth middleware",
  "active_tasks": ["Add regression tests for refresh-token flows", "Add regression tests for expired-session flows"],
  "constraints": [],
  "next_step": "Write the regression tests for refresh-token and expired-session flows",
  "status": "ready_to_resume",
  "confidence": 0.93
}}

Example 2:
Transcript:
User: I can finish the billing export after I get BILLING_DB_URL for staging.
Assistant: Until that lands, keep the export parser branch at fix/billing-export and do not deploy.
User: Once the secret arrives, rerun python3 scripts/export_smoke.py --env staging.

Output:
{{
  "current_goal": "Finish the billing export validation",
  "current_focus": "Waiting on the staging BILLING_DB_URL secret while keeping the parser branch unchanged",
  "active_tasks": ["Get BILLING_DB_URL for staging", "Rerun python3 scripts/export_smoke.py --env staging"],
  "constraints": ["Do not deploy until the secret arrives", "Keep branch fix/billing-export in place"],
  "next_step": "Get BILLING_DB_URL for staging and rerun python3 scripts/export_smoke.py --env staging",
  "status": "blocked",
  "confidence": 0.92
}}

Example 3:
Transcript:
User: We shipped the release and all smoke tests passed.
Assistant: Great, nothing else is queued for this change.

Output:
{{}}

Transcript:
{transcript}

Return only the JSON object.
'''


def parse_extracted_working_memory(raw_output: str) -> ExtractedWorkingMemory | None:
    payload = parse_json_payload(raw_output)
    if payload in ({}, []):
        return None
    if not isinstance(payload, dict):
        raise InferenceError("Working-memory extractor did not return a JSON object")

    try:
        return ExtractedWorkingMemory.model_validate(payload)
    except ValidationError as exc:
        raise InferenceError(f"Invalid working memory: {exc}") from exc


def extract_working_memory_from_session_text(
    session_text: str,
    *,
    model: str | None = None,
    generate_fn: GenerationFn = generate_text,
    timeout_seconds: int = 180,
    source: str = "working_memory_extractor",
    session_id: str | None = None,
) -> ExtractedWorkingMemory | None:
    prompt = build_working_memory_extraction_prompt(session_text)
    last_error: Exception | None = None

    log_working_memory_event(
        "extract_start",
        source=source,
        session_id=session_id,
        session_text=session_text,
        prompt=prompt,
    )

    for attempt_number, attempt_prompt in enumerate((prompt, _with_strict_json_retry(prompt)), start=1):
        try:
            result = generate_fn(
                GenerationRequest(
                    prompt=attempt_prompt,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    temperature=0.0,
                )
            )
            working_memory = parse_extracted_working_memory(result.text)
            log_working_memory_event(
                "extract_result",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                raw_model_output=result.text,
                validated_object=working_memory.model_dump(mode="json") if working_memory else None,
                model=result.model,
            )
            return working_memory
        except Exception as exc:
            last_error = exc
            log_working_memory_event(
                "validation_error",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                prompt=attempt_prompt,
                error=repr(exc),
            )

    if isinstance(last_error, Exception):
        raise last_error
    raise InferenceError("Working-memory extraction failed without an error")


def working_memory_to_semantic_core(memory: ExtractedWorkingMemory | dict | None) -> dict[str, object] | None:
    if memory is None:
        return None
    if isinstance(memory, ExtractedWorkingMemory):
        return {
            "current_goal": memory.current_goal,
            "current_focus": memory.current_focus,
            "active_tasks": list(memory.active_tasks),
            "constraints": list(memory.constraints),
            "next_step": memory.next_step,
            "status": memory.status,
        }
    return {
        "current_goal": str(memory.get("current_goal", "")).strip(),
        "current_focus": str(memory.get("current_focus", "")).strip(),
        "active_tasks": [str(item).strip() for item in memory.get("active_tasks", []) if str(item).strip()],
        "constraints": [str(item).strip() for item in memory.get("constraints", []) if str(item).strip()],
        "next_step": str(memory.get("next_step", "")).strip(),
        "status": str(memory.get("status", "")).strip(),
    }


def normalized_contains(text: str, snippet: str) -> bool:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))
    haystack = normalize(text)
    needle = normalize(snippet)
    return bool(needle) and needle in haystack


def discover_working_memory_fixture_cases(fixtures_root: str | Path) -> list[str]:
    root = Path(fixtures_root)
    sessions_dir = root / "sessions"
    expected_dir = root / "expected"

    cases: list[str] = []
    for session_file in sorted(sessions_dir.glob("*.txt")):
        case_name = session_file.stem
        if (expected_dir / f"{case_name}.json").exists():
            cases.append(case_name)
    return cases
