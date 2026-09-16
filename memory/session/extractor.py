"""LLM-backed compacted-session memory extraction helpers and eval utilities."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from memory.contracts import ExtractedSessionMemory
from memory.llm.inference import (
    GenerationRequest,
    GenerationResult,
    InferenceError,
    generate_text,
    parse_json_payload,
)


_SESSION_MEMORY_PROMPT_VERSION = "session-memory-v1"
_SESSION_MEMORY_LOG_PATH = os.path.expanduser("~/.memory/session_memory.log")

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


def log_session_memory_event(event: str, **payload) -> None:
    try:
        if not _should_write_log_file():
            return
        os.makedirs(os.path.dirname(_SESSION_MEMORY_LOG_PATH), exist_ok=True)
        record = {"timestamp": _utc_now(), "event": event, **payload}
        with open(_SESSION_MEMORY_LOG_PATH, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _with_strict_json_retry(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Reminder: return only one JSON object or {}. Do not use markdown fences. "
        "Preserve exact literals like env vars, branch names, commands, flags, file names, URLs, job names, rollout percentages, and stage names when central. "
        "A valid compacted session memory should make it easy to resume the session later."
    )


def build_session_memory_extraction_prompt(session_text: str) -> str:
    transcript = session_text.strip()
    return f'''You extract one compacted session memory from a conversation transcript.

Prompt version: {_SESSION_MEMORY_PROMPT_VERSION}

A valid compacted session memory is a concise handoff for the overall session. It should help someone resume the work later without rereading the whole transcript.

It is for:
- the main task or theme of the session
- what was tried or changed
- the concrete end state or outcome
- where the work was left off
- the next session's most useful next steps

It is not for:
- durable facts about the user
- durable repeatable procedures or runbooks
- one tiny sub-event when the broader session has a clearer handoff
- transcripts with no meaningful task progress

Return exactly one JSON object with this shape when there is a real resumable session summary:
{{
  "title": "short concrete session title",
  "summary": "1-2 sentence summary of the session and current state",
  "what_was_tried": ["important attempted change or investigation"],
  "outcomes": ["completed result or validated conclusion"],
  "left_off_at": "where the work currently stands",
  "next_steps": ["useful next session step"],
  "confidence": 0.0,
  "source_quote": "optional supporting quote"
}}

If the transcript has no meaningful resumable work summary, return exactly:
{{}}

Rules:
- Use the full transcript, including user and assistant turns.
- Focus on the main session handoff, not every side remark.
- Title should be concrete and specific.
- Summary should mention the task, what changed or was learned, and the current end state.
- Preserve exact literals when central: env vars, branch names, commands, file names, flags, models, dates, rollout percentages, URLs, and stage names.
- what_was_tried should capture the most important implementation or investigation steps.
- outcomes should capture concrete validated results, fixes, failures, or conclusions.
- left_off_at should clearly say what remains unresolved or what state the work is in now.
- next_steps should list only useful concrete continuation steps.
- Return empty arrays when a list has no items.
- Never invent work, results, or next steps.
- Never copy from examples unless the transcript contains those details.
- Return raw JSON only.

Example 1:
Transcript:
User: We moved token validation into shared auth middleware.
Assistant: The redirect loop stopped reproducing locally.
User: I still need regression tests for refresh-token and expired-session flows.

Output:
{{
  "title": "Auth middleware refactor",
  "summary": "The session moved token validation into shared auth middleware and verified that the redirect loop stopped reproducing locally. The refactor is in place, but regression coverage for refresh-token and expired-session flows is still missing.",
  "what_was_tried": ["Moved token validation into shared auth middleware", "Retested the login redirect flow locally"],
  "outcomes": ["Redirect loop stopped reproducing locally"],
  "left_off_at": "The refactor is in place and the remaining gap is regression coverage for refresh-token and expired-session flows",
  "next_steps": ["Add regression tests for refresh-token flows", "Add regression tests for expired-session flows"],
  "confidence": 0.94
}}

Example 2:
Transcript:
User: I tried python3 scripts/backfill.py --job checkout_api_backfill in staging.
Assistant: The counts matched after the rerun.
User: Tomorrow watch the error-rate dashboard before we widen the rollout.

Output:
{{
  "title": "Checkout backfill validation",
  "summary": "The session reran python3 scripts/backfill.py --job checkout_api_backfill in staging and verified that the counts matched afterward. The backfill looked healthy, with one more dashboard watch still queued before widening the rollout.",
  "what_was_tried": ["Ran python3 scripts/backfill.py --job checkout_api_backfill in staging"],
  "outcomes": ["Counts matched after the rerun"],
  "left_off_at": "The backfill appears healthy, but the rollout should not widen until the dashboard watch is done",
  "next_steps": ["Watch the error-rate dashboard before widening the rollout"],
  "confidence": 0.9
}}

Example 3:
Transcript:
User: Thanks for the explanation of embeddings.
Assistant: Happy to help.

Output:
{{}}

Transcript:
{transcript}

Return only the JSON object.
'''


def parse_extracted_session_memory(raw_output: str) -> ExtractedSessionMemory | None:
    payload = parse_json_payload(raw_output)
    if payload in ({}, []):
        return None
    if not isinstance(payload, dict):
        raise InferenceError("Session-memory extractor did not return a JSON object")

    try:
        return ExtractedSessionMemory.model_validate(payload)
    except ValidationError as exc:
        raise InferenceError(f"Invalid session memory: {exc}") from exc


def extract_session_memory_from_session_text(
    session_text: str,
    *,
    model: str | None = None,
    generate_fn: GenerationFn = generate_text,
    timeout_seconds: int = 180,
    source: str = "session_memory_extractor",
    session_id: str | None = None,
) -> ExtractedSessionMemory | None:
    prompt = build_session_memory_extraction_prompt(session_text)
    last_error: Exception | None = None

    log_session_memory_event(
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
            session_memory = parse_extracted_session_memory(result.text)
            log_session_memory_event(
                "extract_result",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                raw_model_output=result.text,
                validated_object=session_memory.model_dump(mode="json") if session_memory else None,
                model=result.model,
            )
            return session_memory
        except Exception as exc:
            last_error = exc
            log_session_memory_event(
                "validation_error",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                prompt=attempt_prompt,
                error=repr(exc),
            )

    if isinstance(last_error, Exception):
        raise last_error
    raise InferenceError("Session-memory extraction failed without an error")


def session_memory_to_semantic_core(memory: ExtractedSessionMemory | dict | None) -> dict[str, object] | None:
    if memory is None:
        return None
    if isinstance(memory, ExtractedSessionMemory):
        return {
            "title": memory.title,
            "summary": memory.summary,
            "what_was_tried": list(memory.what_was_tried),
            "outcomes": list(memory.outcomes),
            "left_off_at": memory.left_off_at,
            "next_steps": list(memory.next_steps),
        }
    return {
        "title": str(memory.get("title", "")).strip(),
        "summary": str(memory.get("summary", "")).strip(),
        "what_was_tried": [str(item).strip() for item in memory.get("what_was_tried", []) if str(item).strip()],
        "outcomes": [str(item).strip() for item in memory.get("outcomes", []) if str(item).strip()],
        "left_off_at": str(memory.get("left_off_at", "")).strip(),
        "next_steps": [str(item).strip() for item in memory.get("next_steps", []) if str(item).strip()],
    }


def normalized_contains(text: str, snippet: str) -> bool:
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))
    haystack = normalize(text)
    needle = normalize(snippet)
    return bool(needle) and needle in haystack


def discover_session_memory_fixture_cases(fixtures_root: str | Path) -> list[str]:
    root = Path(fixtures_root)
    sessions_dir = root / "sessions"
    expected_dir = root / "expected"

    cases: list[str] = []
    for session_file in sorted(sessions_dir.glob("*.txt")):
        case_name = session_file.stem
        if (expected_dir / f"{case_name}.json").exists():
            cases.append(case_name)
    return cases
