"""LLM-backed procedural-memory extraction helpers and eval utilities."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from memory.contracts import ExtractedProcedure
from memory.inference import (
    GenerationRequest,
    GenerationResult,
    InferenceError,
    generate_text,
    parse_json_payload,
)


_PROCEDURAL_PROMPT_VERSION = "procedural-v1"
_PROCEDURAL_LOG_PATH = os.path.expanduser("~/.memory/procedural.log")

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


def log_procedural_event(event: str, **payload) -> None:
    """Append one JSON line to the dedicated procedural-memory log."""
    try:
        if not _should_write_log_file():
            return
        os.makedirs(os.path.dirname(_PROCEDURAL_LOG_PATH), exist_ok=True)
        record = {"timestamp": _utc_now(), "event": event, **payload}
        with open(_PROCEDURAL_LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _with_strict_json_retry(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Reminder: return only one JSON object. Do not use markdown fences. "
        "Return a non-empty steps array. Keep only durable repeatable procedures, not one-off tasks. "
        "Preserve literal commands, flags, file names, branch names, service names, and environments when central."
    )


def build_procedural_extraction_prompt(session_text: str) -> str:
    """Build the pinned extraction prompt for one procedural memory."""
    transcript = session_text.strip()

    return f'''You extract one procedural memory from a conversation transcript.

Prompt version: {_PROCEDURAL_PROMPT_VERSION}

A valid procedural memory is a durable repeatable how-to, workflow, checklist, or operating pattern that should help in future sessions.
A procedural memory is NOT a one-off task, status update, historical outcome, or personal profile fact.

Return exactly one JSON object with this shape:
{{
  "title": "short specific procedure title",
  "summary": "1-2 sentence summary of when to use this procedure and what it does",
  "steps": ["ordered repeatable steps"],
  "trigger_phrases": ["short natural-language prompts this should match"],
  "tools": ["commands, systems, or tools that materially matter"],
  "confidence": 0.0,
  "source_quote": "optional supporting quote"
}}

Rules:
- Use the whole transcript, including user and assistant turns.
- Focus on one durable repeatable procedure only.
- Prefer project-specific workflows over generic advice.
- Keep exact literals when central: commands, flags, env vars, file names, branch names, service names, rollout percentages, and environments.
- Steps must be concrete, reusable, and written as short imperative actions.
- Steps must describe a repeatable workflow, not a one-time historical narration.
- Good procedural memories often answer questions like: how do we deploy this, how do we roll this back, how do we set this up, how do we run this backfill, how do we debug this incident, or what is the release checklist.
- Do NOT emit a procedure for: one-off bug fixes, status updates, summaries of what happened once, decisions without repeatable steps, personal facts, or transcripts that mainly contain requests without a reusable workflow.
- If the transcript does not contain a clear durable procedure, return an empty JSON object: {{}}.
- Do not invent missing steps, tools, or trigger phrases.
- Return raw JSON only. No prose. No markdown fences.

Example 1:
Transcript:
User: The deploy workflow is always build the Docker image, run alembic upgrade, then roll out the web deployment in staging before production.
Assistant: I also verify the healthcheck endpoint before promoting.

Output:
{{
  "title": "Web deploy workflow",
  "summary": "Use this workflow when deploying the web service. It builds the image, runs migrations, validates staging, and checks health before promotion.",
  "steps": [
    "Build the Docker image",
    "Run alembic upgrade",
    "Roll out the web deployment in staging",
    "Verify the healthcheck endpoint before promoting to production"
  ],
  "trigger_phrases": ["how do i deploy the web service", "deploy workflow"],
  "tools": ["Docker", "alembic", "staging", "production"],
  "confidence": 0.92
}}

Example 2:
Transcript:
User: When the checkout flag misbehaves, first disable the feature flag, then clear the edge cache, then retry with an employee account.
Assistant: If it still fails, capture the request id and page support.

Output:
{{
  "title": "Checkout flag incident procedure",
  "summary": "Use this when the checkout feature flag misbehaves. It starts with rollback and cache clearing, then narrows the issue before escalating.",
  "steps": [
    "Disable the checkout feature flag",
    "Clear the edge cache",
    "Retry with an employee account",
    "Capture the request id if it still fails",
    "Page support"
  ],
  "trigger_phrases": ["checkout flag broken", "feature flag rollback steps"],
  "tools": ["feature flag", "edge cache", "request id"],
  "confidence": 0.9
}}

Example 3:
Transcript:
User: We fixed the login loop by moving token validation into shared middleware.
Assistant: The patch is merged and staging is green.
User: Next week we can clean up the old helper.

Output:
{{}}

Transcript:
{transcript}

Return only the JSON object.
'''


def parse_extracted_procedure(raw_output: str) -> ExtractedProcedure | None:
    """Parse model output into a validated extracted procedure.

    ``None`` means the transcript did not contain a durable procedural memory.
    """
    payload = parse_json_payload(raw_output)
    if payload in ({}, []):
        return None
    if not isinstance(payload, dict):
        raise InferenceError("Procedural extractor did not return a JSON object")

    try:
        return ExtractedProcedure.model_validate(payload)
    except ValidationError as exc:
        raise InferenceError(f"Invalid procedural memory: {exc}") from exc


def extract_procedure_from_session_text(
    session_text: str,
    *,
    model: str | None = None,
    generate_fn: GenerationFn = generate_text,
    timeout_seconds: int = 180,
    source: str = "procedural_extractor",
    session_id: str | None = None,
) -> ExtractedProcedure | None:
    """Extract one validated procedural memory from a session transcript."""
    prompt = build_procedural_extraction_prompt(session_text)
    last_error: Exception | None = None

    log_procedural_event(
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
            procedure = parse_extracted_procedure(result.text)
            log_procedural_event(
                "extract_result",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                raw_model_output=result.text,
                validated_object=procedure.model_dump(mode="json") if procedure is not None else None,
                model=result.model,
            )
            return procedure
        except Exception as exc:
            last_error = exc
            log_procedural_event(
                "validation_error",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                prompt=attempt_prompt,
                error=repr(exc),
            )

    if isinstance(last_error, Exception):
        raise last_error
    raise InferenceError("Procedural extraction failed without an error")


def procedure_to_semantic_core(procedure: ExtractedProcedure | dict | None) -> dict[str, object] | None:
    """Return a comparison-friendly reduced shape for tests and fixtures."""
    if procedure is None:
        return None
    if isinstance(procedure, ExtractedProcedure):
        return {
            "title": procedure.title,
            "summary": procedure.summary,
            "steps": list(procedure.steps),
            "trigger_phrases": list(procedure.trigger_phrases),
            "tools": list(procedure.tools),
        }

    return {
        "title": str(procedure.get("title", "")).strip(),
        "summary": str(procedure.get("summary", "")).strip(),
        "steps": [str(item).strip() for item in procedure.get("steps", []) if str(item).strip()],
        "trigger_phrases": [str(item).strip() for item in procedure.get("trigger_phrases", []) if str(item).strip()],
        "tools": [str(item).strip() for item in procedure.get("tools", []) if str(item).strip()],
    }



def normalized_contains(text: str, snippet: str) -> bool:
    """Case-insensitive whitespace-tolerant containment for eval fixtures."""
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))
    haystack = normalize(text)
    needle = normalize(snippet)
    return bool(needle) and needle in haystack


def discover_procedural_fixture_cases(fixtures_root: str | Path) -> list[str]:
    """Return sorted fixture basenames that have both session and expected files."""
    root = Path(fixtures_root)
    sessions_dir = root / "sessions"
    expected_dir = root / "expected"

    cases: list[str] = []
    for session_file in sorted(sessions_dir.glob("*.txt")):
        case_name = session_file.stem
        if (expected_dir / f"{case_name}.json").exists():
            cases.append(case_name)
    return cases
