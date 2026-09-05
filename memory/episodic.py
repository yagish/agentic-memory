"""LLM-backed episodic-memory extraction helpers and eval-harness utilities."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from memory.contracts import ExtractedEpisode
from memory.inference import (
    GenerationRequest,
    GenerationResult,
    InferenceError,
    generate_text,
    parse_json_payload,
)


_EPISODIC_PROMPT_VERSION = "episodic-v1"
_EPISODIC_LOG_PATH = os.path.expanduser("~/.memory/episodic.log")

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


def log_episodic_event(event: str, **payload) -> None:
    """Append one JSON line to the dedicated episodic-memory log."""
    try:
        if not _should_write_log_file():
            return
        os.makedirs(os.path.dirname(_EPISODIC_LOG_PATH), exist_ok=True)
        record = {"timestamp": _utc_now(), "event": event, **payload}
        with open(_EPISODIC_LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _with_strict_json_retry(prompt: str) -> str:
    return (
        f"{prompt}\n"
        "Reminder: return only one JSON object. Do not use markdown fences. "
        "Use arrays for participants, decisions, outcomes, and follow_ups."
    )


def build_episodic_extraction_prompt(session_text: str) -> str:
    """Build the pinned extraction prompt for one episodic memory."""
    transcript = session_text.strip()

    return f"""You extract one episodic memory from a conversation transcript.

Prompt version: {_EPISODIC_PROMPT_VERSION}

Return exactly one JSON object with this shape:
{{
  "title": "short title",
  "abstract": "1-2 sentence summary of what happened and where it ended",
  "participants": ["optional explicit people names"],
  "decisions": ["explicit decisions or chosen approaches"],
  "outcomes": ["completed results or resolved outcomes"],
  "follow_ups": ["next steps or unresolved work"],
  "confidence": 0.0,
  "source_quote": "optional supporting quote"
}}

Rules:
- Use the whole transcript, including user and assistant turns.
- Focus on the main event, discussion, or decision from this session.
- title: max 10 words, concrete and specific.
- abstract: mention what happened and the current result/end state.
- participants: only include explicitly named people; do not include generic roles like "user" or "assistant".
- decisions: keep only explicit decisions or chosen approaches.
- outcomes: keep concrete completed results or resolutions.
- follow_ups: keep concrete next steps or unresolved tasks.
- If a list has no items, return an empty array.
- Return raw JSON only. No prose. No markdown fences.

Example:
Transcript:
User: We decided to move token validation into shared auth middleware.
Assistant: I implemented the middleware and the login loop stopped.
User: Next session add regression tests.

Output:
{{
  "title": "Auth middleware fix",
  "abstract": "The session moved token validation into shared auth middleware and implemented the change. The login loop was fixed, with regression tests still left to add.",
  "participants": [],
  "decisions": ["Move token validation into shared auth middleware"],
  "outcomes": ["Login loop fixed"],
  "follow_ups": ["Add regression tests"],
  "confidence": 0.92
}}

Transcript:
{transcript}

Return only the JSON object.
"""


def parse_extracted_episode(raw_output: str) -> ExtractedEpisode:
    """Parse model output into a validated extracted episode."""
    payload = parse_json_payload(raw_output)
    if not isinstance(payload, dict):
        raise InferenceError("Episodic extractor did not return a JSON object")

    try:
        return ExtractedEpisode.model_validate(payload)
    except ValidationError as exc:
        raise InferenceError(f"Invalid episodic memory: {exc}") from exc


def extract_episode_from_session_text(
    session_text: str,
    *,
    model: str | None = None,
    generate_fn: GenerationFn = generate_text,
    timeout_seconds: int = 180,
    source: str = "episodic_extractor",
    session_id: str | None = None,
) -> ExtractedEpisode:
    """Extract one validated episodic memory from a session transcript."""
    prompt = build_episodic_extraction_prompt(session_text)
    last_error: Exception | None = None

    log_episodic_event(
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
            episode = parse_extracted_episode(result.text)
            log_episodic_event(
                "extract_result",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                raw_model_output=result.text,
                validated_object=episode.model_dump(mode="json"),
                model=result.model,
            )
            return episode
        except Exception as exc:
            last_error = exc
            log_episodic_event(
                "validation_error",
                source=source,
                session_id=session_id,
                attempt=attempt_number,
                prompt=attempt_prompt,
                error=repr(exc),
            )

    if isinstance(last_error, Exception):
        raise last_error
    raise InferenceError("Episodic extraction failed without an error")


def episode_to_semantic_core(episode: ExtractedEpisode | dict) -> dict[str, object]:
    """Return a comparison-friendly reduced shape for tests and fixtures."""
    if isinstance(episode, ExtractedEpisode):
        return {
            "title": episode.title,
            "abstract": episode.abstract,
            "participants": list(episode.participants),
            "decisions": list(episode.decisions),
            "outcomes": list(episode.outcomes),
            "follow_ups": list(episode.follow_ups),
        }

    return {
        "title": str(episode.get("title", "")).strip(),
        "abstract": str(episode.get("abstract", "")).strip(),
        "participants": [str(item).strip() for item in episode.get("participants", []) if str(item).strip()],
        "decisions": [str(item).strip() for item in episode.get("decisions", []) if str(item).strip()],
        "outcomes": [str(item).strip() for item in episode.get("outcomes", []) if str(item).strip()],
        "follow_ups": [str(item).strip() for item in episode.get("follow_ups", []) if str(item).strip()],
    }


def normalized_contains(text: str, snippet: str) -> bool:
    """Case-insensitive whitespace-tolerant containment for eval fixtures."""
    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))
    haystack = normalize(text)
    needle = normalize(snippet)
    return bool(needle) and needle in haystack


def discover_episodic_fixture_cases(fixtures_root: str | Path) -> list[str]:
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
