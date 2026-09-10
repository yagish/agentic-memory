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


_EPISODIC_PROMPT_VERSION = "episodic-v2"
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
        "Use arrays for participants, decisions, outcomes, and follow_ups. "
        "Preserve concrete names, branch names, env vars, flags, file names, model names, and numeric values when they are central to the episode. "
        "Prefer concrete transcript wording over generic paraphrases. Include explicitly named people when they materially participated or were assigned ownership. "
        "When the transcript states a concrete completed result like passed, green, fixed, works now, enabled, acknowledged, shipped, or added, keep that result in outcomes."
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
- Focus on the single main event, discussion, or decision from this session.
- Title requirements:
  - max 10 words
  - concrete and specific, not generic
  - preserve key domain terms from the transcript when central, such as auth middleware, deploy rollback, feature flag, cache invalidation, migration script, or signup flow
- Abstract requirements:
  - 1-2 sentences
  - mention the concrete problem/topic, the main action/decision, and the current result or end state
  - preserve important literal details when central: names, branch names, env vars, flags, file names, model names, percentages, dates, and numeric thresholds
  - prefer transcript wording over generic paraphrases
- Participants:
  - include only explicitly named people
  - do not include generic roles like "user" or "assistant"
  - include named people when they materially participated, requested the work, were assigned ownership, or received the handoff
- Decisions:
  - keep only explicit decisions or chosen approaches
  - preserve concrete chosen details like exact thresholds, branches, models, policies, or architecture choices when stated
- Outcomes:
  - keep concrete completed results, fixes, shipped changes, or communicated outcomes
  - include the actual result when stated, not a vague paraphrase
  - prefer explicit completion/result phrases when present, such as passed, green, fixed, works now, enabled, acknowledged, shipped, or added
- Follow-ups:
  - keep concrete next steps or unresolved tasks
  - preserve named environments, tests, timelines, and owners when stated
- If a list has no items, return an empty array.
- Never invent participants, decisions, outcomes, or follow-ups.
- Never copy values from examples unless they appear in the transcript.
- Return raw JSON only. No prose. No markdown fences.

Example 1:
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

Example 2:
Transcript:
User: Jenna will own the postmortem for yesterday's search outage.
Assistant: We agreed the report should focus on the slow database failover and the missing alert.
User: Add the remediation timeline before Friday's review.

Output:
{{
  "title": "Search outage postmortem",
  "abstract": "The session assigned Jenna to own the postmortem for the search outage and focused the report on the slow database failover and the missing alert. The remediation timeline still needs to be added before Friday's review.",
  "participants": ["Jenna"],
  "decisions": ["Focus the postmortem on the slow database failover and the missing alert"],
  "outcomes": [],
  "follow_ups": ["Add the remediation timeline before Friday's review"],
  "confidence": 0.9
}}

Example 3:
Transcript:
User: We rolled back the API deploy after the service started returning 500s.
Assistant: The root cause was a missing STRIPE_WEBHOOK_SECRET environment variable.
User: We added a startup guard so production will fail fast next time.
Assistant: Tomorrow verify the new guard in staging before redeploying.

Output:
{{
  "title": "API deploy rollback",
  "abstract": "The session rolled back the API deploy after 500s and identified a missing STRIPE_WEBHOOK_SECRET environment variable as the root cause. A startup guard was added, and it still needs staging verification before redeploying.",
  "participants": [],
  "decisions": ["Add a startup guard"],
  "outcomes": ["Rolled back the API deploy", "Added a startup guard"],
  "follow_ups": ["Verify the new guard in staging before redeploying"],
  "confidence": 0.93
}}

Example 4:
Transcript:
User: Miguel wants the new checkout flow behind a feature flag until support is ready.
Assistant: We agreed to launch it to 10 percent of employees first.
User: I enabled the flag for internal accounts and support has the rollback instructions.

Output:
{{
  "title": "Checkout feature flag rollout",
  "abstract": "The session put the new checkout flow behind a feature flag and chose a 10 percent employee rollout. Miguel was the named stakeholder, and the flag was enabled for internal accounts with rollback instructions ready for support.",
  "participants": ["Miguel"],
  "decisions": ["Launch the new checkout flow to 10 percent of employees behind a feature flag"],
  "outcomes": ["Enabled the flag for internal accounts", "Support has rollback instructions"],
  "follow_ups": [],
  "confidence": 0.9
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
