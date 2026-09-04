"""LLM-backed fact extraction helpers and fixture-friendly normalization."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

from pydantic import ValidationError

from memory.contracts import ExtractedFact
from memory.inference import (
    GenerationRequest,
    GenerationResult,
    InferenceError,
    generate_text,
    parse_json_payload,
)


# Keep prompt version explicit so future prompt changes can be tracked in tests.
_FACT_PROMPT_VERSION = "facts-v1"


# Seam so tests can inject a fake model call while production uses Ollama.
GenerationFn = Callable[[GenerationRequest], GenerationResult]


_FACTS_DEBUG_LOG_PATH = os.path.expanduser("~/.memory/facts_debug.log")


def _facts_debug_enabled() -> bool:
    return str(os.environ.get("MEMORY_DEBUG_FACTS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _facts_debug_log(label: str, value: str) -> None:
    """Emit opt-in debug output for fact extraction investigations.

    When MEMORY_DEBUG_FACTS=1, logs are written to both stderr and a dedicated
    file so developers can inspect extraction runs from the console or dashboard.
    """
    if not _facts_debug_enabled():
        return

    message = f"\n=== FACT DEBUG: {label} ===\n{value}\n"

    try:
        os.makedirs(os.path.dirname(_FACTS_DEBUG_LOG_PATH), exist_ok=True)
        with open(_FACTS_DEBUG_LOG_PATH, "a") as f:
            f.write(message)
    except Exception:
        pass

    try:
        sys.stderr.write(message)
        sys.stderr.flush()
    except Exception:
        pass


def _with_strict_json_retry(prompt: str) -> str:
    """Append a stricter reminder for one retry after a bad/ambiguous response."""
    return (
        f"{prompt}\n"
        "Reminder: output only a JSON array. Do not emit request/action facts. "
        "For personal profile facts, prefer entity=\"user\" with attributes like "
        "name, location, timezone, role, company, editor, shell, package_manager, terminal."
    )


def _extract_user_lines(session_text: str) -> list[str]:
    """Return only user-authored lines when speaker prefixes are present.

    If fixtures omit speaker prefixes entirely, we fall back to treating the
    whole text block as user-authored so the extractor still has usable input.
    """
    user_lines: list[str] = []
    for raw_line in session_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("user:"):
            content = line.split(":", 1)[1].strip()
            if content:
                user_lines.append(content)
    if user_lines:
        return user_lines
    return [line.strip() for line in session_text.splitlines() if line.strip()]


def build_fact_extraction_prompt(session_text: str) -> str:
    """Build the pinned extraction prompt for durable facts.

    The prompt is the main place where we steer the LLM toward stable,
    fixture-friendly output: JSON array only, user-only facts, no action items,
    and normalized entity/attribute naming.
    """
    user_lines = _extract_user_lines(session_text)
    # Rebuild a clean user-only transcript so assistant turns never reach the model.
    user_transcript = "\n".join(f"User: {line}" for line in user_lines)

    return f"""You extract durable facts from conversation transcripts.

Prompt version: {_FACT_PROMPT_VERSION}

Rules:
- Extract facts from USER messages only.
- Ignore all assistant statements, guesses, and corrections.
- Output a raw JSON array only. No prose. No markdown fences.
- Every fact must include: entity, attribute, value.
- Optional fields allowed: confidence, evidence, source_quote.
- Use concise, normalized, stable entity names.
- Use snake_case for attributes.
- For personal profile facts, prefer entity "user".
- Preserve the order of first appearance.
- Deduplicate exact duplicates with the same entity, attribute, and value.
- Keep conflicting facts when the entity and attribute are the same but the value differs.
- Extract durable user-stated facts, not requests, questions, tasks, or temporary chatter.
- Never turn a request like "Can you debug this?" into a fact.

Examples:
Input:
User: My name is Yash.
Output:
[
  {{"entity": "user", "attribute": "name", "value": "Yash", "source_quote": "My name is Yash."}}
]

Input:
User: My timezone is PST.
User: Actually my timezone is EST.
Output:
[
  {{"entity": "user", "attribute": "timezone", "value": "PST"}},
  {{"entity": "user", "attribute": "timezone", "value": "EST"}}
]

Input:
User: The repo name is agentic-memory.
User: The default branch is main.
Output:
[
  {{"entity": "repo", "attribute": "name", "value": "agentic-memory"}},
  {{"entity": "repo", "attribute": "default_branch", "value": "main"}}
]

Input:
User: My editor is Neovim.
User: My shell is zsh.
Output:
[
  {{"entity": "user", "attribute": "editor", "value": "Neovim"}},
  {{"entity": "user", "attribute": "shell", "value": "zsh"}}
]

Input:
User: My role is backend engineer.
User: My company is Acme.
Output:
[
  {{"entity": "user", "attribute": "role", "value": "backend engineer"}},
  {{"entity": "user", "attribute": "company", "value": "Acme"}}
]

Input:
User: I'm based in Seattle.
Output:
[
  {{"entity": "user", "attribute": "location", "value": "Seattle"}}
]

Input:
User: Thanks for the help.
User: Can you debug this?
Output:
[]

Transcript:
{user_transcript}

Return only the JSON array.
"""


def _merge_duplicate_fact(existing: ExtractedFact, incoming: ExtractedFact) -> ExtractedFact:
    """Merge duplicate semantic facts while keeping the best available metadata."""
    confidence = existing.confidence
    if incoming.confidence is not None and (
        confidence is None or incoming.confidence > confidence
    ):
        # Use the stronger confidence when both facts describe the same meaning.
        confidence = incoming.confidence

    # Preserve first-seen ordering, but fill in missing metadata from later duplicates.
    evidence = existing.evidence or incoming.evidence
    source_quote = existing.source_quote or incoming.source_quote

    return existing.model_copy(
        update={
            "confidence": confidence,
            "evidence": evidence,
            "source_quote": source_quote,
        }
    )


def normalize_extracted_facts(facts: Iterable[ExtractedFact]) -> list[ExtractedFact]:
    """Deduplicate semantic duplicates while preserving first-seen ordering.

    Key rule: duplicates collapse, conflicts stay.
    So:
    - same entity+attribute+value -> one fact
    - same entity+attribute but different value -> keep both
    """
    deduped: list[ExtractedFact] = []
    index_by_key: dict[tuple[str, str, str], int] = {}

    for fact in facts:
        key = (fact.entity, fact.attribute, fact.value)
        if key in index_by_key:
            existing_index = index_by_key[key]
            deduped[existing_index] = _merge_duplicate_fact(deduped[existing_index], fact)
            continue
        index_by_key[key] = len(deduped)
        deduped.append(fact)

    return deduped


def parse_extracted_facts(raw_output: str) -> list[ExtractedFact]:
    """Parse model output into validated extracted facts.

    Flow:
    1. peel JSON out of any surrounding prose
    2. require a JSON array
    3. validate each item against ExtractedFact
    4. dedupe exact duplicates
    """
    payload = parse_json_payload(raw_output)
    if not isinstance(payload, list):
        raise InferenceError("Fact extractor did not return a JSON array")

    parsed: list[ExtractedFact] = []
    errors: list[str] = []
    for index, item in enumerate(payload):
        try:
            parsed.append(ExtractedFact.model_validate(item))
        except ValidationError as exc:
            errors.append(f"item {index}: {exc}")

    if errors:
        raise InferenceError("Invalid extracted facts: " + "; ".join(errors))

    return normalize_extracted_facts(parsed)


def extract_facts_from_session_text(
    session_text: str,
    *,
    model: str | None = None,
    generate_fn: GenerationFn = generate_text,
    timeout_seconds: int = 180,
) -> list[ExtractedFact]:
    """Extract durable facts from one session transcript using the configured LLM.

    This is the public entry point used by the fixture tests:
    session text in -> prompt -> Qwen -> validated/normalized facts out.
    """
    prompt = build_fact_extraction_prompt(session_text)
    last_error: Exception | None = None

    _facts_debug_log("session_text", session_text)

    for attempt_number, attempt_prompt in enumerate((prompt, _with_strict_json_retry(prompt)), start=1):
        try:
            _facts_debug_log(f"prompt_attempt_{attempt_number}", attempt_prompt)
            result = generate_fn(
                GenerationRequest(
                    prompt=attempt_prompt,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    temperature=0.0,
                )
            )
            _facts_debug_log(f"raw_model_output_attempt_{attempt_number}", result.text)
            facts = parse_extracted_facts(result.text)
            _facts_debug_log(
                f"parsed_facts_attempt_{attempt_number}",
                json.dumps([fact.model_dump(mode="json") for fact in facts], indent=2),
            )
            return facts
        except Exception as exc:
            # Retry once with a stricter reminder before failing the test.
            _facts_debug_log(f"attempt_{attempt_number}_error", repr(exc))
            last_error = exc

    if isinstance(last_error, Exception):
        raise last_error
    raise InferenceError("Fact extraction failed without an error")


def facts_to_semantic_core(facts: Iterable[ExtractedFact | dict]) -> list[dict[str, str]]:
    """Return the exact semantic-core shape used by fixture expectations.

    Tests compare only this reduced shape so optional metadata can evolve
    without making every fixture brittle.
    """
    normalized: list[dict[str, str]] = []
    for fact in facts:
        if isinstance(fact, ExtractedFact):
            normalized.append(
                {
                    "entity": fact.entity,
                    "attribute": fact.attribute,
                    "value": fact.value.strip(),
                }
            )
            continue

        normalized.append(
            {
                "entity": str(fact["entity"]).strip().lower().replace(" ", "_").replace("-", "_"),
                "attribute": str(fact["attribute"]).strip().lower().replace(" ", "_").replace("-", "_"),
                "value": str(fact["value"]).strip(),
            }
        )
    return normalized


def discover_fact_fixture_cases(fixtures_root: str | Path) -> list[str]:
    """Return sorted fixture basenames that have both session and expected files."""
    root = Path(fixtures_root)
    sessions_dir = root / "sessions"
    expected_dir = root / "expected"

    cases = []
    for session_file in sorted(sessions_dir.glob("*.txt")):
        case_name = session_file.stem
        expected_file = expected_dir / f"{case_name}.json"
        # A case is only runnable if both halves of the pair exist.
        if expected_file.exists():
            cases.append(case_name)
    return cases
