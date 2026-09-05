"""LLM-backed fact extraction helpers and fixture-friendly normalization."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
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
_FACT_PROMPT_VERSION = "facts-v3"


# Seam so tests can inject a fake model call while production uses Ollama.
GenerationFn = Callable[[GenerationRequest], GenerationResult]


_FACTS_LOG_PATH = os.path.expanduser("~/.memory/facts.log")


def _should_write_log_file() -> bool:
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    return True



def log_fact_event(event: str, **payload) -> None:
    """Append one JSON line to the retained facts log."""
    if not _should_write_log_file():
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": event,
        **payload,
    }
    try:
        os.makedirs(os.path.dirname(_FACTS_LOG_PATH), exist_ok=True)
        with open(_FACTS_LOG_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass



def _facts_debug_log(label: str, value: str) -> None:
    log_fact_event(label, value=value)


def _with_strict_json_retry(prompt: str) -> str:
    """Append a stricter reminder for one retry after a bad/ambiguous response."""
    return (
        f"{prompt}\n"
        "Reminder: output only a JSON array. Extract only stable user profile facts, "
        "stable user preferences, and clearly stated durable project metadata. "
        "Do not emit request/action facts. Never convert code-edit instructions, "
        "file names, function names, CLI flags, bug reports, or current-session tasks into facts. "
        "If unsure, return []."
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

    Facts should be conservative: mostly stable user profile facts and stable
    user preferences, with occasional explicit durable project metadata.
    Session-specific work requests belong in episodic memory, not here.
    """
    user_lines = _extract_user_lines(session_text)
    # Rebuild a clean user-only transcript so assistant turns never reach the model.
    user_transcript = "\n".join(f"User: {line}" for line in user_lines)

    return f"""You extract durable facts from conversation transcripts.

Prompt version: {_FACT_PROMPT_VERSION}

Definition of a valid fact:
- A fact is stable, reusable knowledge that will likely still be useful in future sessions.
- Most valid facts are about the user: identity, role, company, timezone, location, editor, shell, package manager, terminal, and stable preferences.
- A stable preference includes recurring user preferences such as response style, tooling preference, or workflow preference when the user states them explicitly.
- Only extract non-user facts when the user explicitly states durable project metadata such as repo name or default branch.

Rules:
- Extract facts from USER messages only.
- Ignore all assistant statements, guesses, and corrections.
- Output a raw JSON array only. No prose. No markdown fences.
- Every fact must include: entity, attribute, value, source_quote.
- Optional fields allowed: confidence, evidence.
- Use concise, normalized, stable entity names.
- Use snake_case for attributes.
- For personal profile facts and user preferences, prefer entity "user".
- Preserve the order of first appearance.
- Deduplicate exact duplicates with the same entity, attribute, and value.
- Keep conflicting facts when the entity and attribute are the same but the value differs.
- Extract only explicit user-stated facts. Never infer missing facts.
- Never copy values from examples into the output.
- Do not extract requests, questions, tasks, TODOs, commands, implementation instructions, bug reports, or temporary plans.
- Do not turn file names, function names, flags, or one-off code-change requests into facts.
- If the transcript is mostly current-session work instructions, return [].
- If unsure whether something is a durable fact, return [].
- If the user says they work at a company as a role/title, extract both company and role.

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
  {{"entity": "user", "attribute": "timezone", "value": "PST", "source_quote": "My timezone is PST."}},
  {{"entity": "user", "attribute": "timezone", "value": "EST", "source_quote": "Actually my timezone is EST."}}
]

Input:
User: My editor is Neovim.
User: My shell is zsh.
Output:
[
  {{"entity": "user", "attribute": "editor", "value": "Neovim", "source_quote": "My editor is Neovim."}},
  {{"entity": "user", "attribute": "shell", "value": "zsh", "source_quote": "My shell is zsh."}}
]

Input:
User: I prefer concise answers.
Output:
[
  {{"entity": "user", "attribute": "response_style", "value": "concise", "source_quote": "I prefer concise answers."}}
]

Input:
User: I work at Kroger as a Tech lead.
Output:
[
  {{"entity": "user", "attribute": "company", "value": "Kroger", "source_quote": "I work at Kroger as a Tech lead."}},
  {{"entity": "user", "attribute": "role", "value": "Tech lead", "source_quote": "I work at Kroger as a Tech lead."}}
]

Input:
User: The repo name is agentic-memory.
User: The default branch is main.
Output:
[
  {{"entity": "repo", "attribute": "name", "value": "agentic-memory", "source_quote": "The repo name is agentic-memory."}},
  {{"entity": "repo", "attribute": "default_branch", "value": "main", "source_quote": "The default branch is main."}}
]

Input:
User: Remove ObsoleteTables from db.py.
Output:
[]

Input:
User: Make daemon run facts first, then episodic.
Output:
[]

Input:
User: Use --once as a force flag.
Output:
[]

Input:
User: Let's clean up the DB tables and run extraction.
Output:
[]

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
