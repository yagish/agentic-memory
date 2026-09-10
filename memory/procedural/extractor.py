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


_PROCEDURAL_PROMPT_VERSION = "procedural-v2"
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
        "Keep only durable repeatable procedures, not one-off tasks. "
        "Preserve exact workflow labels, commands, flags, env vars, scripts, branch names, job names, service names, URLs, dashboards, Slack, and stage names when central. "
        "Copy step wording closely instead of paraphrasing away important literals. "
        "Preserve every explicit ordered action from the transcript; do not replace a final promote/publish/resume step with a warning. "
        "If there is no durable procedure, return {}."
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
- Keep exact literals when central: commands, flags, env vars, file names, branch names, job names, service names, rollout percentages, dashboard names, URLs, and environments.
- Do not change the spelling or casing of env vars, commands, file paths, URLs, branch names, or job names.
- Preserve the workflow label from the transcript when one is stated explicitly, such as: deploy workflow, local setup, release cutoff checklist, docs publish flow, backfill procedure, cache bust workflow, flaky test triage workflow, or incident debug workflow.
- The title should usually be: <service or domain> + <exact workflow label>.
- The summary should explicitly repeat the service name and the workflow label from the transcript when present.
- If the procedure is about rollback, disable, or recovery, include the word "rollback" in the summary.
- Steps must be concrete, reusable, and written as short imperative actions.
- Preserve every explicit ordered action from the transcript in order. If the transcript includes a final action like promote production, publish docs, resume the cron, or announce in Slack, keep it as its own step.
- Copy step wording closely when the transcript already provides good wording. Do not swap a stage name for an env var, and do not paraphrase away important literals.
- If the transcript includes a guard condition or escalation, keep it as an additional step instead of replacing the main flow.
- Trigger phrases should be natural future queries. Include one short article-free trigger phrase when possible, such as "deploy <service>", "publish docs for <service>", "cache bust <service>", or "run backfill for <service>". If the transcript names a workflow label, also include one trigger phrase that literally keeps that label, such as "deploy workflow for <service>" or "release cutoff checklist for <service>". You may also include one phrase shaped like "how do I ...".
- Tools should list exact important literals from the transcript: commands/scripts, env vars, flags, branch names, job names, dashboard names, URLs, stages like staging/preview/production, and channels like Slack when central. Reuse the actual stage named in the transcript; do not normalize preview to staging or staging to production. If an env var or literal is called out in a follow-up note like "keep X", "mention X", or "compare X", include it in tools.
- Good procedural memories often answer questions like: how do we deploy this, how do we roll this back, how do we set this up, how do we run this backfill, how do we debug this incident, or what is the release checklist.
- Do NOT emit a procedure for: one-off bug fixes, status updates, summaries of what happened once, decisions without repeatable steps, personal facts, or transcripts that mainly contain requests without a reusable workflow.
- Past-tense incident summaries like "the rollout failed" or "we rolled it back" are not procedures unless the transcript also gives explicit repeatable steps.
- If the transcript does not contain a clear durable procedure, return an empty JSON object: {{}}.
- Do not invent missing steps, tools, or trigger phrases.
- Return raw JSON only. No prose. No markdown fences.
- Do not put unescaped double quotes inside JSON string values.

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
  "trigger_phrases": ["deploy web service", "how do i deploy the web service"],
  "tools": ["Docker", "alembic", "staging", "production"],
  "confidence": 0.92
}}

Example 2:
Transcript:
User: When the checkout flag misbehaves, first disable the feature flag, then clear the edge cache, then retry with an employee account.
Assistant: If it still fails, capture the request id and page support.
User: Mention CHECKOUT_DB_URL in the incident notes if config drift caused it.

Output:
{{
  "title": "Checkout flag rollback procedure",
  "summary": "Use this rollback procedure when the checkout feature flag misbehaves. It starts with rollback and cache clearing, then narrows the issue before escalating.",
  "steps": [
    "Disable the checkout feature flag",
    "Clear the edge cache",
    "Retry with an employee account",
    "Capture the request id if it still fails",
    "Page support"
  ],
  "trigger_phrases": ["checkout flag broken", "feature flag rollback steps"],
  "tools": ["feature flag", "edge cache", "request id", "CHECKOUT_DB_URL"],
  "confidence": 0.9
}}

Example 3:
Transcript:
User: The local setup for checkout-api is clone the repo, copy .env.example to .env.local, export STAGING_DB_URL, run docker compose up checkout-api, then run pytest tests/checkout_api.
Assistant: If migrations are pending, run alembic upgrade before opening the app.

Output:
{{
  "title": "Checkout-api local setup",
  "summary": "Use this local setup for checkout-api. It clones the repo, prepares .env.local, exports STAGING_DB_URL, starts checkout-api with docker compose, runs pytest, and applies migrations if needed.",
  "steps": [
    "Clone the repo",
    "Copy .env.example to .env.local",
    "Export STAGING_DB_URL",
    "Run docker compose up checkout-api",
    "Run pytest tests/checkout_api",
    "Run alembic upgrade if migrations are pending"
  ],
  "trigger_phrases": ["how do i set up checkout-api locally", "run local setup for checkout-api"],
  "tools": [".env.local", "STAGING_DB_URL", "docker compose", "pytest tests/checkout_api", "alembic"],
  "confidence": 0.95
}}

Example 4:
Transcript:
User: The release cutoff checklist for checkout-api is freeze merges at 4 PM, cut branch release/atlas, tag the build, run smoke tests in staging, then announce the cutoff in Slack.
Assistant: If smoke tests fail, do not promote production.

Output:
{{
  "title": "Checkout-api release cutoff checklist",
  "summary": "Use this release cutoff checklist for checkout-api. It freezes merges, cuts release/atlas, runs smoke tests in staging, announces the cutoff in Slack, and blocks promotion to production if smoke tests fail.",
  "steps": [
    "Freeze merges at 4 PM",
    "Cut branch release/atlas",
    "Tag the build",
    "Run smoke tests in staging",
    "Announce the cutoff in Slack",
    "Do not promote production if smoke tests fail"
  ],
  "trigger_phrases": ["how do i run release cutoff for checkout-api", "release cutoff checklist for checkout-api"],
  "tools": ["release/atlas", "staging", "production", "Slack"],
  "confidence": 0.95
}}

Example 5:
Transcript:
User: The safe backfill procedure for checkout-api is pause the cron, run python3 scripts/backfill.py --job checkout_api_backfill --limit 500, verify the row count in staging, then resume the cron.
Assistant: After resuming, watch the error rate dashboard for 15 minutes.
User: Keep STAGING_DB_URL pointed at the backfill target the whole time.

Output:
{{
  "title": "Checkout-api safe backfill procedure",
  "summary": "Use this safe backfill procedure for checkout-api. It pauses the cron, runs the checkout_api_backfill job, verifies staging counts, resumes the cron, and watches the error rate dashboard.",
  "steps": [
    "Pause the cron",
    "Run python3 scripts/backfill.py --job checkout_api_backfill --limit 500",
    "Verify the row count in staging",
    "Resume the cron",
    "Watch the error rate dashboard for 15 minutes"
  ],
  "trigger_phrases": ["how do i run backfill for checkout-api", "run backfill for checkout-api"],
  "tools": ["cron", "python3 scripts/backfill.py", "checkout_api_backfill", "staging", "error rate dashboard", "STAGING_DB_URL"],
  "confidence": 0.95
}}

Example 6:
Transcript:
User: The docs publish flow for checkout-api is run pnpm docs:build, check the preview in staging, execute ./scripts/publish-docs.sh, then smoke-check https://docs.example.com/checkout-api.
Assistant: If the smoke check fails, roll back the docs flag before trying again.
User: Keep STAGING_DOCS_URL in the publish environment.

Output:
{{
  "title": "Checkout-api docs publish flow",
  "summary": "Use this docs publish flow for checkout-api. It builds the docs, checks the preview in staging, publishes them, smoke-checks the live site, and rolls back if the smoke check fails.",
  "steps": [
    "Run pnpm docs:build",
    "Check the preview in staging",
    "Execute ./scripts/publish-docs.sh",
    "Smoke-check https://docs.example.com/checkout-api",
    "Roll back the docs flag if the smoke check fails"
  ],
  "trigger_phrases": ["publish docs for checkout-api", "run docs publish flow for checkout-api"],
  "tools": ["pnpm docs:build", "./scripts/publish-docs.sh", "staging", "https://docs.example.com/checkout-api", "STAGING_DOCS_URL"],
  "confidence": 0.95
}}

Example 7:
Transcript:
User: Our flaky test triage workflow for admin-web is quarantine tests/admin_web/test_retry_logic.py::test_harbor_seed, rerun the suite with PYTEST_ADDOPTS="-x", capture the seed and logs, file the Jira ticket, then remove the quarantine after the fix merges.
Assistant: That should stay as the repeatable process.

Output:
{{
  "title": "Admin-web flaky test triage workflow",
  "summary": "Use this flaky test triage workflow for admin-web. It quarantines the failing test, reruns the suite with the PYTEST_ADDOPTS flag, captures the seed and logs, files the Jira ticket, and removes the quarantine after the fix merges.",
  "steps": [
    "Quarantine tests/admin_web/test_retry_logic.py::test_harbor_seed",
    "Rerun the suite with PYTEST_ADDOPTS=\"-x\"",
    "Capture the seed and logs",
    "File the Jira ticket",
    "Remove the quarantine after the fix merges"
  ],
  "trigger_phrases": ["how do i triage a flaky test for admin-web", "flaky test triage workflow for admin-web"],
  "tools": ["tests/admin_web/test_retry_logic.py::test_harbor_seed", "PYTEST_ADDOPTS=\"-x\"", "Jira"],
  "confidence": 0.95
}}

Example 8:
Transcript:
User: The incident debug workflow for pricing-api is collect the request id, check the pricing-api-latency dashboard, tail the api logs, compare PRICING_DB_URL, then roll back if the mismatch explains the failure.
Assistant: If rollback does not help, page the on-call and capture the failing payload.

Output:
{{
  "title": "Pricing-api incident debug workflow",
  "summary": "Use this incident debug workflow for pricing-api. It collects the request id, checks pricing-api-latency, tails API logs, compares PRICING_DB_URL, rolls back when needed, and escalates to the on-call if rollback does not help.",
  "steps": [
    "Collect the request id",
    "Check the pricing-api-latency dashboard",
    "Tail the API logs",
    "Compare PRICING_DB_URL",
    "Roll back if the mismatch explains the failure",
    "Page the on-call and capture the failing payload if rollback does not help"
  ],
  "trigger_phrases": ["how do i debug a pricing-api incident", "incident debug workflow for pricing-api"],
  "tools": ["request id", "pricing-api-latency", "API logs", "PRICING_DB_URL", "on-call"],
  "confidence": 0.95
}}

Example 9:
Transcript:
User: Yesterday the worker-service rollout failed because Redis was down.
Assistant: We rolled it back.

Output:
{{}}

Example 10:
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


def _repair_common_json_string_escapes(raw_output: str) -> str:
    """Repair common local-model JSON issues like unescaped quotes in string values."""
    repaired: list[str] = []
    in_string = False
    escaped = False
    length = len(raw_output)

    for index, char in enumerate(raw_output):
        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
                escaped = False
            continue

        if escaped:
            repaired.append(char)
            escaped = False
            continue

        if char == "\\":
            repaired.append(char)
            escaped = True
            continue

        if char == '"':
            next_char = raw_output[index + 1] if index + 1 < length else ""
            if next_char and next_char not in {",", "}", "]", ":", " ", "\n", "\r", "\t"}:
                repaired.append('\\"')
                continue
            repaired.append(char)
            in_string = False
            continue

        repaired.append(char)

    return "".join(repaired)



def _enrich_procedure_literals(procedure: ExtractedProcedure) -> ExtractedProcedure:
    """Backfill obvious literal tools from already-extracted procedure text."""
    tools = list(procedure.tools)
    combined = "\n".join(
        [procedure.title, procedure.summary, *procedure.steps, *procedure.trigger_phrases]
    )

    for stage in ("staging", "preview", "production"):
        if re.search(rf"\b{re.escape(stage)}\b", combined, flags=re.IGNORECASE):
            if not any(normalized_contains(item, stage) for item in tools):
                tools.append(stage)

    for env_var in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", combined):
        if not any(normalized_contains(item, env_var) for item in tools):
            tools.append(env_var)

    if tuple(tools) == procedure.tools:
        return procedure
    return procedure.model_copy(update={"tools": tuple(tools)})



def parse_extracted_procedure(raw_output: str) -> ExtractedProcedure | None:
    """Parse model output into a validated extracted procedure.

    ``None`` means the transcript did not contain a durable procedural memory.
    """
    payload = parse_json_payload(raw_output)
    if (payload == [] or not isinstance(payload, dict)) and "{" in raw_output:
        repaired_payload = parse_json_payload(_repair_common_json_string_escapes(raw_output))
        if isinstance(repaired_payload, dict) or repaired_payload in ({}, []):
            payload = repaired_payload
    if payload in ({}, []):
        return None
    if not isinstance(payload, dict):
        raise InferenceError("Procedural extractor did not return a JSON object")

    try:
        procedure = ExtractedProcedure.model_validate(payload)
    except ValidationError as exc:
        raise InferenceError(f"Invalid procedural memory: {exc}") from exc
    return _enrich_procedure_literals(procedure)


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
