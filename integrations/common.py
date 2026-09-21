"""Shared memory integration helpers for agent-specific adapters.

This module keeps save/retrieve policy independent from any specific agent
runtime. Claude hooks and pi extensions should call these functions instead of
re-implementing DB open/save/retrieval behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from memory.db import bootstrap_db, reinforce_episodic_memories
from memory.facts.renderer import render_fact_answer
from memory.servers.ingest_pipeline import IngestOutcome, ingest_session
from memory.retrieval import WakeUpContext, build_wake_up_injection, retrieve_wake_up_context
from memory.utils.logger import activity_log, error_log


DEFAULT_DB_PATH = os.path.expanduser("~/.memory/memory.db")
PROJECT_CONTEXT_FIELDS = ("project_id", "repo_root", "cwd", "git_remote", "git_branch")


def _normalize_project_value(value, *, is_path: bool = False) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if is_path:
        return os.path.abspath(os.path.expanduser(cleaned))
    return cleaned


def normalize_project_context(
    project_context: dict | None = None,
    metadata: dict | None = None,
) -> dict[str, str]:
    """Return normalized ambient project context.

    Explicit top-level project fields win. Missing values are backfilled from
    metadata where possible so older sessions remain usable.
    """
    project_context = project_context if isinstance(project_context, dict) else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    metadata_project = metadata.get("project_context")
    metadata_project = metadata_project if isinstance(metadata_project, dict) else {}

    normalized: dict[str, str] = {}
    for field in PROJECT_CONTEXT_FIELDS:
        value = project_context.get(field)
        if value in (None, ""):
            value = metadata_project.get(field)
        if value in (None, ""):
            if field == "git_branch":
                value = metadata.get("git_branch") or metadata.get("branch")
            else:
                value = metadata.get(field)
        cleaned = _normalize_project_value(value, is_path=field in {"repo_root", "cwd"})
        if cleaned:
            normalized[field] = cleaned

    if "project_id" not in normalized:
        for fallback_field in ("git_remote", "repo_root", "cwd"):
            fallback_value = normalized.get(fallback_field)
            if fallback_value:
                normalized["project_id"] = fallback_value
                break

    return normalized


@dataclass(frozen=True)
class RecallOutcome:
    """Decision for one prompt after memory retrieval."""

    context: WakeUpContext
    answer: str = ""
    injection: str = ""

    @property
    def action(self) -> str:
        if self.answer:
            return "answer"
        if self.injection:
            return "inject"
        return "noop"


def empty_wake_up_context() -> WakeUpContext:
    return WakeUpContext(
        cache_hit=None,
        working_mem=None,
        enrichment=[],
        episodic=[],
        facts=[],
        procedural=[],
        session_memory=[],
        warnings=[],
    )


def open_memory_db_for_ingest(db_path: str = DEFAULT_DB_PATH):
    """Open the memory DB, bootstrapping and migrating it if needed."""
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return bootstrap_db(db_path)


def open_existing_memory_db(db_path: str = DEFAULT_DB_PATH):
    """Open the memory DB only if it already exists and is non-empty."""
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        return None
    return bootstrap_db(db_path)


def save_session_to_memory(
    conn,
    *,
    session_id: str,
    agent: str,
    turns: list[dict],
    started_at: str,
    updated_at: str,
    metadata: dict | None = None,
    project_context: dict | None = None,
) -> IngestOutcome:
    """Persist a session transcript through the shared ingest seam."""
    return ingest_session(
        conn,
        session_id=session_id,
        agent=agent,
        turns=turns,
        started_at=started_at,
        updated_at=updated_at,
        metadata=metadata,
        project_context=normalize_project_context(project_context, metadata),
    )


def retrieve_prompt_memory(
    conn,
    prompt: str,
    *,
    include_working_memory: bool,
    session_id: str | None = None,
    project_context: dict | None = None,
    embed_fn=None,
) -> WakeUpContext:
    """Fetch memory context for one prompt through the shared retrieval seam."""
    kwargs = {
        "include_working_memory": include_working_memory,
        "session_id": session_id,
        "project_context": normalize_project_context(project_context),
    }
    if embed_fn is not None:
        kwargs["embed_fn"] = embed_fn
    context = retrieve_wake_up_context(conn, prompt, **kwargs)
    try:
        reinforce_episodic_memories(
            conn,
            [row.get("id") for row in context.episodic if row.get("id")],
            increment_retrieval=True,
        )
    except Exception as exc:
        error_log("retrieval", f"episodic reinforcement failed: {exc}", exc=exc)
    timings = dict(getattr(context, "timings", {}) or {})
    activity_log(
        "retrieval",
        "recall_timing",
        session=session_id,
        prompt_chars=len(prompt),
        include_working_memory=include_working_memory,
        prompt_embedding_ms=timings.get("prompt_embedding_ms"),
        memory_search_ms=timings.get("memory_search_ms"),
        retrieval_total_ms=timings.get("retrieval_total_ms"),
        facts_count=len(context.facts),
        episodic_count=len(context.episodic),
        procedural_count=len(context.procedural),
        session_memory_count=len(context.session_memory),
        working_memory_count=1 if context.working_mem else 0,
        warnings_count=len(context.warnings),
    )
    return context


def decide_prompt_memory_action(prompt: str, context: WakeUpContext) -> RecallOutcome:
    """Decide whether memory should answer directly or enrich the next turn."""
    has_contextual_memory = bool(
        context.working_mem or context.episodic or context.procedural or context.session_memory
    )
    if context.facts and not has_contextual_memory:
        answer = render_fact_answer(prompt, [str(fact.get("content", "")) for fact in context.facts])
        if answer:
            return RecallOutcome(context=context, answer=answer)
        return RecallOutcome(context=context)

    injection = build_wake_up_injection(context)
    return RecallOutcome(context=context, injection=injection)


def build_recall_response(prompt: str, context: WakeUpContext) -> dict:
    """Serialize one recall decision for HTTP/adapter callers."""
    outcome = decide_prompt_memory_action(prompt, context)
    response = {
        "action": outcome.action,
        "facts_count": len(context.facts),
        "episodic_count": len(context.episodic),
        "procedural_count": len(context.procedural),
        "session_memory_count": len(context.session_memory),
        "working_memory_count": 1 if context.working_mem else 0,
        "warnings": [warning.__dict__ for warning in context.warnings],
        "timings": dict(getattr(context, "timings", {}) or {}),
        "context": {
            "facts": context.facts,
            "episodic": context.episodic,
            "procedural": context.procedural,
            "session_memory": context.session_memory,
            "working_mem": context.working_mem,
        },
    }
    if outcome.answer:
        response["answer"] = outcome.answer
    if outcome.injection:
        response["injection"] = outcome.injection
    return response
