"""Shared memory integration helpers for agent-specific adapters.

This module keeps save/retrieve policy independent from any specific agent
runtime. Claude hooks and pi extensions should call these functions instead of
re-implementing DB open/save/retrieval behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from memory.db import bootstrap_db, open_db
from memory.facts.renderer import render_fact_answer
from memory.servers.ingest_pipeline import IngestOutcome, ingest_session
from memory.retrieval import WakeUpContext, build_wake_up_injection, retrieve_wake_up_context
from memory.utils.logger import activity_log


DEFAULT_DB_PATH = os.path.expanduser("~/.memory/memory.db")


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
    """Open the memory DB, bootstrapping it if needed."""
    directory = os.path.dirname(db_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        return bootstrap_db(db_path)
    return open_db(db_path)


def open_existing_memory_db(db_path: str = DEFAULT_DB_PATH):
    """Open the memory DB only if it already exists and is non-empty."""
    if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
        return None
    return open_db(db_path)


def save_session_to_memory(
    conn,
    *,
    session_id: str,
    agent: str,
    turns: list[dict],
    started_at: str,
    updated_at: str,
    metadata: dict | None = None,
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
    )


def retrieve_prompt_memory(
    conn,
    prompt: str,
    *,
    include_working_memory: bool,
    session_id: str | None = None,
    embed_fn=None,
) -> WakeUpContext:
    """Fetch memory context for one prompt through the shared retrieval seam."""
    kwargs = {"include_working_memory": include_working_memory, "session_id": session_id}
    if embed_fn is not None:
        kwargs["embed_fn"] = embed_fn
    context = retrieve_wake_up_context(conn, prompt, **kwargs)
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
