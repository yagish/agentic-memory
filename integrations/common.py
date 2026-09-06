"""Shared memory integration helpers for agent-specific adapters.

This module keeps save/retrieve policy independent from any specific agent
runtime. Claude hooks and pi extensions should call these functions instead of
re-implementing DB open/save/retrieval behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from memory.db import bootstrap_db, open_db
from memory.fact_renderer import render_fact_answer
from memory.ingest_pipeline import IngestOutcome, ingest_session
from memory.retrieval import WakeUpContext, build_wake_up_injection, retrieve_wake_up_context


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
    embed_fn=None,
) -> WakeUpContext:
    """Fetch memory context for one prompt through the shared retrieval seam."""
    kwargs = {"include_working_memory": include_working_memory}
    if embed_fn is not None:
        kwargs["embed_fn"] = embed_fn
    return retrieve_wake_up_context(conn, prompt, **kwargs)


def decide_prompt_memory_action(prompt: str, context: WakeUpContext) -> RecallOutcome:
    """Decide whether memory should answer directly or enrich the next turn."""
    if context.facts and not context.episodic:
        answer = render_fact_answer(prompt, [str(fact.get("content", "")) for fact in context.facts])
        if answer:
            return RecallOutcome(context=context, answer=answer)
        return RecallOutcome(context=context)

    injection = build_wake_up_injection(context)
    return RecallOutcome(context=context, injection=injection)
