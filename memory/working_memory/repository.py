"""Persistence and retrieval helpers for structured working memory."""

from __future__ import annotations

import json
import sqlite3

from memory.contracts import ExtractedWorkingMemory, WorkingMemory
from memory.db import get_working_memory, upsert_working_memory
from memory.llm.inference import embed_text
from memory.working_memory.extractor import log_working_memory_event


def build_working_memory_semantic_text(memory: ExtractedWorkingMemory) -> str:
    parts = [
        f"Current goal: {memory.current_goal.strip()}",
        f"Current focus: {memory.current_focus.strip()}",
        "Active tasks: " + "; ".join(item.strip() for item in memory.active_tasks if item.strip()),
        f"Next step: {memory.next_step.strip()}",
        f"Status: {memory.status}",
    ]
    if memory.constraints:
        parts.append("Constraints: " + "; ".join(item.strip() for item in memory.constraints if item.strip()))
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."


def save_extracted_working_memory(
    conn: sqlite3.Connection,
    working_memory: ExtractedWorkingMemory,
    *,
    session_id: str,
    updated_at: str,
    source: str = "working_memory_extractor",
    embed_fn=embed_text,
) -> str:
    memory = WorkingMemory(
        current_goal=working_memory.current_goal,
        current_focus=working_memory.current_focus,
        active_tasks=working_memory.active_tasks,
        constraints=working_memory.constraints,
        next_step=working_memory.next_step,
        status=working_memory.status,
        confidence=working_memory.confidence,
        source_quote=working_memory.source_quote,
        source_session_id=session_id,
        updated_at=updated_at,
    )

    semantic_text = build_working_memory_semantic_text(working_memory)
    embedding = None
    try:
        embedding = embed_fn(semantic_text)
    except Exception as exc:
        log_working_memory_event(
            "validation_error",
            source=source,
            session_id=session_id,
            error=f"embedding_failed: {exc}",
        )

    saved_id = upsert_working_memory(
        conn,
        session_id=memory.source_session_id,
        current_goal=memory.current_goal,
        current_focus=memory.current_focus,
        next_step=memory.next_step,
        status=memory.status,
        updated_at=memory.updated_at.isoformat(),
        details={
            "active_tasks": list(memory.active_tasks),
            "constraints": list(memory.constraints),
            "confidence": memory.confidence,
            "source_quote": memory.source_quote,
            "source": source,
            "semantic_text": semantic_text,
        },
        embedding=embedding,
    )
    log_working_memory_event(
        "persist_result",
        source=source,
        session_id=session_id,
        persisted_record_ids=[saved_id],
        validated_object=memory.model_dump(mode="json"),
    )
    return saved_id


def retrieve_working_memory(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    source: str = "manual_debug",
) -> dict | None:
    log_working_memory_event(
        "retrieve_start",
        source=source,
        session_id=session_id,
        query=session_id,
        retrieval_prompt=session_id,
    )
    result = get_working_memory(conn, session_id=session_id)
    log_working_memory_event(
        "retrieve_result",
        source=source,
        session_id=session_id,
        retrieved_records=[result] if result else [],
    )
    return result


def list_working_memories(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, session_id, current_goal, current_focus, next_step, status, updated_at, details
        FROM working_memory
        ORDER BY updated_at DESC
        """
    ).fetchall()
    result: list[dict] = []
    for row in rows:
        details = json.loads(row["details"] or "{}")
        result.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "current_goal": row["current_goal"],
                "current_focus": row["current_focus"],
                "next_step": row["next_step"],
                "status": row["status"],
                "updated_at": row["updated_at"],
                "active_tasks": details.get("active_tasks", []),
                "constraints": details.get("constraints", []),
                "confidence": details.get("confidence"),
                "source_quote": details.get("source_quote"),
                "source": details.get("source", "working_memory_extractor"),
                "semantic_text": details.get("semantic_text", ""),
            }
        )
    return result
