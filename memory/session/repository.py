"""Persistence and retrieval helpers for compacted session memory."""

from __future__ import annotations

import json
import sqlite3

from memory.contracts import ExtractedSessionMemory, SessionMemory
from memory.db import search_session_memory_semantic, upsert_session_memory
from memory.llm.inference import embed_text
from memory.session.extractor import log_session_memory_event


def build_session_memory_semantic_text(memory: ExtractedSessionMemory) -> str:
    parts = [memory.title.strip(), memory.summary.strip(), f"Left off at: {memory.left_off_at.strip()}"]
    if memory.what_was_tried:
        parts.append("Tried: " + "; ".join(item.strip() for item in memory.what_was_tried if item.strip()))
    if memory.outcomes:
        parts.append("Outcomes: " + "; ".join(item.strip() for item in memory.outcomes if item.strip()))
    if memory.next_steps:
        parts.append("Next session: " + "; ".join(item.strip() for item in memory.next_steps if item.strip()))
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."


def save_extracted_session_memory(
    conn: sqlite3.Connection,
    session_memory: ExtractedSessionMemory,
    *,
    session_id: str,
    updated_at: str,
    source: str = "session_memory_extractor",
    embed_fn=embed_text,
) -> str:
    memory = SessionMemory(
        title=session_memory.title,
        summary=session_memory.summary,
        what_was_tried=session_memory.what_was_tried,
        outcomes=session_memory.outcomes,
        left_off_at=session_memory.left_off_at,
        next_steps=session_memory.next_steps,
        confidence=session_memory.confidence,
        source_quote=session_memory.source_quote,
        source_session_id=session_id,
        updated_at=updated_at,
    )

    semantic_text = build_session_memory_semantic_text(session_memory)
    embedding = None
    try:
        embedding = embed_fn(semantic_text)
    except Exception as exc:
        log_session_memory_event(
            "validation_error",
            source=source,
            session_id=session_id,
            error=f"embedding_failed: {exc}",
        )

    saved_id = upsert_session_memory(
        conn,
        session_id=memory.source_session_id,
        title=memory.title,
        summary=memory.summary,
        left_off_at=memory.left_off_at,
        updated_at=memory.updated_at.isoformat(),
        details={
            "what_was_tried": list(memory.what_was_tried),
            "outcomes": list(memory.outcomes),
            "next_steps": list(memory.next_steps),
            "confidence": memory.confidence,
            "source_quote": memory.source_quote,
            "source": source,
            "semantic_text": semantic_text,
        },
        embedding=embedding,
    )
    log_session_memory_event(
        "persist_result",
        source=source,
        session_id=session_id,
        persisted_record_ids=[saved_id],
        validated_object=memory.model_dump(mode="json"),
    )
    return saved_id


def list_session_memory_rows(conn: sqlite3.Connection, *, session_id: str | None = None) -> list[dict]:
    params: tuple = ()
    sql = (
        "SELECT rowid, id, session_id, title, summary, left_off_at, updated_at, details FROM session_memory"
    )
    if session_id is not None:
        sql += " WHERE session_id = ?"
        params = (session_id,)
    sql += " ORDER BY updated_at DESC"

    rows = conn.execute(sql, params).fetchall()
    result: list[dict] = []
    for row in rows:
        details = json.loads(row["details"] or "{}")
        result.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "title": row["title"],
                "summary": row["summary"],
                "left_off_at": row["left_off_at"],
                "updated_at": row["updated_at"],
                "what_was_tried": details.get("what_was_tried", []),
                "outcomes": details.get("outcomes", []),
                "next_steps": details.get("next_steps", []),
                "confidence": details.get("confidence"),
                "source_quote": details.get("source_quote"),
                "source": details.get("source", "session_memory_extractor"),
                "semantic_text": details.get("semantic_text", ""),
            }
        )
    return result


def retrieve_session_memories(
    conn: sqlite3.Connection,
    query: str,
    *,
    query_vector: list[float] | None = None,
    embed_fn=embed_text,
    min_similarity: float = 0.76,
    limit: int = 2,
    source: str = "manual_debug",
    exclude_session_id: str | None = None,
) -> list[dict]:
    log_session_memory_event(
        "retrieve_start",
        source=source,
        query=query,
        retrieval_prompt=query,
        exclude_session_id=exclude_session_id,
    )
    if query_vector is None:
        query_vector = embed_fn(query)
    results = [
        row
        for row in search_session_memory_semantic(conn, query_vector, limit=max(limit * 3, limit))
        if row.get("similarity", 0.0) >= min_similarity and row.get("session_id") != exclude_session_id
    ][:limit]
    log_session_memory_event(
        "retrieve_result",
        source=source,
        query=query,
        similarity_ranking=[
            {"id": row.get("id"), "similarity": row.get("similarity")} for row in results
        ],
        retrieved_records=results,
    )
    return results
