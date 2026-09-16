"""Persistence and retrieval helpers for structured procedural memories."""

from __future__ import annotations

import json
import sqlite3

from memory.contracts import ExtractedProcedure, ProceduralMemory
from memory.db import insert_procedural, search_procedural_semantic
from memory.llm.inference import embed_text
from memory.procedural.extractor import log_procedural_event


def build_procedural_semantic_text(procedure: ExtractedProcedure) -> str:
    """Render one extracted procedure into deterministic retrieval text."""
    parts = [procedure.title.strip(), procedure.summary.strip()]
    if procedure.steps:
        parts.append("Steps: " + "; ".join(step.strip() for step in procedure.steps if step.strip()))
    if procedure.trigger_phrases:
        parts.append("Useful for: " + "; ".join(item.strip() for item in procedure.trigger_phrases if item.strip()))
    if procedure.tools:
        parts.append("Tools: " + "; ".join(item.strip() for item in procedure.tools if item.strip()))
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."



def save_extracted_procedure(
    conn: sqlite3.Connection,
    procedure: ExtractedProcedure,
    *,
    session_id: str,
    updated_at: str,
    source: str = "procedural_extractor",
    embed_fn=embed_text,
) -> str:
    """Persist one extracted procedure to procedural_memory."""
    memory = ProceduralMemory(
        title=procedure.title,
        summary=procedure.summary,
        steps=procedure.steps,
        trigger_phrases=procedure.trigger_phrases,
        tools=procedure.tools,
        confidence=procedure.confidence,
        source_quote=procedure.source_quote,
        source_session_id=session_id,
        updated_at=updated_at,
    )

    semantic_text = build_procedural_semantic_text(procedure)
    embedding = None
    try:
        embedding = embed_fn(semantic_text)
    except Exception as exc:
        log_procedural_event(
            "validation_error",
            source=source,
            session_id=session_id,
            error=f"embedding_failed: {exc}",
        )

    saved_id = insert_procedural(
        conn,
        session_id=memory.source_session_id,
        title=memory.title,
        summary=memory.summary,
        updated_at=memory.updated_at.isoformat(),
        details={
            "steps": list(memory.steps),
            "trigger_phrases": list(memory.trigger_phrases),
            "tools": list(memory.tools),
            "confidence": memory.confidence,
            "source_quote": memory.source_quote,
            "source": source,
            "semantic_text": semantic_text,
        },
        embedding=embedding,
    )
    log_procedural_event(
        "persist_result",
        source=source,
        session_id=session_id,
        persisted_record_ids=[saved_id],
        validated_object=memory.model_dump(mode="json"),
    )
    return saved_id



def list_session_procedures(conn: sqlite3.Connection, *, session_id: str) -> list[dict]:
    """Return procedural memories saved for one session in insertion order."""
    rows = conn.execute(
        """
        SELECT rowid, id, session_id, title, summary, updated_at, details
        FROM procedural_memory
        WHERE session_id = ?
        ORDER BY rowid ASC
        """,
        (session_id,),
    ).fetchall()

    result: list[dict] = []
    for row in rows:
        details = json.loads(row["details"] or "{}")
        result.append(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "title": row["title"],
                "summary": row["summary"],
                "updated_at": row["updated_at"],
                "steps": details.get("steps", []),
                "trigger_phrases": details.get("trigger_phrases", []),
                "tools": details.get("tools", []),
                "confidence": details.get("confidence"),
                "source_quote": details.get("source_quote"),
                "source": details.get("source", "procedural_extractor"),
                "semantic_text": details.get("semantic_text", ""),
            }
        )
    return result



def retrieve_procedural_memories(
    conn: sqlite3.Connection,
    query: str,
    *,
    query_vector: list[float] | None = None,
    embed_fn=embed_text,
    min_similarity: float = 0.72,
    limit: int = 2,
    source: str = "manual_debug",
) -> list[dict]:
    """Retrieve semantically similar procedural memories with dedicated logging."""
    log_procedural_event(
        "retrieve_start",
        source=source,
        query=query,
        retrieval_prompt=query,
    )
    if query_vector is None:
        query_vector = embed_fn(query)
    results = [
        row for row in search_procedural_semantic(conn, query_vector, limit=limit)
        if row.get("similarity", 0.0) >= min_similarity
    ]
    log_procedural_event(
        "retrieve_result",
        source=source,
        query=query,
        similarity_ranking=[
            {"id": row.get("id"), "similarity": row.get("similarity")} for row in results
        ],
        retrieved_records=results,
    )
    return results
