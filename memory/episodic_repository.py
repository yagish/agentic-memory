"""Persistence and retrieval helpers for structured episodic memories."""

from __future__ import annotations

import json
import sqlite3

from memory.contracts import EpisodicMemory, ExtractedEpisode
from memory.db import insert_episodic, list_recent_episodic, search_episodic_semantic
from memory.episodic import log_episodic_event
from memory.inference import embed_text


def save_extracted_episode(
    conn: sqlite3.Connection,
    episode: ExtractedEpisode,
    *,
    session_id: str,
    happened_at: str,
    source: str = "episodic_extractor",
    embed_fn=embed_text,
) -> str:
    """Persist one extracted episode to episodic_memory."""
    memory = EpisodicMemory(
        title=episode.title,
        abstract=episode.abstract,
        participants=episode.participants,
        decisions=episode.decisions,
        outcomes=episode.outcomes,
        follow_ups=episode.follow_ups,
        confidence=episode.confidence,
        source_quote=episode.source_quote,
        source_session_id=session_id,
        happened_at=happened_at,
    )

    embedding = None
    try:
        embedding = embed_fn(f"{memory.title}\n{memory.abstract}")
    except Exception as exc:
        log_episodic_event(
            "validation_error",
            source=source,
            session_id=session_id,
            error=f"embedding_failed: {exc}",
        )

    saved_id = insert_episodic(
        conn,
        session_id=memory.source_session_id,
        title=memory.title,
        abstract=memory.abstract,
        happened_at=memory.happened_at.isoformat(),
        details={
            "participants": list(memory.participants),
            "decisions": list(memory.decisions),
            "outcomes": list(memory.outcomes),
            "follow_ups": list(memory.follow_ups),
            "confidence": memory.confidence,
            "source_quote": memory.source_quote,
            "source": source,
        },
        embedding=embedding,
    )
    log_episodic_event(
        "persist_result",
        source=source,
        session_id=session_id,
        persisted_record_ids=[saved_id],
        validated_object=memory.model_dump(mode="json"),
    )
    return saved_id


def list_session_episodes(conn: sqlite3.Connection, *, session_id: str) -> list[dict]:
    """Return episodic memories saved for one session in insertion order."""
    rows = conn.execute(
        """
        SELECT rowid, id, session_id, title, abstract, happened_at, details
        FROM episodic_memory
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
                "abstract": row["abstract"],
                "happened_at": row["happened_at"],
                "participants": details.get("participants", []),
                "decisions": details.get("decisions", []),
                "outcomes": details.get("outcomes", []),
                "follow_ups": details.get("follow_ups", []),
                "confidence": details.get("confidence"),
                "source_quote": details.get("source_quote"),
                "source": details.get("source", "episodic_extractor"),
            }
        )
    return result


def list_recent_episodic_memories(
    conn: sqlite3.Connection,
    *,
    limit: int = 5,
    source: str = "manual_debug",
) -> list[dict]:
    """Return the most recent episodic memories with dedicated logging."""
    results = list_recent_episodic(conn, limit=limit)
    log_episodic_event(
        "retrieve_recent_result",
        source=source,
        retrieved_records=results,
    )
    return results



def retrieve_episodic_memories(
    conn: sqlite3.Connection,
    query: str,
    *,
    query_vector: list[float] | None = None,
    embed_fn=embed_text,
    min_similarity: float = 0.72,
    limit: int = 3,
    source: str = "manual_debug",
) -> list[dict]:
    """Retrieve semantically similar episodic memories with dedicated logging."""
    log_episodic_event(
        "retrieve_start",
        source=source,
        query=query,
        retrieval_prompt=query,
    )
    if query_vector is None:
        query_vector = embed_fn(query)
    results = [
        row for row in search_episodic_semantic(conn, query_vector, limit=limit)
        if row.get("similarity", 0.0) >= min_similarity
    ]
    log_episodic_event(
        "retrieve_result",
        source=source,
        query=query,
        similarity_ranking=[
            {"id": row.get("id"), "similarity": row.get("similarity")} for row in results
        ],
        retrieved_records=results,
    )
    return results
