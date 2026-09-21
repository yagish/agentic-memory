"""Persistence and retrieval helpers for structured episodic memories."""

from __future__ import annotations

import json
import os
import sqlite3

from memory.contracts import EpisodicMemory, ExtractedEpisode
from memory.db import insert_episodic, list_recent_episodic, reinforce_episodic_memories, search_episodic_semantic
from memory.episodic.extractor import log_episodic_event
from memory.llm.inference import embed_text

_MIN_SIMILARITY = float(os.environ.get("MEMORY_MIN_SIMILARITY", "0.72"))
_EXTRACTION_REINFORCEMENT_SIMILARITY = float(
    os.environ.get("MEMORY_EPISODIC_REINFORCEMENT_SIMILARITY", "0.88")
)


def build_episodic_semantic_text(episode: ExtractedEpisode) -> str:
    parts = [episode.title.strip(), episode.abstract.strip()]
    if episode.participants:
        parts.append("Participants: " + "; ".join(item.strip() for item in episode.participants if item.strip()))
    if episode.decisions:
        parts.append("Decisions: " + "; ".join(item.strip() for item in episode.decisions if item.strip()))
    if episode.outcomes:
        parts.append("Outcomes: " + "; ".join(item.strip() for item in episode.outcomes if item.strip()))
    if episode.follow_ups:
        parts.append("Follow-ups: " + "; ".join(item.strip() for item in episode.follow_ups if item.strip()))
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."


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

    semantic_text = build_episodic_semantic_text(episode)
    embedding = None
    reinforced_memory_id = None
    try:
        embedding = embed_fn(semantic_text)
        for row in search_episodic_semantic(conn, embedding, limit=None):
            if row.get("session_id") == session_id:
                continue
            if row.get("similarity", 0.0) >= _EXTRACTION_REINFORCEMENT_SIMILARITY:
                reinforced_memory_id = row.get("id")
                break
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
            "semantic_text": semantic_text,
        },
        embedding=embedding,
    )
    if reinforced_memory_id:
        reinforce_episodic_memories(conn, [reinforced_memory_id])

    log_episodic_event(
        "persist_result",
        source=source,
        session_id=session_id,
        persisted_record_ids=[saved_id],
        reinforced_memory_id=reinforced_memory_id,
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
                "semantic_text": details.get("semantic_text", ""),
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
    min_similarity: float = _MIN_SIMILARITY,
    limit: int = 3,
    source: str = "manual_debug",
    session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Retrieve semantically similar episodic memories with dedicated logging."""
    log_episodic_event(
        "retrieve_start",
        source=source,
        query=query,
        retrieval_prompt=query,
    )
    if session_ids is not None and not {session_id for session_id in session_ids if session_id}:
        return []
    if query_vector is None:
        query_vector = embed_fn(query)
    results = [
        row
        for row in search_episodic_semantic(conn, query_vector, limit=max(limit * 3, limit), session_ids=session_ids)
        if row.get("similarity", 0.0) >= min_similarity
    ][:limit]
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
