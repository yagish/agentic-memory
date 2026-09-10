from __future__ import annotations

import json
import sqlite3
import uuid

from memory.vectors import cosine_distance, pack_vector
from memory.db._utils import _json_loads, _utc_now


def _session_memory_row_to_memory_row(row: sqlite3.Row, *, similarity: float | None = None) -> dict:
    details = _json_loads(row["details"], {})
    payload = {
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
        "source": details.get("source"),
        "semantic_text": details.get("semantic_text", ""),
    }
    if similarity is not None:
        payload["similarity"] = round(similarity, 4)
    return payload


def upsert_session_memory(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    title: str,
    summary: str,
    left_off_at: str,
    updated_at: str | None = None,
    details: dict | None = None,
    embedding: list[float] | None = None,
) -> str:
    existing = conn.execute(
        "SELECT id FROM session_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    memory_id = existing["id"] if existing else str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO session_memory (id, session_id, title, summary, left_off_at, updated_at, details, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          title = excluded.title,
          summary = excluded.summary,
          left_off_at = excluded.left_off_at,
          updated_at = excluded.updated_at,
          details = excluded.details,
          embedding = excluded.embedding
        """,
        (
            memory_id,
            session_id,
            title,
            summary,
            left_off_at,
            updated_at or _utc_now(),
            json.dumps(details or {}),
            pack_vector(embedding) if embedding is not None else None,
        ),
    )
    conn.commit()
    return memory_id


def search_session_memory_semantic(conn: sqlite3.Connection, query_vector: list[float], limit: int = 2) -> list[dict]:
    rows = conn.execute(
        "SELECT id, session_id, title, summary, left_off_at, updated_at, details, embedding FROM session_memory WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        distance = cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append(_session_memory_row_to_memory_row(row, similarity=1.0 - distance))
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:limit]
