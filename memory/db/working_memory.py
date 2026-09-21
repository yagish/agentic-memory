from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from memory.vectors import pack_vector
from memory.db._utils import _json_loads, _utc_now


def _working_memory_row_to_memory_row(row: sqlite3.Row, *, similarity: float | None = None) -> dict:
    details = _json_loads(row["details"], {})
    payload = {
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
        "source": details.get("source"),
        "semantic_text": details.get("semantic_text", ""),
    }
    if similarity is not None:
        payload["similarity"] = round(similarity, 4)
    return payload


def upsert_working_memory(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    current_goal: str,
    current_focus: str,
    next_step: str,
    status: str,
    updated_at: str | None = None,
    details: dict | None = None,
    embedding: list[float] | None = None,
) -> str:
    existing = conn.execute(
        "SELECT id FROM working_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    memory_id = existing["id"] if existing else str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO working_memory (id, session_id, current_goal, current_focus, next_step, status, updated_at, details, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          current_goal = excluded.current_goal,
          current_focus = excluded.current_focus,
          next_step = excluded.next_step,
          status = excluded.status,
          updated_at = excluded.updated_at,
          details = excluded.details,
          embedding = excluded.embedding
        """,
        (
            memory_id,
            session_id,
            current_goal,
            current_focus,
            next_step,
            status,
            updated_at or _utc_now(),
            json.dumps(details or {}),
            pack_vector(embedding) if embedding is not None else None,
        ),
    )
    conn.commit()
    return memory_id


def get_working_memory(conn: sqlite3.Connection, *, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, session_id, current_goal, current_focus, next_step, status, updated_at, details FROM working_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return _working_memory_row_to_memory_row(row, similarity=1.0)


def prune_stale_working_memory(
    conn: sqlite3.Connection,
    *,
    days: int = 7,
    now: datetime | None = None,
) -> int:
    """Delete working-memory snapshots older than `days` days."""
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=days)).isoformat()
    cursor = conn.execute(
        "DELETE FROM working_memory WHERE datetime(updated_at) < datetime(?)",
        (cutoff,),
    )
    conn.commit()
    return cursor.rowcount
