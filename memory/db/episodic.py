from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from memory.vectors import cosine_distance, pack_vector
from memory.db._utils import _json_loads, _utc_now


def _episodic_row_to_memory_row(row: sqlite3.Row, *, similarity: float | None = None) -> dict:
    details = _json_loads(row["details"], {})
    payload = {
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
        "source": details.get("source"),
        "semantic_text": details.get("semantic_text", ""),
    }
    if similarity is not None:
        payload["similarity"] = round(similarity, 4)
    return payload


def insert_episodic(
    conn: sqlite3.Connection,
    session_id: str,
    title: str,
    abstract: str,
    happened_at: str | None = None,
    *,
    details: dict | None = None,
    embedding: list[float] | None = None,
) -> str:
    ep_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO episodic_memory (id, session_id, title, abstract, happened_at, details, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ep_id,
            session_id,
            title,
            abstract,
            happened_at or _utc_now(),
            json.dumps(details or {}),
            pack_vector(embedding) if embedding is not None else None,
        ),
    )
    conn.commit()
    return ep_id


def list_recent_episodic(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT id, session_id, title, abstract, happened_at, details FROM episodic_memory ORDER BY happened_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_episodic_row_to_memory_row(row) for row in rows]


def search_episodic_semantic(conn: sqlite3.Connection, query_vector: list[float], limit: int = 3) -> list[dict]:
    rows = conn.execute(
        "SELECT id, session_id, title, abstract, happened_at, details, embedding FROM episodic_memory WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        distance = cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append(_episodic_row_to_memory_row(row, similarity=1.0 - distance))
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:limit]


def prune_stale_episodic(conn: sqlite3.Connection, *, days: int = 90) -> int:
    """Delete episodic memories older than `days` days. Returns count deleted."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cursor = conn.execute("DELETE FROM episodic_memory WHERE happened_at < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount
