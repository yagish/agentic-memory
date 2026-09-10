from __future__ import annotations

import json
import sqlite3

from memory.vectors import cosine_distance, embed, pack_vector
from memory.db._utils import _json_loads, _session_text_from_turns, _utc_now


def upsert_session(
    conn: sqlite3.Connection,
    session_id: str,
    agent: str,
    transcript: list[dict],
    started_at: str,
    updated_at: str,
    metadata: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sessions (session_id, agent, started_at, updated_at, turn_count, transcript, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          agent = excluded.agent,
          started_at = excluded.started_at,
          updated_at = excluded.updated_at,
          turn_count = excluded.turn_count,
          transcript = excluded.transcript,
          metadata = excluded.metadata,
          daemon_processed_at = NULL
        """,
        (
            session_id,
            agent,
            started_at,
            updated_at,
            len(transcript),
            json.dumps(transcript),
            json.dumps(metadata) if metadata is not None else None,
        ),
    )
    conn.commit()


def search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT session_id, agent, updated_at, substr(transcript, 1, 120) AS snippet
        FROM sessions
        WHERE transcript LIKE ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (f"%{query}%", limit),
    ).fetchall()
    return [dict(row) for row in rows]


def semantic_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    query_vector = embed(query)
    rows = conn.execute(
        "SELECT session_id, agent, updated_at, transcript FROM sessions"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        turns = _json_loads(row["transcript"], [])
        text = _session_text_from_turns(turns)
        if not text.strip():
            continue
        distance = cosine_distance(query_vector, pack_vector(embed(text)))
        scored.append(
            {
                "session_id": row["session_id"],
                "agent": row["agent"],
                "updated_at": row["updated_at"],
                "distance": distance,
            }
        )
    scored.sort(key=lambda item: item["distance"])
    return scored[:limit]


def hybrid_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    keyword_rows = search(conn, query, limit=limit)
    seen = {row["session_id"] for row in keyword_rows}
    semantic_rows = [row for row in semantic_search(conn, query, limit=limit) if row["session_id"] not in seen]
    return (keyword_rows + semantic_rows)[:limit]


def get_unprocessed_sessions(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT session_id, agent, started_at, updated_at, turn_count, transcript,
               metadata, daemon_processed_at, compacted_text
        FROM sessions
        WHERE daemon_processed_at IS NULL
        ORDER BY updated_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_session_by_id(conn: sqlite3.Connection, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT session_id, agent, started_at, updated_at, turn_count, transcript, metadata, daemon_processed_at FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def get_latest_session(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT session_id, agent, started_at, updated_at, turn_count, transcript, metadata, daemon_processed_at FROM sessions ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def save_session_compaction(conn: sqlite3.Connection, session_id: str, compacted_text: str) -> None:
    conn.execute(
        "UPDATE sessions SET compacted_text = ? WHERE session_id = ?",
        (compacted_text, session_id),
    )
    conn.commit()


def mark_session_processed(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "UPDATE sessions SET daemon_processed_at = ? WHERE session_id = ?",
        (_utc_now(), session_id),
    )
    conn.commit()
