from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from memory.vectors import cosine_distance, pack_vector
from memory.db._utils import _json_loads, _utc_now
from memory.db.typed_memory_fts import annotate_keyword_rows, delete_fts_rows, index_episodic_memory_fts, _build_match_query


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
    existing = conn.execute(
        "SELECT id FROM episodic_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    ep_id = existing["id"] if existing else str(uuid.uuid4())
    details_payload = details or {}
    conn.execute(
        """
        INSERT INTO episodic_memory (id, session_id, title, abstract, happened_at, details, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          title = excluded.title,
          abstract = excluded.abstract,
          happened_at = excluded.happened_at,
          details = excluded.details,
          embedding = excluded.embedding
        """,
        (
            ep_id,
            session_id,
            title,
            abstract,
            happened_at or _utc_now(),
            json.dumps(details_payload),
            pack_vector(embedding) if embedding is not None else None,
        ),
    )
    index_episodic_memory_fts(
        conn,
        memory_id=ep_id,
        session_id=session_id,
        title=title,
        abstract=abstract,
        details=details_payload,
    )
    conn.commit()
    return ep_id


def list_recent_episodic(conn: sqlite3.Connection, limit: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT id, session_id, title, abstract, happened_at, details FROM episodic_memory ORDER BY happened_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_episodic_row_to_memory_row(row) for row in rows]


def search_episodic_semantic(
    conn: sqlite3.Connection,
    query_vector: list[float],
    limit: int | None = 3,
    *,
    session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    allowed_session_ids = None if session_ids is None else {session_id for session_id in session_ids if session_id}
    rows = conn.execute(
        "SELECT id, session_id, title, abstract, happened_at, details, embedding FROM episodic_memory WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        if allowed_session_ids is not None and row["session_id"] not in allowed_session_ids:
            continue
        distance = cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append(_episodic_row_to_memory_row(row, similarity=1.0 - distance))
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored if limit is None else scored[:limit]


def search_episodic_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int | None = 3,
    *,
    session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    match_query = _build_match_query(query)
    if not match_query:
        return []
    allowed_session_ids = None if session_ids is None else sorted({session_id for session_id in session_ids if session_id})
    params: list[object] = [match_query]
    session_clause = ""
    if allowed_session_ids is not None:
        if not allowed_session_ids:
            return []
        placeholders = ", ".join("?" for _ in allowed_session_ids)
        session_clause = f" AND episodic_memory.session_id IN ({placeholders})"
        params.extend(allowed_session_ids)
    sql = f"""
        SELECT episodic_memory.id, episodic_memory.session_id, episodic_memory.title,
               episodic_memory.abstract, episodic_memory.happened_at, episodic_memory.details,
               bm25(episodic_memory_fts, 6.0, 3.0, 2.0) AS keyword_rank
        FROM episodic_memory_fts
        JOIN episodic_memory ON episodic_memory.id = episodic_memory_fts.memory_id
        WHERE episodic_memory_fts MATCH ?{session_clause}
        ORDER BY keyword_rank, episodic_memory.happened_at DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    results: list[dict] = []
    for row in rows:
        payload = _episodic_row_to_memory_row(row)
        payload["keyword_rank"] = float(row["keyword_rank"])
        results.append(payload)
    return annotate_keyword_rows(results)


def prune_stale_episodic(conn: sqlite3.Connection, *, days: int = 90) -> int:
    """Delete episodic memories older than `days` days. Returns count deleted."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    stale_ids = [
        row["id"]
        for row in conn.execute("SELECT id FROM episodic_memory WHERE happened_at < ?", (cutoff,)).fetchall()
    ]
    cursor = conn.execute("DELETE FROM episodic_memory WHERE happened_at < ?", (cutoff,))
    delete_fts_rows(conn, "episodic_memory_fts", stale_ids)
    conn.commit()
    return cursor.rowcount
