from __future__ import annotations

import json
import sqlite3
import uuid

from memory.vectors import cosine_distance, pack_vector
from memory.db._utils import _json_loads, _utc_now
from memory.db.typed_memory_fts import annotate_keyword_rows, index_procedural_memory_fts, _build_match_query


def _procedural_row_to_memory_row(row: sqlite3.Row, *, similarity: float | None = None) -> dict:
    details = _json_loads(row["details"], {})
    payload = {
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
        "source": details.get("source"),
        "semantic_text": details.get("semantic_text", ""),
    }
    if similarity is not None:
        payload["similarity"] = round(similarity, 4)
    return payload


def insert_procedural(
    conn: sqlite3.Connection,
    session_id: str,
    title: str,
    summary: str,
    updated_at: str | None = None,
    *,
    details: dict | None = None,
    embedding: list[float] | None = None,
) -> str:
    existing = conn.execute(
        "SELECT id FROM procedural_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    procedure_id = existing["id"] if existing else str(uuid.uuid4())
    details_payload = details or {}
    conn.execute(
        """
        INSERT INTO procedural_memory (id, session_id, title, summary, updated_at, details, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          title = excluded.title,
          summary = excluded.summary,
          updated_at = excluded.updated_at,
          details = excluded.details,
          embedding = excluded.embedding
        """,
        (
            procedure_id,
            session_id,
            title,
            summary,
            updated_at or _utc_now(),
            json.dumps(details_payload),
            pack_vector(embedding) if embedding is not None else None,
        ),
    )
    index_procedural_memory_fts(
        conn,
        memory_id=procedure_id,
        session_id=session_id,
        title=title,
        summary=summary,
        details=details_payload,
    )
    conn.commit()
    return procedure_id


def search_procedural_semantic(
    conn: sqlite3.Connection,
    query_vector: list[float],
    limit: int | None = 2,
    *,
    session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    allowed_session_ids = None if session_ids is None else {session_id for session_id in session_ids if session_id}
    rows = conn.execute(
        "SELECT id, session_id, title, summary, updated_at, details, embedding FROM procedural_memory WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        if allowed_session_ids is not None and row["session_id"] not in allowed_session_ids:
            continue
        distance = cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append(_procedural_row_to_memory_row(row, similarity=1.0 - distance))
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored if limit is None else scored[:limit]


def search_procedural_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int | None = 2,
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
        session_clause = f" AND procedural_memory.session_id IN ({placeholders})"
        params.extend(allowed_session_ids)
    sql = f"""
        SELECT procedural_memory.id, procedural_memory.session_id, procedural_memory.title,
               procedural_memory.summary, procedural_memory.updated_at, procedural_memory.details,
               bm25(procedural_memory_fts, 6.0, 3.0, 2.0) AS keyword_rank
        FROM procedural_memory_fts
        JOIN procedural_memory ON procedural_memory.id = procedural_memory_fts.memory_id
        WHERE procedural_memory_fts MATCH ?{session_clause}
        ORDER BY keyword_rank, procedural_memory.updated_at DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    results: list[dict] = []
    for row in rows:
        payload = _procedural_row_to_memory_row(row)
        payload["keyword_rank"] = float(row["keyword_rank"])
        results.append(payload)
    return annotate_keyword_rows(results)
