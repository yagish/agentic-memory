from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterable

from memory.vectors import cosine_distance, pack_vector
from memory.db._utils import _json_loads, _utc_now
from memory.db.typed_memory_fts import annotate_keyword_rows, delete_fts_rows, index_episodic_memory_fts, _build_match_query

_MAX_REINFORCEMENT_SIGNALS = 5


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = str(value)
        if parsed.endswith("Z"):
            parsed = parsed[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(parsed)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)
    except Exception:
        return None


def _episodic_retention_extension_days(*, days: int, reinforcement_count: int) -> int:
    bounded_count = max(0, min(int(reinforcement_count or 0), _MAX_REINFORCEMENT_SIGNALS))
    extension_per_signal = max(1, days // 3)
    return bounded_count * extension_per_signal



def _episodic_row_to_memory_row(row: sqlite3.Row, *, similarity: float | None = None) -> dict:
    details = _json_loads(row["details"], {})
    row_keys = set(row.keys())
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
        "retrieval_count": int(row["retrieval_count"] or 0) if "retrieval_count" in row_keys else 0,
        "last_retrieved_at": row["last_retrieved_at"] if "last_retrieved_at" in row_keys else None,
        "reinforcement_count": int(row["reinforcement_count"] or 0) if "reinforcement_count" in row_keys else 0,
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
        """
        SELECT id, session_id, title, abstract, happened_at, details,
               retrieval_count, last_retrieved_at, reinforcement_count
        FROM episodic_memory
        ORDER BY happened_at DESC
        LIMIT ?
        """,
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
        """
        SELECT id, session_id, title, abstract, happened_at, details, embedding,
               retrieval_count, last_retrieved_at, reinforcement_count
        FROM episodic_memory
        WHERE embedding IS NOT NULL
        """
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
               episodic_memory.retrieval_count, episodic_memory.last_retrieved_at,
               episodic_memory.reinforcement_count,
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



def reinforce_episodic_memories(
    conn: sqlite3.Connection,
    memory_ids: Iterable[str],
    *,
    increment_retrieval: bool = False,
    retrieved_at: str | None = None,
) -> int:
    unique_ids = sorted({str(memory_id) for memory_id in memory_ids if memory_id})
    if not unique_ids:
        return 0

    effective_retrieved_at = retrieved_at or _utc_now()
    updated = 0
    for memory_id in unique_ids:
        if increment_retrieval:
            cursor = conn.execute(
                """
                UPDATE episodic_memory
                SET retrieval_count = COALESCE(retrieval_count, 0) + 1,
                    last_retrieved_at = ?,
                    reinforcement_count = COALESCE(reinforcement_count, 0) + 1
                WHERE id = ?
                """,
                (effective_retrieved_at, memory_id),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE episodic_memory
                SET reinforcement_count = COALESCE(reinforcement_count, 0) + 1
                WHERE id = ?
                """,
                (memory_id,),
            )
        updated += max(0, cursor.rowcount)
    conn.commit()
    return updated



def prune_stale_episodic(conn: sqlite3.Connection, *, days: int = 90, now: str | None = None) -> int:
    """Delete episodic memories whose reinforced retention window has elapsed."""
    current_time = _parse_timestamp(now) or datetime.now(timezone.utc)
    rows = conn.execute(
        "SELECT id, happened_at, last_retrieved_at, reinforcement_count FROM episodic_memory"
    ).fetchall()

    stale_ids: list[str] = []
    for row in rows:
        happened_at = _parse_timestamp(row["happened_at"])
        last_retrieved_at = _parse_timestamp(row["last_retrieved_at"])
        reference_times = [timestamp for timestamp in (happened_at, last_retrieved_at) if timestamp is not None]
        if not reference_times:
            stale_ids.append(row["id"])
            continue

        retention_days = days + _episodic_retention_extension_days(
            days=days,
            reinforcement_count=int(row["reinforcement_count"] or 0),
        )
        expires_at = max(reference_times) + timedelta(days=retention_days)
        if expires_at < current_time:
            stale_ids.append(row["id"])

    if not stale_ids:
        return 0

    placeholders = ", ".join("?" for _ in stale_ids)
    conn.execute(f"DELETE FROM episodic_memory WHERE id IN ({placeholders})", stale_ids)
    delete_fts_rows(conn, "episodic_memory_fts", stale_ids)
    conn.commit()
    return len(stale_ids)
