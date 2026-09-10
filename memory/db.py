"""Minimal SQLite storage layer for sessions and structured memory types.

The retained persistent tables are:
- sessions
- facts
- episodic_memory
- procedural_memory
- working_memory
- session_memory

Older storage concerns (chunks, retrievals, compaction, compression,
clustering, response cache) were removed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

from memory.fact_text import build_canonical_fact_content, build_semantic_fact_text
from memory.vectors import cosine_distance, embed, pack_vector


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id           TEXT PRIMARY KEY,
  agent                TEXT NOT NULL DEFAULT 'claude',
  started_at           TEXT,
  updated_at           TEXT,
  turn_count           INTEGER,
  transcript           TEXT,
  metadata             TEXT,
  daemon_processed_at  TEXT
);

CREATE TABLE IF NOT EXISTS facts (
  id               TEXT PRIMARY KEY,
  entity           TEXT NOT NULL,
  attribute        TEXT NOT NULL,
  value            TEXT NOT NULL,
  semantic_content TEXT NOT NULL,
  tags             TEXT,
  source           TEXT,
  session_id       TEXT,
  created_at       TEXT,
  updated_at       TEXT,
  embedding        BLOB
);

CREATE TABLE IF NOT EXISTS episodic_memory (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  title       TEXT NOT NULL,
  abstract    TEXT NOT NULL,
  happened_at TEXT NOT NULL,
  details     TEXT,
  embedding   BLOB
);

CREATE TABLE IF NOT EXISTS procedural_memory (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  title       TEXT NOT NULL,
  summary     TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  details     TEXT,
  embedding   BLOB
);

CREATE TABLE IF NOT EXISTS working_memory (
  id            TEXT PRIMARY KEY,
  session_id    TEXT NOT NULL UNIQUE,
  current_goal  TEXT NOT NULL,
  current_focus TEXT NOT NULL,
  next_step     TEXT NOT NULL,
  status        TEXT NOT NULL,
  updated_at    TEXT NOT NULL,
  details       TEXT,
  embedding     BLOB
);

CREATE TABLE IF NOT EXISTS session_memory (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL UNIQUE,
  title        TEXT NOT NULL,
  summary      TEXT NOT NULL,
  left_off_at  TEXT NOT NULL,
  updated_at   TEXT NOT NULL,
  details      TEXT,
  embedding    BLOB
);
"""



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()



def _json_loads(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback



def _session_text_from_turns(turns: list[dict]) -> str:
    parts: list[str] = []
    for turn in turns:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            parts.append(f"{role}: {content}")
    return "\n".join(parts)



def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    if path != ":memory:":
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
    return conn



def _fact_payload(
    *,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    semantic_content: str | None,
) -> tuple[str, str, str, str]:
    if not entity or not attribute or value is None:
        raise ValueError("fact requires entity, attribute, and value")
    semantic = (semantic_content or build_semantic_fact_text(entity, attribute, value)).strip()
    return entity, attribute, value.strip(), semantic



def _fact_dict_from_row(row: sqlite3.Row | dict) -> dict:
    data = dict(row)
    data["content"] = build_canonical_fact_content(data["entity"], data["attribute"], data["value"])
    return data



def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()



def bootstrap_db(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    conn = open_db(path)
    try:
        ensure_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn



def init_db(path: str) -> sqlite3.Connection:
    return bootstrap_db(path)



def log_retrieval(conn: sqlite3.Connection, tool: str, query: str | None, result_size: int) -> None:
    """Backward-compatible no-op after retrieval logging removal."""
    del conn, tool, query, result_size



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



def insert_fact(
    conn: sqlite3.Connection,
    *,
    entity: str,
    attribute: str,
    value: str,
    semantic_content: str | None = None,
    tags: list[str] | None = None,
    source: str = "manual",
    session_id: str | None = None,
) -> str:
    fact_id = str(uuid.uuid4())
    now = _utc_now()

    entity, attribute, value, semantic_content = _fact_payload(
        entity=entity,
        attribute=attribute,
        value=value,
        semantic_content=semantic_content,
    )

    embedding_blob = None
    try:
        embedding_blob = pack_vector(embed(semantic_content))
    except Exception:
        pass
    conn.execute(
        """
        INSERT INTO facts (id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (fact_id, entity, attribute, value, semantic_content, json.dumps(tags or []), source, session_id, now, now, embedding_blob),
    )
    conn.commit()
    return fact_id


def upsert_fact(
    conn: sqlite3.Connection,
    *,
    entity: str,
    attribute: str,
    value: str,
    semantic_content: str | None = None,
    tags: list[str] | None = None,
    source: str = "manual",
    session_id: str | None = None,
) -> str:
    """Insert or merge a fact by (entity, attribute).

    - New pair: insert a fresh row.
    - Same value: touch updated_at only (no write amplification).
    - Changed value: overwrite value, semantic_content, and embedding in place.

    Returns the fact id.
    """
    entity, attribute, value, semantic = _fact_payload(
        entity=entity, attribute=attribute, value=value, semantic_content=semantic_content
    )
    now = _utc_now()

    existing = conn.execute(
        "SELECT id, value FROM facts WHERE entity = ? AND attribute = ?",
        (entity, attribute),
    ).fetchone()

    if existing is None:
        fact_id = str(uuid.uuid4())
        embedding_blob = None
        try:
            embedding_blob = pack_vector(embed(semantic))
        except Exception:
            pass
        conn.execute(
            """
            INSERT INTO facts (id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (fact_id, entity, attribute, value, semantic, json.dumps(tags or []), source, session_id, now, now, embedding_blob),
        )
        conn.commit()
        return fact_id

    fact_id = existing["id"]

    if existing["value"] == value:
        conn.execute("UPDATE facts SET updated_at = ? WHERE id = ?", (now, fact_id))
    else:
        embedding_blob = None
        try:
            embedding_blob = pack_vector(embed(semantic))
        except Exception:
            pass
        conn.execute(
            """
            UPDATE facts
            SET value = ?, semantic_content = ?, updated_at = ?, session_id = ?, embedding = ?
            WHERE id = ?
            """,
            (value, semantic, now, session_id, embedding_blob, fact_id),
        )

    conn.commit()
    return fact_id



def update_fact(
    conn: sqlite3.Connection,
    fact_id: str,
    *,
    entity: str | None = None,
    attribute: str | None = None,
    value: str | None = None,
    semantic_content: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    now = _utc_now()
    existing = conn.execute("SELECT entity, attribute, value, semantic_content, tags FROM facts WHERE id = ?", (fact_id,)).fetchone()
    if existing is None:
        return False

    entity = existing["entity"] if entity is None else entity
    attribute = existing["attribute"] if attribute is None else attribute
    value = existing["value"] if value is None else value
    semantic_content = existing["semantic_content"] if semantic_content is None else semantic_content

    entity, attribute, value, semantic_content = _fact_payload(
        entity=entity,
        attribute=attribute,
        value=value,
        semantic_content=semantic_content,
    )
    new_tags = _json_loads(existing["tags"], []) if tags is None else tags

    embedding_blob = None
    try:
        embedding_blob = pack_vector(embed(semantic_content))
    except Exception:
        pass
    conn.execute(
        """
        UPDATE facts
        SET entity = ?, attribute = ?, value = ?, semantic_content = ?, tags = ?, updated_at = ?, embedding = ?
        WHERE id = ?
        """,
        (entity, attribute, value, semantic_content, json.dumps(new_tags), now, embedding_blob, fact_id),
    )
    conn.commit()
    return True



def delete_fact(conn: sqlite3.Connection, fact_id: str) -> bool:
    cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    conn.commit()
    return cursor.rowcount > 0



def list_facts(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at
        FROM facts
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            **_fact_dict_from_row(row),
            "tags": _json_loads(row["tags"], []),
        }
        for row in rows
    ]



def search_facts(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at,
               substr(semantic_content, 1, 120) AS snippet
        FROM facts
        WHERE semantic_content LIKE ?
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (f"%{query}%", limit),
    ).fetchall()
    return [
        {
            **_fact_dict_from_row(row),
            "tags": _json_loads(row["tags"], []),
        }
        for row in rows
    ]



def search_facts_semantic(conn: sqlite3.Connection, query_vector: list[float], limit: int = 5) -> list[dict]:
    rows = conn.execute(
        "SELECT id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at, embedding FROM facts WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        distance = cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append(
            {
                "id": row["id"],
                "content": build_canonical_fact_content(row["entity"], row["attribute"], row["value"]),
                "semantic_content": row["semantic_content"],
                "entity": row["entity"],
                "attribute": row["attribute"],
                "value": row["value"],
                "tags": _json_loads(row["tags"], []),
                "source": row["source"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "similarity": round(1.0 - distance, 4),
            }
        )
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:limit]



def get_unprocessed_sessions(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    rows = conn.execute(
        """
        SELECT session_id, agent, started_at, updated_at, turn_count, transcript, metadata, daemon_processed_at
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



def mark_session_processed(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "UPDATE sessions SET daemon_processed_at = ? WHERE session_id = ?",
        (_utc_now(), session_id),
    )
    conn.commit()



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
    procedure_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO procedural_memory (id, session_id, title, summary, updated_at, details, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            procedure_id,
            session_id,
            title,
            summary,
            updated_at or _utc_now(),
            json.dumps(details or {}),
            pack_vector(embedding) if embedding is not None else None,
        ),
    )
    conn.commit()
    return procedure_id



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



def search_procedural_semantic(conn: sqlite3.Connection, query_vector: list[float], limit: int = 2) -> list[dict]:
    rows = conn.execute(
        "SELECT id, session_id, title, summary, updated_at, details, embedding FROM procedural_memory WHERE embedding IS NOT NULL"
    ).fetchall()
    scored: list[dict] = []
    for row in rows:
        distance = cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append(_procedural_row_to_memory_row(row, similarity=1.0 - distance))
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:limit]


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


def get_working_memory(conn: sqlite3.Connection, *, session_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, session_id, current_goal, current_focus, next_step, status, updated_at, details FROM working_memory WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return _working_memory_row_to_memory_row(row, similarity=1.0)


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
