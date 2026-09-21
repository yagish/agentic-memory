from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from memory.vectors import cosine_distance, embed, pack_vector
from memory.db._utils import _json_loads, _utc_now
from memory.facts.scope import classify_fact_scope, ensure_scope_tag


def _fact_payload(
    *,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    semantic_content: str | None,
) -> tuple[str, str, str, str]:
    if not entity or not attribute or value is None:
        raise ValueError("fact requires entity, attribute, and value")
    from memory.facts.text import build_semantic_fact_text  # lazy: avoids circular import
    semantic = (semantic_content or build_semantic_fact_text(entity, attribute, value)).strip()
    return entity, attribute, value.strip(), semantic


def _fact_dict_from_row(row: sqlite3.Row | dict) -> dict:
    from memory.facts.text import build_canonical_fact_content  # lazy: avoids circular import
    data = dict(row)
    tags = _json_loads(data.get("tags"), [])
    data["content"] = build_canonical_fact_content(data["entity"], data["attribute"], data["value"])
    data["tags"] = tags
    data["fact_scope"] = classify_fact_scope(
        entity=data.get("entity"),
        attribute=data.get("attribute"),
        value=data.get("value"),
        tags=tags,
        semantic_content=data.get("semantic_content"),
    )
    return data


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
        entity=entity, attribute=attribute, value=value, semantic_content=semantic_content,
    )
    embedding_blob = None
    try:
        embedding_blob = pack_vector(embed(semantic_content))
    except Exception:
        pass
    tags = ensure_scope_tag(
        tags,
        entity=entity,
        attribute=attribute,
        value=value,
        semantic_content=semantic_content,
    )
    conn.execute(
        """
        INSERT INTO facts (id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (fact_id, entity, attribute, value, semantic_content, json.dumps(tags), source, session_id, now, now, embedding_blob),
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

    Same value: touch updated_at only. Changed value: overwrite in place.
    """
    entity, attribute, value, semantic = _fact_payload(
        entity=entity, attribute=attribute, value=value, semantic_content=semantic_content,
    )
    tags = ensure_scope_tag(
        tags,
        entity=entity,
        attribute=attribute,
        value=value,
        semantic_content=semantic,
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
            (fact_id, entity, attribute, value, semantic, json.dumps(tags), source, session_id, now, now, embedding_blob),
        )
        conn.commit()
        return fact_id

    fact_id = existing["id"]
    if existing["value"] == value:
        conn.execute(
            "UPDATE facts SET updated_at = ?, session_id = ?, tags = ? WHERE id = ?",
            (now, session_id, json.dumps(tags), fact_id),
        )
    else:
        embedding_blob = None
        try:
            embedding_blob = pack_vector(embed(semantic))
        except Exception:
            pass
        conn.execute(
            """
            UPDATE facts
            SET value = ?, semantic_content = ?, tags = ?, updated_at = ?, session_id = ?, embedding = ?
            WHERE id = ?
            """,
            (value, semantic, json.dumps(tags), now, session_id, embedding_blob, fact_id),
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
    existing = conn.execute(
        "SELECT entity, attribute, value, semantic_content, tags FROM facts WHERE id = ?", (fact_id,)
    ).fetchone()
    if existing is None:
        return False

    entity = existing["entity"] if entity is None else entity
    attribute = existing["attribute"] if attribute is None else attribute
    value = existing["value"] if value is None else value
    semantic_content = existing["semantic_content"] if semantic_content is None else semantic_content

    entity, attribute, value, semantic_content = _fact_payload(
        entity=entity, attribute=attribute, value=value, semantic_content=semantic_content,
    )
    new_tags = _json_loads(existing["tags"], []) if tags is None else tags
    new_tags = ensure_scope_tag(
        new_tags,
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
    return [_fact_dict_from_row(row) for row in rows]


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
    return [_fact_dict_from_row(row) for row in rows]


def search_facts_semantic(conn: sqlite3.Connection, query_vector: list[float], limit: int | None = 5) -> list[dict]:
    from memory.facts.text import build_canonical_fact_content  # lazy: avoids circular import
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
                "fact_scope": classify_fact_scope(
                    entity=row["entity"],
                    attribute=row["attribute"],
                    value=row["value"],
                    tags=_json_loads(row["tags"], []),
                    semantic_content=row["semantic_content"],
                ),
                "source": row["source"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "similarity": round(1.0 - distance, 4),
            }
        )
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored if limit is None else scored[:limit]


def prune_stale_facts(conn: sqlite3.Connection, *, days: int = 180) -> int:
    """Delete facts not updated in the last `days` days. Returns count deleted."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    cursor = conn.execute("DELETE FROM facts WHERE updated_at < ?", (cutoff,))
    conn.commit()
    return cursor.rowcount
