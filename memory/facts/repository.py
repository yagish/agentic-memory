"""Persistence helpers for structured extracted facts.

This module is the bridge between the new semantic fact extractor and the
existing SQLite facts table. It stores extracted facts in a deterministic,
reviewable shape without changing dashboard/logging behavior.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

from memory.contracts import ExtractedFact
from memory.facts.scope import classify_fact_scope, ensure_scope_tag
from memory.facts.text import build_canonical_fact_content, generate_semantic_fact_text
from memory.facts.extractor import normalize_extracted_facts


def build_fact_content(fact: ExtractedFact) -> str:
    """Render one extracted fact into the canonical deterministic text format."""
    return build_canonical_fact_content(fact.entity, fact.attribute, fact.value)



def build_fact_semantic_content(fact: ExtractedFact) -> str:
    """Render one extracted fact into model-generated retrieval text."""
    source_text = fact.source_quote or ("\n".join(fact.evidence) if fact.evidence else None)
    return generate_semantic_fact_text(
        fact.entity,
        fact.attribute,
        fact.value,
        source_text=source_text,
    )


def build_fact_tags(fact: ExtractedFact) -> list[str]:
    """Attach structured tags so later retrieval can filter by entity/attribute."""
    tags = [
        "memory_type:fact",
        "origin:llm_extractor",
        f"entity:{fact.entity}",
        f"attribute:{fact.attribute}",
    ]
    if fact.evidence:
        tags.append("has:evidence")
    if fact.source_quote:
        tags.append("has:source_quote")
    if fact.confidence is not None:
        tags.append("has:confidence")
    return ensure_scope_tag(
        tags,
        entity=fact.entity,
        attribute=fact.attribute,
        value=fact.value,
        source_quote=fact.source_quote,
        llm_scope=fact.scope,
    )


def save_extracted_facts(
    conn: sqlite3.Connection,
    facts: Iterable[ExtractedFact],
    *,
    session_id: str,
    source: str = "fact_extractor",
) -> list[str]:
    """Persist extracted facts into the existing facts table.

    Safety rules:
    - exact semantic duplicates are deduped before insert
    - conflicts are preserved as separate rows
    - insertion order matches first semantic appearance
    """
    from memory.db.facts import upsert_fact

    saved_ids: list[str] = []
    for fact in normalize_extracted_facts(list(facts)):
        fact_id = upsert_fact(
            conn,
            tags=build_fact_tags(fact),
            source=source,
            session_id=session_id,
            entity=fact.entity,
            attribute=fact.attribute,
            value=fact.value,
            semantic_content=build_fact_semantic_content(fact),
        )
        saved_ids.append(fact_id)
    return saved_ids


def list_session_facts(conn: sqlite3.Connection, *, session_id: str) -> list[dict]:
    """Return facts saved for one session in insertion order."""
    rows = conn.execute(
        """
        SELECT rowid, id, semantic_content, entity, attribute, value, tags, source, session_id, created_at, updated_at
        FROM facts
        WHERE session_id = ?
        ORDER BY rowid ASC
        """,
        (session_id,),
    ).fetchall()

    result: list[dict] = []
    for row in rows:
        tags = row["tags"]
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []
        result.append(
            {
                "id": row["id"],
                "content": build_canonical_fact_content(row["entity"], row["attribute"], row["value"]),
                "semantic_content": row["semantic_content"],
                "entity": row["entity"],
                "attribute": row["attribute"],
                "value": row["value"],
                "tags": tags or [],
                "fact_scope": classify_fact_scope(
                    entity=row["entity"],
                    attribute=row["attribute"],
                    value=row["value"],
                    tags=tags or [],
                    semantic_content=row["semantic_content"],
                ),
                "source": row["source"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return result
