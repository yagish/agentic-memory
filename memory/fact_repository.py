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
from memory.db import insert_fact
from memory.facts import normalize_extracted_facts


def build_fact_content(fact: ExtractedFact) -> str:
    """Render one extracted fact into the current facts-table text format."""
    return f"{fact.entity}.{fact.attribute} = {fact.value}"


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
    return tags


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
    saved_ids: list[str] = []
    for fact in normalize_extracted_facts(list(facts)):
        fact_id = insert_fact(
            conn,
            build_fact_content(fact),
            tags=build_fact_tags(fact),
            source=source,
            session_id=session_id,
        )
        saved_ids.append(fact_id)
    return saved_ids


def list_session_facts(conn: sqlite3.Connection, *, session_id: str) -> list[dict]:
    """Return facts saved for one session in insertion order."""
    rows = conn.execute(
        """
        SELECT rowid, id, content, tags, source, session_id, created_at, updated_at
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
                "content": row["content"],
                "tags": tags or [],
                "source": row["source"],
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )
    return result
