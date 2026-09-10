"""Explicit procedural-memory backfill helpers.

This module exists so historical sessions can be re-extracted in a one-off,
inspectable way instead of relying on hidden runtime migration.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from typing import Any

from memory.db import open_db
from memory.inference import embed_text
from memory.procedural.extractor import extract_procedure_from_session_text
from memory.procedural.repository import save_extracted_procedure


DEFAULT_DB_PATH = os.path.expanduser("~/.memory/memory.db")

ExtractFn = Callable[..., Any]
EmbedFn = Callable[[str], list[float]]


def _transcript_json_to_text(transcript_json: str | None) -> str:
    try:
        turns = json.loads(transcript_json or "[]")
    except Exception:
        turns = []

    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip()
        content = turn.get("content", "")
        if isinstance(content, str) and content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines).strip()


def list_backfill_candidate_sessions(
    conn: sqlite3.Connection,
    *,
    session_id: str | None = None,
    limit: int | None = None,
    include_existing: bool = False,
) -> list[dict]:
    query = """
        SELECT
            s.session_id,
            s.updated_at,
            s.transcript,
            EXISTS(
                SELECT 1 FROM procedural_memory p WHERE p.session_id = s.session_id
            ) AS has_procedure
        FROM sessions s
    """
    conditions: list[str] = []
    params: list[object] = []

    if session_id:
        conditions.append("s.session_id = ?")
        params.append(session_id)
    if not include_existing:
        conditions.append(
            "NOT EXISTS (SELECT 1 FROM procedural_memory p WHERE p.session_id = s.session_id)"
        )

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.updated_at ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    return [dict(row) for row in conn.execute(query, tuple(params)).fetchall()]


def backfill_procedural_memory(
    db_path: str = DEFAULT_DB_PATH,
    *,
    session_id: str | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    model: str | None = None,
    source: str = "procedural_backfill",
    extract_fn: ExtractFn = extract_procedure_from_session_text,
    embed_fn: EmbedFn = embed_text,
) -> dict[str, int]:
    conn = open_db(db_path)
    try:
        candidates = list_backfill_candidate_sessions(
            conn,
            session_id=session_id,
            limit=limit,
            include_existing=force,
        )

        stats = {
            "scanned": len(candidates),
            "created": 0,
            "replaced": 0,
            "would_create": 0,
            "would_replace": 0,
            "no_procedure": 0,
            "empty_transcript": 0,
            "errors": 0,
        }

        for row in candidates:
            session_text = _transcript_json_to_text(row.get("transcript"))
            if not session_text:
                stats["empty_transcript"] += 1
                continue

            try:
                procedure = extract_fn(
                    session_text,
                    model=model,
                    source=source,
                    session_id=row["session_id"],
                )
            except Exception:
                stats["errors"] += 1
                continue

            if procedure is None:
                stats["no_procedure"] += 1
                continue

            has_existing = bool(row.get("has_procedure"))
            if dry_run:
                stats["would_replace" if has_existing else "would_create"] += 1
                continue

            if has_existing:
                conn.execute("DELETE FROM procedural_memory WHERE session_id = ?", (row["session_id"],))
                conn.commit()
                stats["replaced"] += 1
            else:
                stats["created"] += 1

            save_extracted_procedure(
                conn,
                procedure,
                session_id=row["session_id"],
                updated_at=row.get("updated_at") or "",
                source=source,
                embed_fn=embed_fn,
            )

        return stats
    finally:
        conn.close()
