from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable

from memory.db._utils import _json_loads, _session_text_from_turns

_TYPED_MEMORY_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS episodic_memory_fts USING fts5(
  memory_id UNINDEXED,
  session_id UNINDEXED,
  title,
  body,
  anchors
);

CREATE VIRTUAL TABLE IF NOT EXISTS procedural_memory_fts USING fts5(
  memory_id UNINDEXED,
  session_id UNINDEXED,
  title,
  body,
  anchors
);

CREATE VIRTUAL TABLE IF NOT EXISTS session_memory_fts USING fts5(
  memory_id UNINDEXED,
  session_id UNINDEXED,
  title,
  body,
  anchors
);

CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
  session_id UNINDEXED,
  search_text
);
"""

_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_FTS_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "up",
    "use",
    "what",
    "when",
    "where",
    "with",
}


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _stringify_values(values: Iterable[object] | None) -> str:
    parts: list[str] = []
    for value in values or ():
        cleaned = _normalize_text(str(value))
        if cleaned:
            parts.append(cleaned)
    return "; ".join(parts)


def _fts_body(*parts: str | None) -> str:
    sentences = [part.rstrip(". ") for part in (_normalize_text(part) for part in parts) if part]
    if not sentences:
        return ""
    return ". ".join(sentences) + "."


def _session_search_text(transcript: str | list[dict] | None, compacted_text: str | None = None) -> str:
    turns = transcript if isinstance(transcript, list) else _json_loads(transcript, [])
    transcript_text = _session_text_from_turns(turns) if isinstance(turns, list) else ""
    return _fts_body(compacted_text, transcript_text)


def build_episodic_fts_document(title: str, abstract: str, details: dict | None = None) -> tuple[str, str, str]:
    details = details if isinstance(details, dict) else {}
    return (
        _normalize_text(title),
        _fts_body(
            abstract,
            details.get("semantic_text"),
            details.get("source_quote"),
        ),
        _stringify_values(
            [
                *list(details.get("participants", []) or []),
                *list(details.get("decisions", []) or []),
                *list(details.get("outcomes", []) or []),
                *list(details.get("follow_ups", []) or []),
            ]
        ),
    )


def build_procedural_fts_document(title: str, summary: str, details: dict | None = None) -> tuple[str, str, str]:
    details = details if isinstance(details, dict) else {}
    return (
        _normalize_text(title),
        _fts_body(
            summary,
            details.get("semantic_text"),
            details.get("source_quote"),
        ),
        _stringify_values(
            [
                *list(details.get("steps", []) or []),
                *list(details.get("trigger_phrases", []) or []),
                *list(details.get("tools", []) or []),
            ]
        ),
    )


def build_session_memory_fts_document(
    title: str,
    summary: str,
    left_off_at: str,
    details: dict | None = None,
) -> tuple[str, str, str]:
    details = details if isinstance(details, dict) else {}
    return (
        _normalize_text(title),
        _fts_body(
            summary,
            f"Left off at: {left_off_at}",
            details.get("semantic_text"),
            details.get("source_quote"),
        ),
        _stringify_values(
            [
                *list(details.get("what_was_tried", []) or []),
                *list(details.get("outcomes", []) or []),
                *list(details.get("next_steps", []) or []),
            ]
        ),
    )


def _replace_fts_row(
    conn: sqlite3.Connection,
    table: str,
    *,
    memory_id: str,
    session_id: str,
    title: str,
    body: str,
    anchors: str,
) -> None:
    conn.execute(f"DELETE FROM {table} WHERE memory_id = ?", (memory_id,))
    conn.execute(
        f"INSERT INTO {table} (memory_id, session_id, title, body, anchors) VALUES (?, ?, ?, ?, ?)",
        (memory_id, session_id, title, body, anchors),
    )


def index_episodic_memory_fts(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    session_id: str,
    title: str,
    abstract: str,
    details: dict | None = None,
) -> None:
    fts_title, body, anchors = build_episodic_fts_document(title, abstract, details)
    _replace_fts_row(
        conn,
        "episodic_memory_fts",
        memory_id=memory_id,
        session_id=session_id,
        title=fts_title,
        body=body,
        anchors=anchors,
    )


def index_procedural_memory_fts(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    session_id: str,
    title: str,
    summary: str,
    details: dict | None = None,
) -> None:
    fts_title, body, anchors = build_procedural_fts_document(title, summary, details)
    _replace_fts_row(
        conn,
        "procedural_memory_fts",
        memory_id=memory_id,
        session_id=session_id,
        title=fts_title,
        body=body,
        anchors=anchors,
    )


def index_session_memory_fts(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    session_id: str,
    title: str,
    summary: str,
    left_off_at: str,
    details: dict | None = None,
) -> None:
    fts_title, body, anchors = build_session_memory_fts_document(title, summary, left_off_at, details)
    _replace_fts_row(
        conn,
        "session_memory_fts",
        memory_id=memory_id,
        session_id=session_id,
        title=fts_title,
        body=body,
        anchors=anchors,
    )


def index_session_fts(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    transcript: str | list[dict] | None,
    compacted_text: str | None = None,
) -> None:
    search_text = _session_search_text(transcript, compacted_text)
    conn.execute("DELETE FROM sessions_fts WHERE session_id = ?", (session_id,))
    conn.execute(
        "INSERT INTO sessions_fts (session_id, search_text) VALUES (?, ?)",
        (session_id, search_text),
    )


def delete_fts_rows(conn: sqlite3.Connection, table: str, memory_ids: Iterable[str]) -> None:
    ids = [memory_id for memory_id in memory_ids if memory_id]
    if not ids:
        return
    placeholders = ", ".join("?" for _ in ids)
    conn.execute(f"DELETE FROM {table} WHERE memory_id IN ({placeholders})", ids)


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"] if isinstance(row, sqlite3.Row) else row[0])


def _backfill_episodic_memory_fts(conn: sqlite3.Connection) -> None:
    if _count_rows(conn, "episodic_memory") == _count_rows(conn, "episodic_memory_fts"):
        return
    conn.execute("DELETE FROM episodic_memory_fts")
    rows = conn.execute("SELECT id, session_id, title, abstract, details FROM episodic_memory").fetchall()
    for row in rows:
        index_episodic_memory_fts(
            conn,
            memory_id=row["id"],
            session_id=row["session_id"],
            title=row["title"],
            abstract=row["abstract"],
            details=_json_loads(row["details"], {}),
        )


def _backfill_procedural_memory_fts(conn: sqlite3.Connection) -> None:
    if _count_rows(conn, "procedural_memory") == _count_rows(conn, "procedural_memory_fts"):
        return
    conn.execute("DELETE FROM procedural_memory_fts")
    rows = conn.execute("SELECT id, session_id, title, summary, details FROM procedural_memory").fetchall()
    for row in rows:
        index_procedural_memory_fts(
            conn,
            memory_id=row["id"],
            session_id=row["session_id"],
            title=row["title"],
            summary=row["summary"],
            details=_json_loads(row["details"], {}),
        )


def _backfill_session_memory_fts(conn: sqlite3.Connection) -> None:
    if _count_rows(conn, "session_memory") == _count_rows(conn, "session_memory_fts"):
        return
    conn.execute("DELETE FROM session_memory_fts")
    rows = conn.execute("SELECT id, session_id, title, summary, left_off_at, details FROM session_memory").fetchall()
    for row in rows:
        index_session_memory_fts(
            conn,
            memory_id=row["id"],
            session_id=row["session_id"],
            title=row["title"],
            summary=row["summary"],
            left_off_at=row["left_off_at"],
            details=_json_loads(row["details"], {}),
        )


def _backfill_sessions_fts(conn: sqlite3.Connection) -> None:
    if _count_rows(conn, "sessions") == _count_rows(conn, "sessions_fts"):
        return
    conn.execute("DELETE FROM sessions_fts")
    rows = conn.execute("SELECT session_id, transcript, compacted_text FROM sessions").fetchall()
    for row in rows:
        index_session_fts(
            conn,
            session_id=row["session_id"],
            transcript=row["transcript"],
            compacted_text=row["compacted_text"],
        )


def ensure_typed_memory_fts(conn: sqlite3.Connection) -> None:
    conn.executescript(_TYPED_MEMORY_FTS_SCHEMA)
    _backfill_episodic_memory_fts(conn)
    _backfill_procedural_memory_fts(conn)
    _backfill_session_memory_fts(conn)
    _backfill_sessions_fts(conn)
    conn.commit()


def _build_match_query(query: str) -> str | None:
    raw_tokens = [token for token in _WORD_RE.findall((query or "").lower())]
    tokens = [token for token in raw_tokens if token not in _FTS_STOPWORDS] or raw_tokens
    ordered_tokens = list(dict.fromkeys(tokens))
    if not ordered_tokens:
        return None
    return " OR ".join(f'{token}*' for token in ordered_tokens)


def annotate_keyword_rows(rows: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for index, row in enumerate(rows):
        payload = dict(row)
        payload["keyword_hit"] = True
        payload["keyword_score"] = round(1.0 / (index + 1), 4)
        annotated.append(payload)
    return annotated
