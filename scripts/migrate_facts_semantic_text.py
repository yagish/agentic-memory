#!/usr/bin/env python3
"""One-off migration for structured fact storage.

Rebuilds the facts table so canonical content is no longer stored in SQLite.
The persisted truth becomes:
- entity
- attribute
- value
- semantic_content
- metadata
- embedding

Legacy rows are read from either:
- old schema with `content`
- transitional schema with both `content` and structured fields

Rows that cannot be represented as structured facts are skipped.

Run:
  python3 scripts/migrate_facts_semantic_text.py [--db ~/.memory/memory.db]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import embed, open_db, pack_vector  # noqa: E402
from memory.fact_text import generate_semantic_fact_text  # noqa: E402


DEFAULT_DB_PATH = os.path.expanduser("~/.memory/memory.db")
_FACT_RE = re.compile(r"^\s*([a-z0-9_]+)\.([a-z0-9_]+)\s*=\s*(.+?)\s*$", re.IGNORECASE)


NEW_FACTS_SCHEMA = """
CREATE TABLE facts (
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
)
"""



def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}



def _parse_legacy_content(content: str) -> tuple[str | None, str | None, str | None]:
    match = _FACT_RE.fullmatch(content or "")
    if not match:
        return None, None, None
    entity, attribute, value = match.groups()
    return entity.strip().lower(), attribute.strip().lower(), value.strip()



def migrate_fact_schema(db_path: str = DEFAULT_DB_PATH) -> dict[str, int]:
    conn = open_db(db_path)
    try:
        columns = _table_columns(conn, "facts")

        select_cols = [
            "id",
            "tags",
            "source",
            "session_id",
            "created_at",
            "updated_at",
            "embedding",
        ]
        for optional in ["content", "semantic_content", "entity", "attribute", "value"]:
            if optional in columns:
                select_cols.append(optional)

        rows = conn.execute(f"SELECT {', '.join(select_cols)} FROM facts").fetchall()

        migrated_rows: list[tuple] = []
        migrated = 0
        embedded = 0
        skipped = 0

        for row in rows:
            entity = row["entity"] if "entity" in columns else None
            attribute = row["attribute"] if "attribute" in columns else None
            value = row["value"] if "value" in columns else None
            embedding_blob = row["embedding"]

            if (not entity or not attribute or value is None) and "content" in columns:
                entity, attribute, value = _parse_legacy_content(row["content"])

            if not entity or not attribute or value is None:
                skipped += 1
                continue

            semantic_content = generate_semantic_fact_text(entity, attribute, value).strip()
            try:
                embedding_blob = pack_vector(embed(semantic_content))
                embedded += 1
            except Exception:
                pass

            migrated_rows.append(
                (
                    row["id"],
                    entity,
                    attribute,
                    value,
                    semantic_content,
                    row["tags"] if "tags" in columns else json.dumps([]),
                    row["source"] if "source" in columns else None,
                    row["session_id"] if "session_id" in columns else None,
                    row["created_at"] if "created_at" in columns else None,
                    row["updated_at"] if "updated_at" in columns else None,
                    embedding_blob,
                )
            )
            migrated += 1

        conn.execute("ALTER TABLE facts RENAME TO facts_old")
        conn.execute(NEW_FACTS_SCHEMA)
        conn.executemany(
            """
            INSERT INTO facts (id, entity, attribute, value, semantic_content, tags, source, session_id, created_at, updated_at, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            migrated_rows,
        )
        conn.execute("DROP TABLE facts_old")
        conn.commit()
        return {"migrated": migrated, "embedded": embedded, "skipped": skipped}
    finally:
        conn.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate facts to structured storage without canonical content")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to memory.db")
    args = parser.parse_args()

    result = migrate_fact_schema(args.db)
    print(
        f"migrated facts: migrated={result['migrated']} embedded={result['embedded']} skipped={result['skipped']} db={args.db}"
    )


if __name__ == "__main__":
    main()
