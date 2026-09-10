#!/usr/bin/env python3
"""One-off procedural-memory backfill.

Use this to extract procedural memories for historical sessions that were already
stored before procedural extraction existed, or to re-run procedural extraction
for existing sessions with --force.

Examples:
  python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --dry-run
  python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --limit 25
  python3 scripts/backfill_procedural_memory.py --db ~/.memory/memory.db --session-id <id> --force
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.procedural.backfill import DEFAULT_DB_PATH, backfill_procedural_memory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill procedural memory from stored sessions")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to memory.db")
    parser.add_argument("--session-id", help="Only process one stored session")
    parser.add_argument("--limit", type=int, help="Maximum number of sessions to scan")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be created/replaced without writing")
    parser.add_argument("--force", action="store_true", help="Re-extract even when a session already has procedural memory")
    parser.add_argument("--model", help="Optional Ollama model override for extraction")
    args = parser.parse_args()

    result = backfill_procedural_memory(
        args.db,
        session_id=args.session_id,
        limit=args.limit,
        dry_run=args.dry_run,
        force=args.force,
        model=args.model,
    )
    print(
        "procedural backfill: "
        f"scanned={result['scanned']} "
        f"created={result['created']} "
        f"replaced={result['replaced']} "
        f"would_create={result['would_create']} "
        f"would_replace={result['would_replace']} "
        f"no_procedure={result['no_procedure']} "
        f"empty_transcript={result['empty_transcript']} "
        f"errors={result['errors']} "
        f"db={args.db}"
    )


if __name__ == "__main__":
    main()
