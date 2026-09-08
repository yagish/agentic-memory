"""Minimal command-line tool for the simplified memory database."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.db import (
    bootstrap_db,
    delete_fact,
    insert_fact,
    open_db,
    search,
    semantic_search,
)
from memory.debug import enable_debug


DB_PATH = os.path.expanduser("~/.memory/memory.db")
DASHBOARD_URL = "http://127.0.0.1:7748"


def get_conn():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        print("No memory database found at", DB_PATH)
        print("Run: python3 cli.py bootstrap")
        sys.exit(1)
    return open_db(DB_PATH)


def cmd_bootstrap(_args):
    conn = bootstrap_db(DB_PATH)
    conn.close()
    print("Bootstrapped", DB_PATH)


def cmd_status(_args):
    conn = get_conn()
    row = conn.execute(
        "SELECT COUNT(*) AS sessions, COALESCE(SUM(turn_count), 0) AS turns, MIN(updated_at) AS oldest, MAX(updated_at) AS newest FROM sessions"
    ).fetchone()
    facts = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
    episodes = conn.execute("SELECT COUNT(*) AS c FROM episodic_memory").fetchone()["c"]
    procedures = conn.execute("SELECT COUNT(*) AS c FROM procedural_memory").fetchone()["c"]
    working = conn.execute("SELECT COUNT(*) AS c FROM working_memory").fetchone()["c"]
    conn.close()

    print("=== Memory Status ===")
    print(f"Sessions : {row['sessions']}")
    print(f"Turns    : {row['turns']}")
    print(f"Facts    : {facts}")
    print(f"Episodes : {episodes}")
    print(f"Procedures : {procedures}")
    print(f"Working : {working}")
    print(f"Oldest   : {row['oldest'] or 'none'}")
    print(f"Newest   : {row['newest'] or 'none'}")


def cmd_search(args):
    conn = get_conn()
    results = search(conn, args.query, limit=args.limit)
    conn.close()
    if not results:
        print(f"No results for: {args.query!r}")
        return
    for index, row in enumerate(results, start=1):
        print(f"[{index}] {row['session_id']}  {row['updated_at']}")
        print(f"    {row['snippet']}")


def cmd_semantic(args):
    conn = get_conn()
    results = semantic_search(conn, args.query, limit=args.limit)
    conn.close()
    if not results:
        print(f"No results for: {args.query!r}")
        return
    for index, row in enumerate(results, start=1):
        similarity = round((1 - row['distance'] / 2) * 100, 1)
        print(f"[{index}] {row['session_id']}  {row['updated_at']}  similarity={similarity}%")


def cmd_get_session(args):
    conn = get_conn()
    row = conn.execute(
        "SELECT session_id, agent, started_at, updated_at, turn_count, transcript FROM sessions WHERE session_id = ?",
        (args.session_id,),
    ).fetchone()
    conn.close()
    if row is None:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)
    print(json.dumps({key: row[key] for key in row.keys()}, indent=2))


def cmd_tail(args):
    conn = get_conn()
    rows = conn.execute(
        "SELECT session_id, updated_at, turn_count, transcript FROM sessions ORDER BY updated_at DESC LIMIT ?",
        (args.n,),
    ).fetchall()
    conn.close()
    for row in rows:
        try:
            transcript = json.loads(row["transcript"] or "[]")
            preview = next((turn.get("content", "") for turn in transcript if turn.get("role") == "user"), "")
        except Exception:
            preview = ""
        print(f"{row['updated_at']}  {row['session_id']}  turns={row['turn_count']}")
        print(f"    {preview[:140]}")


def cmd_add_fact(args):
    conn = bootstrap_db(DB_PATH)
    try:
        fact_id = insert_fact(
            conn,
            entity=args.entity,
            attribute=args.attribute,
            value=args.value,
            semantic_content=args.semantic_content,
            tags=args.tags,
            source="manual",
            session_id=args.session_id,
        )
    finally:
        conn.close()
    print(fact_id)


def cmd_delete_fact(args):
    conn = get_conn()
    try:
        ok = delete_fact(conn, args.fact_id)
    finally:
        conn.close()
    if not ok:
        print(f"Fact not found: {args.fact_id}")
        sys.exit(1)
    print("deleted")


def cmd_dashboard(_args):
    webbrowser.open(DASHBOARD_URL)
    print(DASHBOARD_URL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic memory CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap")
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("semantic")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_semantic)

    p = sub.add_parser("get-session")
    p.add_argument("session_id")
    p.set_defaults(func=cmd_get_session)

    p = sub.add_parser("tail")
    p.add_argument("n", nargs="?", type=int, default=10)
    p.set_defaults(func=cmd_tail)

    p = sub.add_parser("add-fact")
    p.add_argument("entity")
    p.add_argument("attribute")
    p.add_argument("value")
    p.add_argument("--semantic-content")
    p.add_argument("--tag", dest="tags", action="append", default=[])
    p.add_argument("--session-id")
    p.set_defaults(func=cmd_add_fact)

    p = sub.add_parser("delete-fact")
    p.add_argument("fact_id")
    p.set_defaults(func=cmd_delete_fact)

    p = sub.add_parser("dashboard")
    p.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    enable_debug("cli")
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
