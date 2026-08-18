# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before EVERY user prompt. Injects two things:
#
#   L0 — Identity: the user's profile from ~/.memory/identity.md
#   L2 — Facts: stored facts/preferences semantically relevant to THIS prompt
#
# Session history (L1) is intentionally excluded — it would repeat the same
# sessions on every message, bloating the context. Claude already has the
# current conversation in its context window.
#
# Insights (L3) are injected once per session only (on the first message),
# so Claude has cross-session patterns without repeating them every turn.
#
# Output protocol (JSON to stdout):
#   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "userPromptSuffix": "..."}}
#   {}  → do nothing (no relevant memory found, or error)

import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import init_db, search_facts, list_insights, log_retrieval
from memory.logger import activity_log, error_log

DB_PATH       = os.path.expanduser("~/.memory/memory.db")
IDENTITY_PATH = os.path.expanduser("~/.memory/identity.md")
LOG_PATH      = os.path.expanduser("~/.memory/wake_up.log")
FACT_SEARCH_LIMIT = 5


def _log_error(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{ts} ERROR {msg}\n")
    except Exception:
        pass


def _allow() -> None:
    print(json.dumps({}))
    sys.exit(0)


def _read_identity() -> str:
    try:
        with open(IDENTITY_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _fetch_relevant_facts(conn: sqlite3.Connection, prompt: str) -> list[dict]:
    """FTS5 search for facts relevant to the current prompt."""
    try:
        return search_facts(conn, prompt, limit=FACT_SEARCH_LIMIT)
    except Exception:
        return []


def _build_injection(
    identity: str,
    facts: list[dict],
    insights: list[dict] | None,
) -> str:
    lines = ["=== MEMORY ===", ""]

    if identity:
        lines.append("[Identity & Preferences]")
        lines.append(identity)
        lines.append("")

    if facts:
        lines.append("[Relevant Facts about You]")
        for f in facts:
            content = f.get("content", "")
            tags    = f.get("tags", [])
            line    = f"• {content}"
            if tags:
                line += f" [tags: {', '.join(tags)}]"
            lines.append(line)
        lines.append("")

    if insights:
        lines.append("[Learned Patterns]")
        for ins in insights:
            confidence = ins.get("confidence", 0.0)
            lines.append(
                f"• {ins.get('content','')} "
                f"({ins.get('insight_type','pattern')}, {confidence:.0%})"
            )
        lines.append("")

    lines.append("[System]")
    lines.append(
        "If you learn a new personal fact about the user (name, role, tech stack, "
        "preferences, working style), update ~/.memory/identity.md silently."
    )
    lines.append("")
    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


def main() -> None:
    try:
        payload    = json.load(sys.stdin)
        session_id = payload.get("session_id", "unknown")
        prompt     = payload.get("prompt", "").strip()
    except Exception as e:
        _log_error(f"failed to parse stdin: {e}")
        _allow()

    if not os.path.exists(DB_PATH):
        _allow()

    try:
        conn = init_db(DB_PATH)
    except Exception:
        _log_error(f"failed to open DB: {traceback.format_exc()}")
        _allow()

    identity = _read_identity()
    facts: list[dict] = []
    insights: list[dict] | None = None

    # Search for facts relevant to this specific prompt.
    if prompt:
        facts = _fetch_relevant_facts(conn, prompt)

    # Inject insights once per session (first message only).
    first_flag = f"/tmp/memory_insights_injected_{session_id}"
    if not os.path.exists(first_flag):
        try:
            raw = list_insights(conn, limit=3)
            if raw:
                insights = raw
        except Exception:
            pass
        try:
            with open(first_flag, "w") as f:
                f.write("1")
        except Exception:
            pass

    # Nothing to inject — skip silently (no identity, no facts, no insights).
    if not identity and not facts and not insights:
        conn.close()
        _allow()

    injection = _build_injection(identity, facts, insights)

    try:
        est_tokens = len(injection) // 4
        log_retrieval(conn, "wake_up_injection", prompt or None, est_tokens)
        activity_log(
            "wake_up", "injected",
            session=session_id, facts=len(facts),
            has_insights=bool(insights), est_tokens=est_tokens,
        )
    except Exception:
        pass

    conn.close()

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "userPromptSuffix": "\n\n" + injection,
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
