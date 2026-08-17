# wake_up.py — Claude Code UserPromptSubmit hook for memory wake-up injection.
#
# Claude Code runs this script before processing the user's first message.
# It injects a digest of who the user is (L0) and recent past sessions (L1)
# so Claude has memory context without the user having to repeat themselves.
#
# Output protocol (JSON to stdout):
#   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "userPromptSuffix": "<digest>"}}
#   {}  → do nothing, Claude proceeds normally
#
# NOTE: do NOT use {"decision": "block"/"allow"} for UserPromptSubmit hooks.
# The `decision` field is only valid for Stop hooks. Using it here causes a
# schema validation error on every message.
#
# This script must NEVER crash visibly. All errors go to the log file.

import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timezone


# Paths — expanduser() converts "~" to the actual home directory at runtime
DB_PATH = os.path.expanduser("~/.memory/memory.db")
IDENTITY_PATH = os.path.expanduser("~/.memory/identity.md")
LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")

# How many recent sessions to include in the L1 digest
RECENT_SESSION_COUNT = 5


def log_error(message: str) -> None:
    """Append an error message to the wake-up log file."""
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{timestamp} ERROR {message}\n")
    except Exception:
        pass  # if we can't even write the log, silently give up


def allow() -> None:
    """Output an empty JSON object and exit. This tells Claude Code to proceed normally."""
    # For UserPromptSubmit hooks, {} means "do nothing".
    # {"decision": "allow"} is only valid for Stop hooks — don't use it here.
    print(json.dumps({}))
    sys.exit(0)


def read_identity() -> str:
    """Read the user's identity file. Returns empty string if the file doesn't exist."""
    try:
        with open(IDENTITY_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def fetch_recent_sessions(session_id: str) -> list[dict]:
    """
    Query the database for the 5 most recent sessions, excluding the current one.
    Returns a list of dicts with keys: session_id, updated_at, turn_count, first_user_message.
    Returns an empty list if the database doesn't exist yet.
    """
    if not os.path.exists(DB_PATH):
        return []

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT session_id, updated_at, turn_count, transcript
        FROM sessions
        WHERE session_id != ?      -- exclude the current session
        ORDER BY updated_at DESC   -- most recent first
        LIMIT ?
        """,
        (session_id, RECENT_SESSION_COUNT),
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        # Parse the transcript JSON to extract the first user message
        try:
            turns = json.loads(row["transcript"])
            # Find the first turn where role is "user"
            first_user = next(
                (t["content"] for t in turns if t.get("role") == "user"), ""
            )
            # Truncate to 100 characters to keep the digest compact
            preview = first_user[:100].replace("\n", " ")
        except Exception:
            preview = ""

        results.append({
            "session_id": row["session_id"],
            "updated_at": row["updated_at"],
            "turn_count": row["turn_count"],
            "first_user_message": preview,
        })

    return results


def build_digest(identity: str, recent_sessions: list[dict]) -> str:
    """
    Format the L0 (identity) and L1 (recent sessions) into a compact digest string.
    This is the text Claude will receive as context before the user's first message.
    """
    lines = ["=== MEMORY WAKE-UP ===", ""]

    # L0: who the user is
    lines.append("[L0 — Identity]")
    if identity:
        lines.append(identity)
    else:
        lines.append("(no identity.md found — create ~/.memory/identity.md to set up your profile)")
    lines.append("")

    # L1: recent past sessions
    lines.append("[L1 — Recent Sessions]")
    if recent_sessions:
        for i, session in enumerate(recent_sessions, start=1):
            updated = session["updated_at"] or "unknown date"
            turns = session["turn_count"] or 0
            preview = session["first_user_message"]
            lines.append(f"{i}. {updated} ({turns} turns) — {preview}")
    else:
        lines.append("(no past sessions found)")
    lines.append("")

    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


def main() -> None:
    # Read the hook payload from stdin.
    # Claude Code sends a JSON object with at least: session_id, hook_event_name.
    try:
        payload = json.load(sys.stdin)
        session_id = payload.get("session_id", "unknown")
    except Exception as e:
        log_error(f"failed to parse stdin: {e}")
        allow()

    # Check if we already injected memory for this session.
    # We use a temp file as a simple flag — it only exists during the session.
    flag_path = f"/tmp/memory_injected_{session_id}"
    if os.path.exists(flag_path):
        # Already ran for this session — skip to avoid injecting on every message
        allow()

    try:
        identity = read_identity()
        recent_sessions = fetch_recent_sessions(session_id)
        digest = build_digest(identity, recent_sessions)

        # Mark this session as injected so subsequent messages don't re-inject
        with open(flag_path, "w") as f:
            f.write("1")

        # Append the digest to the user's prompt via userPromptSuffix.
        # hookEventName is required by Claude Code's schema validation.
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "userPromptSuffix": "\n\n" + digest,
            }
        }))
        sys.exit(0)

    except Exception as e:
        log_error(f"unexpected error: {traceback.format_exc()}")
        allow()


if __name__ == "__main__":
    main()
