# wake_up.py — Claude Code UserPromptSubmit hook for memory wake-up injection.
#
# Claude Code runs this script before processing the user's first message.
# It injects a digest of who the user is (L0), relevant past sessions (L1),
# and relevant stored facts (L2) so Claude has memory context without the
# user having to repeat themselves.
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

# Add the project root to Python's module search path so we can import
# from memory.db even when this script is run directly (not as a package).
# __file__ is the path to this script; dirname×2 climbs up two levels
# (hooks/ → project root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the shared database helpers from the memory package.
# semantic_search_chunks — finds chunks topically similar to a query string.
# search_facts — FTS5 full-text search over stored facts.
# init_db — opens (or creates) the SQLite DB with all schema migrations applied.
from memory.db import init_db, semantic_search_chunks, search_facts, list_insights, log_retrieval
from memory.logger import activity_log, error_log


# Paths — expanduser() converts "~" to the actual home directory at runtime
DB_PATH = os.path.expanduser("~/.memory/memory.db")
IDENTITY_PATH = os.path.expanduser("~/.memory/identity.md")
LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")

# How many recent sessions to include in the L1 digest (recency fallback)
RECENT_SESSION_COUNT = 5

# How many chunk results to request from semantic search
CHUNK_SEARCH_LIMIT = 5

# How many facts to retrieve from FTS5 search
FACT_SEARCH_LIMIT = 5


def log_error(message: str) -> None:
    """Append an error message to the wake-up log file."""
    # datetime.now gives the current time; isoformat() turns it into a readable string.
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{timestamp} ERROR {message}\n")
    except Exception:
        pass  # if we can't even write the log, silently give up


def allow() -> None:
    """Output an empty JSON object and exit. This tells Claude Code to proceed normally."""
    # For UserPromptSubmit hooks, {} means "do nothing, continue".
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

    This is the recency fallback — used when no prompt is available or when
    semantic search fails or returns no results.

    Returns a list of dicts with keys: session_id, updated_at, turn_count,
    first_user_message. Returns an empty list if the database doesn't exist yet.
    """
    # If the database file doesn't exist at all, return empty gracefully.
    if not os.path.exists(DB_PATH):
        return []

    # Open a direct SQLite connection — row_factory lets us access columns by name.
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
        # Parse the transcript JSON to extract the first user message preview.
        try:
            turns = json.loads(row["transcript"])
            # Find the first turn where role is "user"
            first_user = next(
                (t["content"] for t in turns if t.get("role") == "user"), ""
            )
            # Truncate to 100 characters to keep the digest compact.
            # replace("\n", " ") prevents multi-line entries from breaking the layout.
            preview = first_user[:100].replace("\n", " ")
        except Exception:
            preview = ""

        results.append({
            "session_id":        row["session_id"],
            "updated_at":        row["updated_at"],
            "turn_count":        row["turn_count"],
            "first_user_message": preview,
        })

    return results


def fetch_relevant_context(
    conn: sqlite3.Connection,
    prompt: str,
) -> tuple[list[dict], list[dict]]:
    """
    Run semantic chunk search and fact FTS5 search against the given prompt.

    Returns a pair (sessions, facts):
      - sessions: list of dicts with keys: session_id, updated_at, turn_count,
                  snippet, similarity.  Sorted by similarity descending (most
                  relevant first).  De-duplicated per session_id — only the
                  closest chunk per session is kept.
      - facts:    list of dicts from search_facts() — content, tags, etc.
                  Returns [] if FTS5 search fails or no facts match.

    Raises ImportError if sentence-transformers is not installed (propagates
    from semantic_search_chunks so callers can fall back to recency).
    """
    # Run semantic search over stored chunks.
    # This raises ImportError if sentence-transformers is not installed —
    # the caller (get_wake_up_digest / main) catches that and falls back.
    chunk_results = semantic_search_chunks(conn, prompt, limit=CHUNK_SEARCH_LIMIT)

    # Group chunks by session_id, keeping only the closest chunk per session.
    # "Closest" = lowest cosine distance (0 = identical, 2 = opposite).
    best_chunks: dict[str, dict] = {}
    for chunk in chunk_results:
        sid = chunk["session_id"]
        if sid not in best_chunks or chunk["distance"] < best_chunks[sid]["distance"]:
            best_chunks[sid] = chunk

    # Fetch session metadata (updated_at, turn_count) for each matched session.
    sessions = []
    for sid, chunk in best_chunks.items():
        row = conn.execute(
            "SELECT updated_at, turn_count FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
        if row is None:
            # Session was deleted after chunks were stored — skip it.
            continue

        # Convert cosine distance to a human-readable similarity percentage.
        # Cosine distance ranges 0–2; distance 0 → 100%, distance 2 → 0%.
        # Formula: similarity% = (1 - distance/2) * 100
        similarity = round((1 - chunk["distance"] / 2) * 100, 1)

        sessions.append({
            "session_id": sid,
            "updated_at": row["updated_at"],
            "turn_count": row["turn_count"],
            "snippet":    chunk["snippet"],
            "similarity": similarity,
        })

    # Sort sessions so the most relevant (highest similarity) appears first.
    sessions.sort(key=lambda s: s["similarity"], reverse=True)

    # Log the semantic search result so the activity log shows what was found.
    activity_log(
        "wake_up", "semantic_search",
        query=prompt, results=len(sessions),
        top=sessions[0]["session_id"] if sessions else "none",
        similarity=sessions[0]["similarity"] if sessions else 0,
    )

    # Search stored facts using FTS5 keyword matching.
    # FTS5 raises sqlite3.OperationalError for queries containing special
    # characters like + - * : ( ), so we wrap in try/except and fall back to [].
    facts: list[dict] = []
    try:
        facts = search_facts(conn, prompt, limit=FACT_SEARCH_LIMIT)
        activity_log("wake_up", "fact_search", query=prompt, results=len(facts))
    except sqlite3.OperationalError as e:
        log_error(f"facts FTS5 search failed (query may contain special chars): {e}")
        error_log("wake_up", f"FTS5 fact search failed for query: {prompt[:60]}", exc=e)

    return sessions, facts


def build_digest(
    identity: str,
    sessions: list[dict],
    facts: list[dict] | None = None,
    insights: list[dict] | None = None,
) -> str:
    """
    Format the wake-up digest from L0 (identity), L1 (sessions), L2 (facts),
    and optionally L3 (learned insights from the background daemon).

    Sessions can be in two formats:
      - Recency format: dict has 'first_user_message' key (from fetch_recent_sessions).
        Produces "[L1 — Recent Sessions]" heading.
      - Relevance format: dict has 'snippet' and 'similarity' keys (from fetch_relevant_context).
        Produces "[L1 — Relevant Sessions]" heading.

    The 'facts' parameter is only present on the relevance path:
      - None (or not passed) → no [L2] section, recency heading
      - [] → no [L2] section, but relevance heading (search ran, found nothing)
      - non-empty list → [L2 — Relevant Facts] section with bullet points

    The 'insights' parameter controls the optional [L3 — Learned Insights] section:
      - None (default) → no [L3] section (daemon has not run yet or no insights stored)
      - [] → no [L3] section (daemon ran but found nothing noteworthy)
      - non-empty list → [L3 — Learned Insights] with one bullet per insight

    The insights=None default keeps backward compatibility with existing callers
    that don't pass insights.
    """
    # Start with the outer markers so Claude can easily spot the injected block.
    lines = ["=== MEMORY WAKE-UP ===", ""]

    # --- L0: Identity section ---
    lines.append("[L0 — Identity]")
    if identity:
        lines.append(identity)
    else:
        # Friendly hint so the user knows how to set up their profile.
        lines.append("(no identity.md found — create ~/.memory/identity.md to set up your profile)")
    lines.append("")

    # --- L1: Sessions section ---
    # The heading changes depending on which retrieval path produced the sessions.
    # When facts is None, we used the recency fallback.
    # When facts is a list (even empty), we used the relevance path.
    if facts is not None:
        lines.append("[L1 — Relevant Sessions]")
    else:
        lines.append("[L1 — Recent Sessions]")

    if sessions:
        for i, session in enumerate(sessions, start=1):
            # .get() with fallback handles sessions that might be missing a field.
            updated = session.get("updated_at") or "unknown date"
            turns = session.get("turn_count") or 0

            if "similarity" in session:
                # Relevance-based format: includes similarity % and chunk snippet.
                similarity = session["similarity"]
                snippet = session.get("snippet", "")
                lines.append(
                    f"{i}. {updated} ({turns} turns, relevance {similarity}%) — {snippet}"
                )
            else:
                # Recency-based format: includes first user message preview.
                preview = session.get("first_user_message", "")
                lines.append(f"{i}. {updated} ({turns} turns) — {preview}")
    else:
        lines.append("(no past sessions found)")
    lines.append("")

    # --- L2: Facts section (only on relevance path with results) ---
    # facts=None → recency path, no L2.
    # facts=[] → relevance path, but no matching facts found, no L2.
    # facts=[...] → show the matching facts.
    if facts:
        lines.append("[L2 — Relevant Facts]")
        for fact in facts:
            content = fact.get("content", "")
            # tags is already a Python list (search_facts decodes the JSON array).
            tags = fact.get("tags", [])
            if tags:
                # Join multiple tags with a comma for compact display.
                tags_str = ", ".join(tags)
                lines.append(f"• {content} [tags: {tags_str}]")
            else:
                lines.append(f"• {content}")
        lines.append("")

    # --- L3: Learned Insights section (Phase 13) ---
    # Only shown when the daemon has generated insights (insights is a non-empty list).
    # insights=None → omit section entirely (daemon has never run or no insights).
    # insights=[]   → omit section (daemon ran but found nothing).
    # insights=[..] → emit the [L3] block with one bullet per insight.
    if insights:
        lines.append("[L3 — Learned Insights]")
        for insight in insights:
            content      = insight.get("content", "")
            itype        = insight.get("insight_type", "pattern")
            confidence   = insight.get("confidence", 0.0)
            # Format: "• content (type, confidence XX%)"
            lines.append(f"• {content} ({itype}, confidence {confidence:.0%})")
        lines.append("")
        # Log so the activity log reflects how many insights were injected.
        activity_log("wake_up", "insights_injected", count=len(insights))

    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


def get_wake_up_digest(
    conn: sqlite3.Connection | None,
    session_id: str,
    prompt: str,
    identity: str,
) -> str:
    """
    Build the full wake-up digest, trying relevance-based retrieval first.

    This is the main testable unit — main() wraps it with I/O and flag files.

    Falls back to recency (last N sessions by updated_at, no facts) when:
      - prompt is empty or absent
      - semantic_search_chunks raises (e.g. sentence-transformers not installed)
      - any other exception occurs during relevance retrieval
      - no relevant sessions were found (empty result)

    Args:
        conn       — open DB connection (from init_db), or None if DB unavailable
        session_id — the current session's ID (excluded from recency results)
        prompt     — the user's prompt text (empty string triggers recency fallback)
        identity   — content of ~/.memory/identity.md (may be empty string)

    Returns:
        A formatted string ready to inject as userPromptSuffix.
    """
    # Start with no sessions and no facts — filled in below.
    sessions: list[dict] = []
    facts: list[dict] | None = None

    # Attempt relevance-based retrieval only when we have a prompt and a DB connection.
    if prompt and conn is not None:
        try:
            sessions, facts = fetch_relevant_context(conn, prompt)
        except Exception as e:
            # Log the failure but do not re-raise — we fall back gracefully below.
            log_error(
                f"relevance retrieval failed, using recency fallback: "
                f"{traceback.format_exc()}"
            )
            error_log("wake_up", "relevance retrieval failed; falling back to recency", exc=e)
            activity_log("wake_up", "fallback", reason="retrieval_error")
            sessions = []
            facts = None

    # If relevance retrieval produced no sessions (either because it failed,
    # or it returned an empty list, or prompt was empty), use recency fallback.
    if not sessions:
        # fetch_recent_sessions uses the module-level DB_PATH variable so that
        # tests can monkey-patch it; it opens its own SQLite connection internally.
        sessions = fetch_recent_sessions(session_id)
        # Reset facts to None — the recency path does not include a facts section.
        facts = None
        # Log why the fallback was triggered.
        reason = "no_prompt" if not prompt else "no_results"
        activity_log("wake_up", "fallback", reason=reason, recency_sessions=len(sessions))

    mode = "recency" if facts is None else "relevance"

    # --- L3: Load top 3 insights from the daemon (Phase 13) ---
    # We attempt to read insights only when we have a DB connection.
    # If the insights table is empty or the DB is unavailable, insights stays None
    # so build_digest omits the [L3] section gracefully.
    insights: list[dict] | None = None
    if conn is not None:
        try:
            raw_insights = list_insights(conn, limit=3)
            if raw_insights:
                # Only populate the list when there are actual insights to show.
                insights = raw_insights
        except Exception:
            # Never let insight loading crash the wake-up hook.
            insights = None

    activity_log(
        "wake_up", "injected",
        session=session_id, mode=mode,
        sessions=len(sessions), facts=len(facts) if facts else 0,
    )
    return build_digest(identity, sessions, facts, insights=insights)


def main() -> None:
    """
    Entry point: read the hook payload from stdin, build the digest, write to stdout.

    Claude Code calls this script once per session (subsequent messages are
    skipped via a flag file at /tmp/memory_injected_{session_id}).
    """
    # Read the hook payload from stdin.
    # Claude Code sends a JSON object with at least: session_id, hook_event_name.
    # The 'prompt' field contains the user's current message text.
    try:
        payload = json.load(sys.stdin)
        session_id = payload.get("session_id", "unknown")
        # prompt may be absent on the very first message or in edge cases.
        prompt = payload.get("prompt", "")
    except Exception as e:
        log_error(f"failed to parse stdin: {e}")
        allow()

    # Check if we already injected memory for this session.
    # We use a temp file as a simple flag — it only exists during the session.
    # Subsequent messages in the same session should NOT re-inject memory.
    flag_path = f"/tmp/memory_injected_{session_id}"
    if os.path.exists(flag_path):
        # Already ran for this session — skip to avoid injecting on every message.
        allow()

    try:
        identity = read_identity()

        # Open the database if it exists.
        # We pass conn=None to get_wake_up_digest if DB is unavailable; it falls
        # back to the file-based fetch_recent_sessions path automatically.
        conn = None
        if os.path.exists(DB_PATH):
            try:
                conn = init_db(DB_PATH)
            except Exception:
                log_error(f"failed to open DB: {traceback.format_exc()}")
                conn = None

        # Build the digest using relevance search (or recency fallback).
        digest = get_wake_up_digest(conn, session_id, prompt, identity)

        # Log the injection size to the retrievals table so token economics
        # can be tracked. We estimate tokens by dividing character count by 4
        # (a standard rough approximation). This reuses the existing retrievals
        # table — no schema change needed.
        try:
            injection_tokens = len(digest) // 4
            if conn is not None:
                log_retrieval(conn, "wake_up_injection", None, injection_tokens)
            activity_log("wake_up", "injection_size", chars=len(digest), est_tokens=injection_tokens)
        except Exception:
            # Never let logging block the injection — silently ignore failures.
            pass

        if conn is not None:
            conn.close()

        # Mark this session as injected so subsequent messages don't re-inject.
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
