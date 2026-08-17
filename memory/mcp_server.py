# mcp_server.py — MCP server that exposes the memory database to Claude.
#
# MCP (Model Context Protocol) is how Claude Code talks to external tools.
# This server wraps the SQLite database so Claude can call three tools:
#
#   memory_status       — how many sessions are stored and when
#   memory_search       — full-text search across all saved conversations
#   memory_get_session  — retrieve one full conversation by its session ID
#
# Run standalone (for debugging):
#   python3 memory/mcp_server.py
#
# Registered in .claude/settings.json so Claude starts it automatically.

import json
import os
import sys

# Add the project root to Python's module search path so we can import memory.db.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    init_db,
    search as db_search,
    semantic_search as db_semantic_search,
    semantic_search_chunks as db_semantic_search_chunks,
    log_retrieval,
)

# FastMCP is the simplest way to build an MCP server in Python.
# You decorate plain functions with @mcp.tool() and FastMCP handles
# the protocol, serialisation, and stdin/stdout transport automatically.
from mcp.server.fastmcp import FastMCP

# Where the memory database lives. Must match the path used by the hook.
DB_PATH = os.path.expanduser("~/.memory/memory.db")


def get_conn():
    """
    Open a connection to the memory database.

    Creates the DB file (and the ~/.memory directory) if they don't exist yet,
    so the server starts cleanly even before any sessions have been saved.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return init_db(DB_PATH)


# Create the MCP server. The name "memory" is what Claude sees in its tool list.
mcp = FastMCP("memory")


# ---------------------------------------------------------------------------
# Tool 1: memory_status
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_status() -> dict:
    """
    Return a summary of everything stored in the memory database.

    Claude can call this to quickly understand how much memory is available
    before deciding whether to search or retrieve a session.

    Returns a dict with:
        total_sessions  — number of conversations stored
        total_turns     — total number of individual messages across all sessions
        oldest_session  — timestamp of the earliest saved session
        newest_session  — timestamp of the most recently saved session
    """
    conn = get_conn()

    # fetchone() returns a single row; we use [0] to get the first (only) column.
    total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    total_turns    = conn.execute("SELECT COALESCE(SUM(turn_count), 0) FROM sessions").fetchone()[0]

    # MIN/MAX on the updated_at column give us the date range.
    date_row = conn.execute(
        "SELECT MIN(updated_at), MAX(updated_at) FROM sessions"
    ).fetchone()

    result = {
        "total_sessions": total_sessions,
        "total_turns":    total_turns,
        "oldest_session": date_row[0],   # None if no sessions yet
        "newest_session": date_row[1],
    }

    # Log that this tool was called so the dashboard can count retrievals.
    log_retrieval(conn, "memory_status", None, len(str(result)))
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Tool 2: memory_search
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_search(query: str, limit: int = 10) -> list:
    """
    Full-text search across all saved conversation transcripts.

    Uses SQLite FTS5 (full-text search) to find sessions where the query
    terms appear. Results are ranked by relevance — best match first.

    Args:
        query  — the words or phrase to search for (e.g. "quantum entanglement")
        limit  — maximum number of results to return (default 10)

    Returns a list of dicts, each with:
        session_id  — ID of the matching session
        agent       — which agent saved this session (e.g. "claude")
        updated_at  — when the session was last saved
        snippet     — a short excerpt showing where the match was found,
                      with matching words wrapped in [square brackets]
    """
    conn = get_conn()
    results = db_search(conn, query, limit=limit)
    log_retrieval(conn, "memory_search", query, len(str(results)))
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Tool 3: memory_get_session
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_get_session(session_id: str) -> dict:
    """
    Retrieve the full verbatim transcript for one conversation session.

    Use memory_search first to find the session_id, then call this to
    read the complete conversation.

    Args:
        session_id — the unique ID of the session to retrieve

    Returns a dict with:
        session_id  — echoed back for confirmation
        agent       — which agent saved this session
        started_at  — when the conversation started
        updated_at  — when it was last saved
        turn_count  — number of turns in the transcript
        transcript  — the full list of {"role": ..., "content": ...} turns

    Returns {"error": "session not found"} if the session_id does not exist.
    """
    conn = get_conn()

    row = conn.execute(
        "SELECT session_id, agent, started_at, updated_at, turn_count, transcript "
        "FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()

    if row is None:
        conn.close()
        return {"error": f"session not found: {session_id}"}

    result = {
        "session_id": row["session_id"],
        "agent":      row["agent"],
        "started_at": row["started_at"],
        "updated_at": row["updated_at"],
        "turn_count": row["turn_count"],
        # json.loads converts the stored JSON string back into a Python list.
        "transcript": json.loads(row["transcript"]),
    }
    log_retrieval(conn, "memory_get_session", session_id, len(str(result)))
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Tool 4: memory_semantic_search
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_semantic_search(query: str, limit: int = 5) -> dict:
    """
    Semantic (vector) search across all saved conversation transcripts.

    Unlike keyword search, this finds conversations with similar *meaning*
    even when the exact words don't appear. For example, a query about
    "broken login" can match a session that discusses "authentication bug".

    Requires sentence-transformers and sqlite-vec to be installed.
    Returns an empty results list (not an error) if they are unavailable.

    Args:
        query  — natural-language question or topic to search for
        limit  — maximum results to return (default 5)

    Returns a dict with:
        query    — the original query (echoed back)
        count    — number of results returned
        results  — list of dicts with session_id, agent, updated_at, distance
                   (distance is cosine distance: lower = more similar)
    """
    conn = get_conn()

    # Use chunk-level search so we find the closest window within each session,
    # not a blended whole-session vector. Falls back to [] if the model is missing.
    try:
        # Fetch more chunk results than limit so we can collapse duplicates
        # (multiple chunks from the same session) down to one per session.
        chunk_results = db_semantic_search_chunks(conn, query, limit=limit * 10)
    except (ImportError, Exception):
        # If sentence-transformers isn't installed or anything else fails,
        # return an empty results list — same graceful behaviour as before.
        chunk_results = []

    # Group by session_id, keeping only the closest chunk for each session.
    # Because chunk_results is already sorted by distance ascending, the first
    # chunk we encounter for each session_id is the best match for that session.
    seen_sessions: dict[str, dict] = {}
    for chunk in chunk_results:
        sid = chunk["session_id"]
        if sid not in seen_sessions:
            # Preserve the same keys callers expect (session_id, distance, snippet)
            # plus chunk_index so callers know which part of the session matched.
            seen_sessions[sid] = {
                "session_id":  sid,
                "chunk_index": chunk["chunk_index"],
                "distance":    chunk["distance"],
                "snippet":     chunk["snippet"],
            }

    # Collect the best-per-session results, sorted by distance, up to limit.
    results = sorted(seen_sessions.values(), key=lambda r: r["distance"])[:limit]

    result = {
        "query":   query,
        "count":   len(results),
        "results": results,
    }
    log_retrieval(conn, "memory_semantic_search", query, len(str(result)))
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # mcp.run() starts the server and listens on stdin/stdout using the MCP
    # stdio transport. Claude Code launches this process and communicates
    # with it over those streams — no network port needed.
    mcp.run(transport="stdio")
