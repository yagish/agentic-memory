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

from memory.logger import activity_log, error_log
from memory.debug import enable_debug

from memory.db import (
    init_db,
    search as db_search,
    semantic_search as db_semantic_search,
    semantic_search_chunks as db_semantic_search_chunks,
    hybrid_search as db_hybrid_search,
    log_retrieval,
    insert_fact,
    update_fact,
    delete_fact,
    search_facts as db_search_facts,
    list_facts as db_list_facts,
    # Phase 13: daemon-generated insights and topic clusters
    list_insights as db_list_insights,
    get_clusters as db_get_clusters,
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
    activity_log("mcp", "memory_status", sessions=total_sessions, turns=total_turns)
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
    activity_log("mcp", "memory_search", query=query, results=len(results), result_bytes=len(str(results)))
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
    activity_log("mcp", "memory_get_session", session=session_id, turns=result.get("turn_count"))
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
    activity_log("mcp", "memory_semantic_search", query=query, results=len(results), result_bytes=len(str(result)))
    conn.close()
    return result


# ---------------------------------------------------------------------------
# Tools 5–8: fact CRUD tools (Phase 8)
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_save_fact(content: str, tags: list[str] | None = None) -> dict:
    """
    Save a structured fact to permanent memory.

    Use this to proactively record something worth remembering across sessions —
    a user preference, a key decision, a domain fact, or any insight that
    shouldn't be lost when the conversation ends.

    Args:
        content — the fact text (e.g. "User prefers dark mode in all editors")
        tags    — optional list of tag strings for filtering later (e.g. ["preferences", "ui"])

    Returns a dict with:
        id      — the UUID assigned to this fact (use it to update or delete later)
        content — the text that was saved
        tags    — the tag list (empty list if none were provided)
    """
    conn = get_conn()

    # insert_fact writes to the database and returns the new fact's UUID.
    # source="agent" marks that this came through an MCP tool call, not manual entry.
    fact_id = insert_fact(conn, content, tags=tags, source="agent")

    # Log the first 100 characters of the content as the "query" for the retrievals table.
    # This gives the dashboard a meaningful preview of what was saved.
    log_retrieval(conn, "memory_save_fact", content[:100], len(fact_id))
    activity_log("mcp", "memory_save_fact", content=content, tags=str(tags or []), id=fact_id)
    conn.close()
    return {"id": fact_id, "content": content, "tags": tags or []}


@mcp.tool()
def memory_update_fact(
    fact_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """
    Update the content or tags of a previously saved fact.

    Pass only the fields you want to change — omitted fields are left unchanged.
    updated_at is always refreshed automatically.

    Args:
        fact_id — the UUID returned by memory_save_fact
        content — new text for the fact, or omit to leave the text unchanged
        tags    — new tag list, or omit to leave tags unchanged

    Returns a dict with:
        updated — True if the fact was found and changed, False if fact_id not found
        fact_id — the UUID you passed in (echoed back for confirmation)
    """
    conn = get_conn()

    # update_fact returns True if the row was found and modified, False otherwise.
    updated = update_fact(conn, fact_id, content=content, tags=tags)

    log_retrieval(conn, "memory_update_fact", fact_id, len(str(updated)))
    activity_log("mcp", "memory_update_fact", id=fact_id, updated=updated)
    conn.close()
    return {"updated": updated, "fact_id": fact_id}


@mcp.tool()
def memory_delete_fact(fact_id: str) -> dict:
    """
    Delete a fact from permanent memory.

    The FTS5 search index is updated automatically — deleted facts will no
    longer appear in memory_search results.

    Args:
        fact_id — the UUID returned by memory_save_fact

    Returns a dict with:
        deleted — True if the fact was found and removed, False if fact_id not found
        fact_id — the UUID you passed in (echoed back for confirmation)
    """
    conn = get_conn()

    # delete_fact removes the row and returns True if something was actually deleted.
    deleted = delete_fact(conn, fact_id)

    log_retrieval(conn, "memory_delete_fact", fact_id, len(str(deleted)))
    activity_log("mcp", "memory_delete_fact", id=fact_id, deleted=deleted)
    conn.close()
    return {"deleted": deleted, "fact_id": fact_id}


@mcp.tool()
def memory_list_facts(tag: str | None = None, limit: int = 20) -> list:
    """
    List stored facts, optionally filtered to a specific tag.

    Use this to browse what Claude has saved, or to find facts in a category
    before deciding whether to update or delete them.

    Args:
        tag   — if given, only return facts that carry this tag string
        limit — maximum number of facts to return (default 20)

    Returns a list of dicts, each with:
        id, content, tags, source, session_id, created_at, updated_at
    """
    conn = get_conn()

    # db_list_facts returns a list of dicts with tags already parsed back to lists.
    results = db_list_facts(conn, tag=tag, limit=limit)

    # Log "(all)" as the query when no tag filter was applied — gives the dashboard
    # a readable label instead of a blank entry.
    log_retrieval(conn, "memory_list_facts", tag or "(all)", len(str(results)))
    activity_log("mcp", "memory_list_facts", tag=tag or "(all)", results=len(results))
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Tool: memory_hybrid_search
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_hybrid_search(query: str, limit: int = 10) -> dict:
    """
    Hybrid search: combines full-text (FTS5) keyword search and semantic
    (embedding-based) search using Reciprocal Rank Fusion (RRF).

    USE THIS AS THE DEFAULT RETRIEVAL TOOL. It almost always outperforms
    keyword-only (memory_search) or semantic-only (memory_semantic_search)
    searches because sessions that rank highly in BOTH result lists are
    promoted. Sessions that match only one approach still appear, but rank
    lower. If sentence-transformers is not installed, it falls back gracefully
    to FTS5-only results.

    Args:
        query — natural-language question or keyword phrase
        limit — maximum number of sessions to return (default 10)

    Returns a dict with:
        query   — the query you passed in (echoed back for reference)
        count   — number of results returned
        results — list of dicts, each with session_id, agent, updated_at,
                  snippet, and rrf_score (the fused relevance score)
    """
    # Open a connection to the memory database for this request.
    conn = get_conn()

    # Call the hybrid_search function from db.py, which runs FTS5 + semantic
    # search and fuses the ranked lists with RRF.
    results = db_hybrid_search(conn, query, limit=limit)

    # Build the response dict that gets returned to Claude.
    result = {
        "query":   query,
        "count":   len(results),
        "results": results,
    }

    # Log this retrieval event to the retrievals table for dashboard reporting.
    log_retrieval(conn, "memory_hybrid_search", query, len(str(result)))

    # Log to the activity log so hooks and monitoring can track usage.
    activity_log(
        "mcp",
        "memory_hybrid_search",
        query=query,
        results=len(results),
        result_bytes=len(str(result)),
    )

    conn.close()
    return result


# ---------------------------------------------------------------------------
# Tool: memory_list_insights (Phase 13)
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_list_insights(insight_type: str | None = None, limit: int = 20) -> list:
    """
    List cross-session insights discovered by the background daemon.

    Insights are patterns, preferences, and skills the system learned
    from analysing multiple conversations over time.  The daemon must
    have run at least once (and processed at least INSIGHT_EVERY_N=10
    sessions) before any insights appear.

    Args:
        insight_type — optional filter: "pattern", "preference", "skill",
                       or "topic_cluster"; pass None (or omit) to return all types
        limit        — maximum number of insights to return (default 20)

    Returns a list of dicts, each with:
        id, insight_type, content, evidence, confidence, created_at
    """
    conn = get_conn()

    # Fetch insights from the database, optionally filtered by type.
    results = db_list_insights(conn, insight_type=insight_type, limit=limit)

    # Log the retrieval so the dashboard can count how often this tool is called.
    log_retrieval(conn, "memory_list_insights", insight_type or "(all)", len(str(results)))
    activity_log(
        "mcp", "memory_list_insights",
        insight_type=insight_type or "(all)",
        results=len(results),
    )
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Tool: memory_list_clusters (Phase 13)
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_list_clusters() -> list:
    """
    List topic clusters discovered by the background daemon.

    The daemon groups sessions by transcript similarity into topic clusters.
    Each cluster has a label (derived from the first session's text),
    a member count, and a last-updated timestamp.

    Returns a list of dicts, each with:
        id, label, member_count, updated_at
    """
    conn = get_conn()

    # Fetch all clusters ordered by member_count descending (biggest first).
    results = db_get_clusters(conn)

    # Log the retrieval for dashboard reporting.
    log_retrieval(conn, "memory_list_clusters", "(all)", len(str(results)))
    activity_log("mcp", "memory_list_clusters", results=len(results))
    conn.close()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle("AgenticMemoryMCP")
    enable_debug("mcp")
    # mcp.run() starts the server and listens on stdin/stdout using the MCP
    # stdio transport. Claude Code launches this process and communicates
    # with it over those streams — no network port needed.
    mcp.run(transport="stdio")
