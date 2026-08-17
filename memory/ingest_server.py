# ingest_server.py — HTTP ingest endpoint for agent-agnostic session writes.
#
# Any agent (Cursor, LangChain, custom scripts) can call this local API.
# It supports:
#   POST /ingest  — store a full session transcript
#   POST /recall  — build a wake-up digest for a prompt
#   POST /answer  — direct-answer lookup for repeated prompts
#   GET  /status  — health summary
# Sessions are stored, chunked, and embedded exactly as if they came from
# the Claude Code Stop hook.
#
# Run:
#   python3 memory/ingest_server.py          # listens on localhost:7747
#   MEMORY_INGEST_PORT=8080 python3 ...      # custom port
#
# Manage via CLI:
#   python3 cli.py ingest-server start|stop|status

import json       # for JSON serialisation/deserialisation
import os         # for environment variables and file paths
import sqlite3    # used for graceful FTS error handling in /recall
import sys        # for modifying the Python module search path

# Add the project root (one level up from this file) to the module search
# path so we can import from the memory package even when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone   # for generating the updated_at timestamp

# FastAPI is a modern Python web framework for building APIs quickly.
# Pydantic is used for request validation — if a required field is missing
# or has the wrong type, FastAPI returns a 422 error automatically.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # allows browser-based agents to POST
from pydantic import BaseModel, field_validator
import uvicorn   # the ASGI server that actually listens for HTTP connections

# Import the database helpers from our memory package.
from memory.db import (
    init_db,           # opens (or creates) the SQLite database
    upsert_session,    # saves or updates a session row
    embed,             # converts text into a 384-dimensional vector
    store_embedding,   # persists the vector in the session_vecs table
    chunk_transcript,  # splits a transcript into overlapping text windows
    delete_chunks_for_session,  # removes old chunks before re-chunking
    store_chunk,       # persists one chunk + its embedding
    semantic_search_chunks,
    search_facts,
    list_insights,
    find_direct_answer,
    log_retrieval,
)
from memory.logger import activity_log, error_log  # structured logging helpers

# Where the SQLite database lives on disk — must match the hook and MCP server.
DB_PATH = os.path.expanduser("~/.memory/memory.db")
IDENTITY_PATH = os.path.expanduser("~/.memory/identity.md")

# Retrieval tuning for /recall.
RECENT_SESSION_COUNT = 5
CHUNK_SEARCH_LIMIT = 5
FACT_SEARCH_LIMIT = 5

# Port the server listens on — override with MEMORY_INGEST_PORT environment variable.
PORT = int(os.environ.get("MEMORY_INGEST_PORT", "7747"))

# Create the FastAPI application instance.
# The title and version appear in the auto-generated /docs page.
app = FastAPI(title="Memory Ingest Server", version="1.0.0")

# ── CORS — allow browser-based agents to POST from their domains ──────────────
# Browsers enforce the Same-Origin Policy: a page at another origin cannot
# fetch http://localhost:7747 unless the server explicitly says it's OK.
# CORSMiddleware adds the necessary Access-Control-* response headers.
#
# Override the allowed list with MEMORY_INGEST_CORS_ORIGINS (comma-separated):
#   export MEMORY_INGEST_CORS_ORIGINS="https://example.com,https://github.com"
_default_origins = "https://copilot.microsoft.com,https://github.com"
_origins_raw = os.environ.get("MEMORY_INGEST_CORS_ORIGINS", _default_origins)
_allowed_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,   # the sites that may call us
    allow_methods=["GET", "POST"],    # only the verbs we actually handle
    allow_headers=["Content-Type"],   # only the header the integrations send
)


# ---------------------------------------------------------------------------
# Request models — Pydantic validates incoming JSON automatically.
# If a field is missing or has the wrong type, FastAPI returns 422 Unprocessable
# Entity before our handler even runs, which keeps handler code clean.
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """One turn in a conversation — a single message from user or assistant."""
    role: str      # must be "user" or "assistant"
    content: str   # the text of the message

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        """Reject any role that is not 'user' or 'assistant'."""
        # Pydantic calls this validator when the model is constructed.
        # Raising ValueError causes a 422 response with a descriptive error.
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v


class IngestRequest(BaseModel):
    """
    The JSON body expected by POST /ingest.

    All fields except agent, started_at, and metadata are required.
    FastAPI returns 422 automatically if session_id or turns are absent.
    """
    session_id: str            # unique identifier for the conversation
    agent: str = "unknown"     # name of the sending agent; defaults to "unknown"
    turns: list[Turn]          # must be non-empty (validated below)
    started_at: str | None = None  # ISO timestamp of the first turn; generated if absent
    metadata: dict | None = None   # optional agent-supplied key-value pairs

    @field_validator("turns")
    @classmethod
    def turns_must_not_be_empty(cls, v: list) -> list:
        """Require at least one turn — an empty transcript is not useful."""
        if len(v) == 0:
            raise ValueError("turns must not be empty")
        return v


class RecallRequest(BaseModel):
    """Request body for POST /recall — build a wake-up digest for a prompt."""
    query: str = ""
    session_id: str | None = None


class AnswerRequest(BaseModel):
    """Request body for POST /answer — direct answer lookup for repeated prompts."""
    query: str
    session_id: str | None = None
    min_score: float = 0.93

    @field_validator("min_score")
    @classmethod
    def score_must_be_unit_interval(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("min_score must be between 0 and 1")
        return v


# ---------------------------------------------------------------------------
# Helper functions for /recall and /answer
# ---------------------------------------------------------------------------


def _read_identity() -> str:
    """Read ~/.memory/identity.md, returning an empty string if absent."""
    try:
        with open(IDENTITY_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _fetch_recent_sessions(conn, session_id: str | None) -> list[dict]:
    """Return a compact list of recent sessions, excluding the current one."""
    rows = conn.execute(
        """
        SELECT session_id, updated_at, turn_count, transcript
        FROM sessions
        WHERE (? IS NULL OR session_id != ?)
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (session_id, session_id, RECENT_SESSION_COUNT),
    ).fetchall()

    results = []
    for row in rows:
        try:
            turns = json.loads(row["transcript"] or "[]")
            first_user = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
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


def _fetch_relevant_context(conn, query: str, current_session_id: str | None) -> tuple[list[dict], list[dict]]:
    """
    Run semantic chunk search plus FTS fact search for a prompt.

    Returns a pair (sessions, facts). Sessions are de-duplicated per session_id
    and exclude the current session when session_id is provided.
    """
    chunk_results = semantic_search_chunks(conn, query, limit=CHUNK_SEARCH_LIMIT)

    best_chunks: dict[str, dict] = {}
    for chunk in chunk_results:
        sid = chunk["session_id"]
        if current_session_id and sid == current_session_id:
            continue
        if sid not in best_chunks or chunk["distance"] < best_chunks[sid]["distance"]:
            best_chunks[sid] = chunk

    sessions = []
    for sid, chunk in best_chunks.items():
        row = conn.execute(
            "SELECT updated_at, turn_count FROM sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
        if row is None:
            continue

        similarity = round((1 - chunk["distance"] / 2) * 100, 1)
        sessions.append({
            "session_id": sid,
            "updated_at": row["updated_at"],
            "turn_count": row["turn_count"],
            "snippet": chunk["snippet"],
            "similarity": similarity,
        })

    sessions.sort(key=lambda s: s["similarity"], reverse=True)

    facts: list[dict] = []
    try:
        facts = search_facts(conn, query, limit=FACT_SEARCH_LIMIT)
    except sqlite3.OperationalError:
        facts = []

    return sessions, facts


def _build_digest(
    identity: str,
    sessions: list[dict],
    facts: list[dict] | None,
    insights: list[dict] | None,
) -> str:
    """Format a wake-up digest that Pi can inject before an LLM turn."""
    lines = ["=== MEMORY WAKE-UP ===", ""]

    lines.append("[L0 — Identity]")
    if identity:
        lines.append(identity)
    else:
        lines.append("(no identity.md found — create ~/.memory/identity.md to set up your profile)")
    lines.append("")

    lines.append("[L1 — Relevant Sessions]" if facts is not None else "[L1 — Recent Sessions]")
    if sessions:
        for i, session in enumerate(sessions, start=1):
            updated = session.get("updated_at") or "unknown date"
            turns = session.get("turn_count") or 0
            if "similarity" in session:
                lines.append(
                    f"{i}. {updated} ({turns} turns, relevance {session['similarity']}%) — {session.get('snippet', '')}"
                )
            else:
                lines.append(
                    f"{i}. {updated} ({turns} turns) — {session.get('first_user_message', '')}"
                )
    else:
        lines.append("(no past sessions found)")
    lines.append("")

    if facts:
        lines.append("[L2 — Relevant Facts]")
        for fact in facts:
            tags = fact.get("tags", [])
            if tags:
                lines.append(f"• {fact.get('content', '')} [tags: {', '.join(tags)}]")
            else:
                lines.append(f"• {fact.get('content', '')}")
        lines.append("")

    if insights:
        lines.append("[L3 — Learned Insights]")
        for insight in insights:
            lines.append(
                f"• {insight.get('content', '')} ({insight.get('insight_type', 'pattern')}, confidence {insight.get('confidence', 0.0):.0%})"
            )
        lines.append("")

    lines.append("[System — Identity Updates]")
    lines.append(
        "If you learn something new and personal about the user during this conversation "
        "(their name, role, team, tech stack, project context, working style, or preferences), "
        "update ~/.memory/identity.md directly using file-write tools. "
        "Add to the relevant section or append a new bullet — one concise line per fact. "
        "Do this silently; don't announce it to the user."
    )
    lines.append("")
    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest")
def post_ingest(req: IngestRequest) -> dict:
    """
    Accept a session from any agent and run the full save pipeline.

    Steps:
      1. upsert_session — saves the transcript row to the sessions table
      2. embed + store_embedding — stores a semantic vector for the full text
      3. chunk_transcript + embed each chunk + store_chunk — finer-grained search

    Returns {"ok": true, "session_id": "..."} on success.
    Returns {"ok": false, "error": "..."} with HTTP 500 on failure.
    """
    try:
        # Convert Pydantic Turn objects back to plain dicts for the DB helpers.
        # The helpers expect list[{"role": str, "content": str}].
        turns_as_dicts = [{"role": t.role, "content": t.content} for t in req.turns]

        # Ensure the ~/.memory directory exists before opening the DB.
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        # Open a connection to the SQLite database.
        conn = init_db(DB_PATH)

        # Generate updated_at as the current UTC time in ISO 8601 format.
        updated_at = datetime.now(timezone.utc).isoformat()

        # Use provided started_at or fall back to updated_at (same moment).
        started_at = req.started_at or updated_at

        # Step 1: Save the session row (or update if it already exists).
        upsert_session(
            conn,
            session_id=req.session_id,
            agent=req.agent,
            transcript=turns_as_dicts,
            started_at=started_at,
            updated_at=updated_at,
            metadata=req.metadata,
        )

        # Step 2: Embed the full conversation text and store the vector.
        # We wrap this in try/except so a missing model never blocks the save.
        try:
            # Join all turn content into one string for the embedding model.
            full_text = " ".join(
                t.content for t in req.turns
                if isinstance(t.content, str)
            )
            if full_text.strip():
                vector = embed(full_text)
                store_embedding(conn, req.session_id, vector)
        except Exception as embed_exc:
            # Log the error but do not abort — the session is already saved.
            error_log("ingest", f"embedding failed for session {req.session_id}", exc=embed_exc)

        # Step 3: Chunk the transcript and embed each chunk for finer search.
        # We wrap this in try/except for the same reason as embedding above.
        try:
            chunk_texts = chunk_transcript(turns_as_dicts)
            # Remove any old chunks so we don't accumulate stale rows.
            delete_chunks_for_session(conn, req.session_id)
            for chunk_index, chunk_text in enumerate(chunk_texts):
                # Embed only non-empty chunks — empty strings produce poor vectors.
                chunk_vector = embed(chunk_text) if chunk_text.strip() else None
                store_chunk(conn, req.session_id, chunk_index, chunk_text, chunk_vector)
        except Exception as chunk_exc:
            error_log("ingest", f"chunking failed for session {req.session_id}", exc=chunk_exc)

        conn.close()

        # Record the ingest in the activity log for dashboard visibility.
        activity_log(
            "ingest", "ingest",
            session=req.session_id,
            agent=req.agent,
            turns=len(req.turns),
        )

        # Return success — the caller can confirm using the echoed session_id.
        return {"ok": True, "session_id": req.session_id}

    except Exception as exc:
        # Catch any unexpected error and return a structured 500 response.
        # We log it so operators can diagnose problems without tailing server output.
        error_log("ingest", "unhandled error in POST /ingest", exc=exc)
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.post("/recall")
def post_recall(req: RecallRequest) -> dict:
    """
    Build a memory wake-up digest for a prompt.

    This is the HTTP equivalent of the Claude wake-up hook: agents can call it
    before sending a prompt to the LLM and inject the returned digest as hidden
    context.
    """
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = init_db(DB_PATH)
        query = (req.query or "").strip()

        identity = _read_identity()
        sessions: list[dict] = []
        facts: list[dict] | None = None
        mode = "relevance"

        if query:
            try:
                sessions, facts = _fetch_relevant_context(conn, query, req.session_id)
            except Exception:
                sessions = []
                facts = None

        if not sessions:
            sessions = _fetch_recent_sessions(conn, req.session_id)
            facts = None
            mode = "recency"

        try:
            insights = list_insights(conn, limit=3)
        except Exception:
            insights = []

        digest = _build_digest(identity, sessions, facts, insights)
        log_retrieval(conn, "memory_recall", query or None, len(digest))
        activity_log(
            "ingest", "recall",
            session=req.session_id or "",
            mode=mode,
            sessions=len(sessions),
            facts=len(facts or []),
        )
        conn.close()

        return {
            "ok": True,
            "mode": mode,
            "digest": digest,
            "session_count": len(sessions),
            "fact_count": len(facts or []),
            "insight_count": len(insights or []),
        }
    except Exception as exc:
        error_log("ingest", "unhandled error in POST /recall", exc=exc)
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.post("/answer")
def post_answer(req: AnswerRequest) -> dict:
    """
    Try to answer a prompt directly from memory without using the LLM.

    This is intentionally conservative: it only returns an answer when the new
    prompt is a near-repeat of a previously answered user question.
    """
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = init_db(DB_PATH)
        match = find_direct_answer(
            conn,
            req.query,
            min_score=req.min_score,
            exclude_session_id=req.session_id,
        )

        result = {
            "ok": True,
            "answered": match is not None,
            "match": match,
        }
        log_retrieval(conn, "memory_direct_answer", req.query, len(json.dumps(result)))
        activity_log(
            "ingest", "answer",
            session=req.session_id or "",
            answered=match is not None,
            similarity=match["similarity"] if match else 0,
        )
        conn.close()
        return result
    except Exception as exc:
        error_log("ingest", "unhandled error in POST /answer", exc=exc)
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.get("/status")
def get_status() -> dict:
    """
    Return a brief health summary of the memory system.

    Useful for callers that want to confirm the server is up before
    sending a large ingest request.

    Returns:
        {
          "status": "ok",
          "total_sessions": <int>,
          "newest_session": "<ISO timestamp or null>",
          "db_size_bytes": <int>
        }
    """
    try:
        # Count sessions and find the most recent update timestamp.
        conn = init_db(DB_PATH)
        row = conn.execute(
            "SELECT COUNT(*) as total, MAX(updated_at) as newest FROM sessions"
        ).fetchone()
        total_sessions = row["total"] if row else 0
        newest_session = row["newest"] if row else None
        conn.close()
    except Exception:
        # If the DB isn't accessible yet, report zero without crashing.
        total_sessions = 0
        newest_session = None

    # os.path.getsize returns the file size in bytes; return 0 if file absent.
    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    return {
        "status": "ok",
        "total_sessions": total_sessions,
        "newest_session": newest_session,
        "db_size_bytes": db_size_bytes,
    }


# ---------------------------------------------------------------------------
# Entry point — run with: python3 memory/ingest_server.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # uvicorn.run() starts the ASGI server and blocks until interrupted (Ctrl-C).
    # log_level="warning" suppresses the per-request INFO lines to keep logs clean.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
