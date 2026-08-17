# ingest_server.py — HTTP ingest endpoint for agent-agnostic session writes.
#
# Any agent (Cursor, LangChain, custom scripts) can POST a session here
# and it will be stored, chunked, and embedded exactly as if it came from
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
from fastapi.responses import FileResponse          # for serving the Pi userscript
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
)
from memory.logger import activity_log, error_log  # structured logging helpers

# Where the SQLite database lives on disk — must match the hook and MCP server.
DB_PATH = os.path.expanduser("~/.memory/memory.db")

# Port the server listens on — override with MEMORY_INGEST_PORT environment variable.
PORT = int(os.environ.get("MEMORY_INGEST_PORT", "7747"))

# Create the FastAPI application instance.
# The title and version appear in the auto-generated /docs page.
app = FastAPI(title="Memory Ingest Server", version="1.0.0")

# ── CORS — allow browser-based agents to POST from their domains ──────────────
# Browsers enforce the Same-Origin Policy: a page at https://pi.ai cannot
# fetch http://localhost:7747 unless the server explicitly says it's OK.
# CORSMiddleware adds the necessary Access-Control-* response headers.
#
# Override the allowed list with MEMORY_INGEST_CORS_ORIGINS (comma-separated):
#   export MEMORY_INGEST_CORS_ORIGINS="https://pi.ai,https://github.com"
_default_origins = "https://pi.ai,https://copilot.microsoft.com,https://github.com"
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


@app.get("/pi-script")
def get_pi_script() -> FileResponse:
    """
    Serve the Pi Tampermonkey userscript over HTTP.

    Tampermonkey intercepts .user.js URLs from HTTP servers and shows its
    install dialog automatically — much more reliably than file:// URLs.
    The install.sh script opens this endpoint in the browser after loading
    Tampermonkey, so the user just needs to click Install once.
    """
    # The userscript lives at integrations/pi/pi_memory.user.js relative to
    # the project root.  __file__ is memory/ingest_server.py, so go up twice.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, "integrations", "pi", "pi_memory.user.js")

    if not os.path.exists(script_path):
        raise HTTPException(
            status_code=404,
            detail="Pi userscript not found — make sure integrations/pi/pi_memory.user.js exists",
        )

    # application/javascript with the .user.js filename is the MIME type
    # Tampermonkey watches for when deciding to show the install dialog.
    return FileResponse(
        script_path,
        media_type="application/javascript",
        filename="pi_memory.user.js",
    )


# ---------------------------------------------------------------------------
# Entry point — run with: python3 memory/ingest_server.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # uvicorn.run() starts the ASGI server and blocks until interrupted (Ctrl-C).
    # log_level="warning" suppresses the per-request INFO lines to keep logs clean.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
