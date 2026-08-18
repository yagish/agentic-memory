# query_server.py — Read-only dashboard and memory query server.
#
# Runs on port 7748 (separate from the ingest server on 7747).
# This server is for humans (dashboard) and read-only lookups only.
# It never writes to the database.
#
# Routes:
#   GET  /           — serves dashboard.html (same-origin, no CORS needed)
#   GET  /services   — live service health: daemon, ollama, DB stats
#   GET  /status     — quick health check
#   GET  /memory/sessions — all sessions with turn counts and message preview
#   GET  /memory/chunks   — last 20 semantic search chunks with text
#   GET  /memory/facts    — all extracted facts with tags
#
# Run manually:
#   python3 memory/query_server.py
#
# Managed by launchd via com.memory.query.plist.

import json           # JSON parsing for ollama API responses
import os             # file paths, env vars
import re             # parsing daemon.log lines
import subprocess     # querying launchctl for daemon PID
import sys            # module path manipulation
import urllib.request # checking ollama status (no third-party deps)

# Add project root to path so we can import from the memory package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse  # returns raw HTML for the dashboard
import uvicorn

from memory.db import init_db  # opens (or creates) the SQLite DB with schema applied

# ── Config ───────────────────────────────────────────────────────────────────
DB_PATH       = os.path.expanduser("~/.memory/memory.db")
DAEMON_LOG    = os.path.expanduser("~/.memory/daemon.log")

# Port this server listens on.  Override with MEMORY_QUERY_PORT env var.
PORT = int(os.environ.get("MEMORY_QUERY_PORT", "7748"))

# Absolute path to the project root so we can find dashboard.html.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Memory Query Server", version="1.0.0")


# ── Helper: ollama status ─────────────────────────────────────────────────────

def _check_ollama() -> tuple[bool, str | None]:
    """
    Check if ollama is running and which model is available.

    Returns (running: bool, model_name: str | None).
    Tries /api/ps first (models loaded in memory), falls back to /api/tags
    (models downloaded but not yet loaded).
    """
    # Try /api/ps — shows models currently held in GPU/CPU memory.
    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=2) as resp:
            if resp.status == 200:
                data = json.loads(resp.read())
                models = data.get("models", [])
                if models:
                    return True, models[0].get("name")
    except Exception:
        pass

    # Fall back to /api/tags — at least the server is up and has a model.
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            if resp.status == 200:
                data = json.loads(resp.read())
                models = data.get("models", [])
                if models:
                    return True, models[0].get("name")
    except Exception:
        pass

    return False, None


# ── Helper: daemon log parsing ────────────────────────────────────────────────

def _parse_daemon_log() -> tuple[str | None, int]:
    """
    Scan daemon.log for the most recent fact-extraction timestamp and total count.

    Returns (last_run_iso: str | None, facts_total: int).
    """
    last_run_iso = None
    facts_total = 0
    if not os.path.exists(DAEMON_LOG):
        return last_run_iso, facts_total
    try:
        with open(DAEMON_LOG) as f:
            for line in f:
                line = line.strip()
                m = re.search(r"extracted (\d+) facts", line)
                if m:
                    facts_total += int(m.group(1))
                    last_run_iso = line.split()[0]   # ISO timestamp is the first token
    except Exception:
        pass
    return last_run_iso, facts_total


# ── Helper: daemon launchctl check ───────────────────────────────────────────

def _check_daemon() -> tuple[bool, int | None]:
    """
    Ask launchctl if com.memory.daemon is running.

    Returns (running: bool, pid: int | None).
    """
    try:
        result = subprocess.run(
            ["launchctl", "list", "com.memory.daemon"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            pid_match = re.search(r'"PID"\s*=\s*(\d+)', result.stdout)
            if pid_match:
                return True, int(pid_match.group(1))
    except Exception:
        pass
    return False, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """
    Serve the dashboard HTML.  Since the page is fetched from the same origin
    as the API (both on port 7748), all fetch() calls in the page are
    same-origin and CORS never applies.
    """
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard.html")
    try:
        with open(dashboard_path) as f:
            html = f.read()
    except FileNotFoundError:
        html = f"<h1>dashboard.html not found</h1><p>Expected at: {dashboard_path}</p>"
    return HTMLResponse(content=html)


@app.get("/services")
def get_services() -> dict:
    """
    Return a full snapshot of all running services and memory stats.

    Used by the dashboard to populate every section on the page.
    """
    # Service checks — run first, they're fast.
    daemon_running, _ = _check_daemon()
    last_run_iso, facts_extracted_total = _parse_daemon_log()
    ollama_running, ollama_model = _check_ollama()

    # DB reads.
    total_sessions = 0
    total_facts = 0
    total_insights = 0
    chunks_indexed = 0
    injections = 0
    tokens_from_memory = 0
    mcp_retrievals = 0
    recent_sessions = []

    try:
        conn = init_db(DB_PATH)

        # Core counts.
        total_sessions  = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        total_facts     = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
        total_insights  = conn.execute("SELECT COUNT(*) AS c FROM insights").fetchone()["c"]
        chunks_indexed  = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]

        # Retrieval stats.
        inj = conn.execute(
            "SELECT COUNT(*) AS c, SUM(result_size) AS t FROM retrievals WHERE tool = 'wake_up_injection'"
        ).fetchone()
        injections        = inj["c"] or 0
        tokens_from_memory = inj["t"] or 0

        mcp_retrievals = conn.execute(
            "SELECT COUNT(*) AS c FROM retrievals WHERE tool != 'wake_up_injection'"
        ).fetchone()["c"]

        # Last 3 sessions for the dashboard table.
        for s in conn.execute(
            "SELECT session_id, turn_count, started_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 3"
        ).fetchall():
            recent_sessions.append({
                "session_id": s["session_id"],
                "turn_count": s["turn_count"],
                "started_at": s["started_at"],
                "updated_at": s["updated_at"],
            })

        conn.close()
    except Exception:
        pass

    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    return {
        "query_server":  {"running": True, "port": PORT},
        "ingest_server": {"port": 7747},   # write server — separate process
        "daemon":  {"running": daemon_running, "last_run": last_run_iso, "facts_extracted": facts_extracted_total},
        "ollama":  {"running": ollama_running, "model": ollama_model},
        "memory":  {"total_sessions": total_sessions, "total_facts": total_facts,
                    "total_insights": total_insights,
                    "chunks_indexed": chunks_indexed, "db_size_bytes": db_size_bytes},
        "retrieval": {"injections": injections, "tokens_from_memory": tokens_from_memory,
                      "mcp_retrievals": mcp_retrievals},
        "recent_sessions": recent_sessions,
    }


@app.get("/status")
def get_status() -> dict:
    """Quick health check — confirms the server is up and returns session count."""
    try:
        conn = init_db(DB_PATH)
        row = conn.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS newest FROM sessions").fetchone()
        total_sessions = row["total"] if row else 0
        newest_session = row["newest"] if row else None
        conn.close()
    except Exception:
        total_sessions = 0
        newest_session = None
    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    return {"status": "ok", "total_sessions": total_sessions,
            "newest_session": newest_session, "db_size_bytes": db_size_bytes}


@app.get("/memory/sessions")
def get_memory_sessions() -> dict:
    """All sessions with turn counts, dates, and a preview of the first user message."""
    rows = []
    try:
        conn = init_db(DB_PATH)
        for s in conn.execute(
            "SELECT session_id, agent, turn_count, started_at, updated_at, transcript FROM sessions ORDER BY updated_at DESC"
        ).fetchall():
            preview = ""
            try:
                turns = json.loads(s["transcript"])
                first = next((t["content"] for t in turns if t.get("role") == "user"), "")
                preview = first[:200].replace("\n", " ")
            except Exception:
                pass
            rows.append({
                "session_id": s["session_id"],
                "agent":      s["agent"],
                "turn_count": s["turn_count"],
                "started_at": s["started_at"],
                "updated_at": s["updated_at"],
                "preview":    preview,
            })
        conn.close()
    except Exception:
        pass
    return {"sessions": rows, "total": len(rows)}


@app.get("/memory/chunks")
def get_memory_chunks() -> dict:
    """All semantic search chunks with their text content."""
    rows = []
    try:
        conn = init_db(DB_PATH)
        for c in conn.execute(
            "SELECT id, session_id, chunk_index, text, created_at FROM chunks ORDER BY created_at DESC"
        ).fetchall():
            rows.append({
                "id":          c["id"],
                "session_id":  c["session_id"],
                "chunk_index": c["chunk_index"],
                "text":        (c["text"] or "")[:300],   # trim for display
                "created_at":  c["created_at"],
            })
        conn.close()
    except Exception:
        pass
    return {"chunks": rows, "total": len(rows)}


@app.get("/memory/facts")
def get_memory_facts() -> dict:
    """All extracted facts with tags and source session."""
    rows = []
    try:
        conn = init_db(DB_PATH)
        for f in conn.execute(
            "SELECT id, content, tags, source_session_id, created_at FROM facts ORDER BY created_at DESC"
        ).fetchall():
            try:
                tags = json.loads(f["tags"]) if f["tags"] else []
            except Exception:
                tags = []
            rows.append({
                "id":                f["id"],
                "content":           f["content"],
                "tags":              tags,
                "source_session_id": f["source_session_id"],
                "created_at":        f["created_at"],
            })
        conn.close()
    except Exception:
        pass
    return {"facts": rows, "total": len(rows)}


@app.get("/memory/insights")
def get_memory_insights() -> dict:
    """All cross-session insights generated by the daemon."""
    rows = []
    try:
        conn = init_db(DB_PATH)
        for i in conn.execute(
            "SELECT id, insight_type, content, confidence, created_at, updated_at FROM insights ORDER BY updated_at DESC"
        ).fetchall():
            rows.append({
                "id":           i["id"],
                "insight_type": i["insight_type"],
                "content":      i["content"],
                "confidence":   round(float(i["confidence"] or 0), 2),
                "created_at":   i["created_at"],
                "updated_at":   i["updated_at"],
            })
        conn.close()
    except Exception:
        pass
    return {"insights": rows, "total": len(rows)}


@app.post("/compress")
def post_compress() -> dict:
    """
    Compress all sessions into a structured memory document and delete them.

    This is the only write endpoint on the query server — it exists here so
    the dashboard (same-origin, port 7748) can trigger compression without
    a cross-origin fetch to the ingest server.

    Returns the compressed content and session count on success, or raises
    HTTP 500 if ollama is unreachable or there are no sessions to compress.
    """
    # Lazy import so compress.py's sys.path manipulation doesn't run at startup.
    from memory.compress import compress_memory  # noqa: PLC0415

    try:
        conn = init_db(DB_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    try:
        result = compress_memory(conn)
        return {
            "ok": True,
            "sessions_compressed": result.sessions_compressed,
            "model": result.model,
            "created_at": result.created_at,
            "content": result.content,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    finally:
        conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
