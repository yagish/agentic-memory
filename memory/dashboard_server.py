# dashboard_server.py — Human-facing dashboard and operational API.
#
# Runs on port 7748 (separate from the ingest server on 7747).
# Serves the dashboard UI and exposes read + operational endpoints.
#
# Routes are split into three APIRouter groups:
#   ui_router     — GET /          dashboard HTML (same-origin, no CORS)
#                   GET /services  live service health + memory stats
#                   GET /status    quick health check
#                   GET /stats/charts  time-series data for landing charts
#   memory_router — GET /memory/sessions|chunks|facts|insights
#   ops_router    — GET /logs/{service}   tail last N lines of a service log
#                   POST /restart/{service}  launchctl kickstart or open Ollama
#                   POST /compress           map-reduce session compression
#
# Run manually:
#   python3 memory/dashboard_server.py
#
# Managed by launchd via com.memory.query.plist.

import json
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

from memory.db import init_db

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = os.path.expanduser("~/.memory/memory.db")
DAEMON_LOG   = os.path.expanduser("~/.memory/daemon.log")
PORT         = int(os.environ.get("MEMORY_QUERY_PORT", "7748"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Log files written by launchd (StandardOutPath in each plist).
_LOG_FILES: dict[str, str] = {
    "daemon": os.path.expanduser("~/.memory/daemon.log"),
    "query":  os.path.expanduser("~/.memory/query.log"),
    "ingest": os.path.expanduser("~/.memory/ingest.log"),
    "ollama": "/opt/homebrew/var/log/ollama.log",
}

# launchd service labels for launchctl kickstart.
_LAUNCHCTL_LABELS: dict[str, str] = {
    "daemon": "com.memory.daemon",
    "query":  "com.memory.query",
    "ingest": "com.memory.ingest",
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _check_ollama() -> tuple[bool, str | None]:
    """Return (running, model_name|None). Running = API responds 200."""
    for path in ("/api/ps", "/api/tags"):
        try:
            with urllib.request.urlopen(f"http://localhost:11434{path}", timeout=2) as r:
                if r.status == 200:
                    models = json.loads(r.read()).get("models", [])
                    model = models[0].get("name") if models else None
                    return True, model
        except Exception:
            pass
    return False, None


def _parse_daemon_log() -> tuple[str | None, int]:
    """Scan daemon.log for last fact-extraction timestamp and running total."""
    last_run_iso, facts_total = None, 0
    if not os.path.exists(DAEMON_LOG):
        return last_run_iso, facts_total
    try:
        with open(DAEMON_LOG) as f:
            for line in f:
                line = line.strip()
                m = re.search(r"extracted (\d+) facts", line)
                if m:
                    facts_total += int(m.group(1))
                    last_run_iso = line.split()[0]
    except Exception:
        pass
    return last_run_iso, facts_total


def _check_daemon() -> tuple[bool, int | None]:
    """Ask launchctl if com.memory.daemon is running. Returns (running, pid)."""
    try:
        result = subprocess.run(
            ["launchctl", "list", "com.memory.daemon"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            m = re.search(r'"PID"\s*=\s*(\d+)', result.stdout)
            if m:
                return True, int(m.group(1))
    except Exception:
        pass
    return False, None


# ── UI router — dashboard HTML + service stats ────────────────────────────────

ui_router = APIRouter()


@ui_router.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve dashboard.html. Same-origin as the API so no CORS applies."""
    path = os.path.join(PROJECT_ROOT, "dashboard.html")
    try:
        with open(path) as f:
            html = f.read()
    except FileNotFoundError:
        html = f"<h1>dashboard.html not found</h1><p>Expected at: {path}</p>"
    return HTMLResponse(content=html)


@ui_router.get("/services")
def get_services() -> dict:
    """Full snapshot of all running services and memory stats for the dashboard."""
    daemon_running, _ = _check_daemon()
    last_run_iso, facts_extracted_total = _parse_daemon_log()
    ollama_running, ollama_model = _check_ollama()

    total_sessions = total_facts = total_insights = chunks_indexed = 0
    injections = tokens_from_memory = mcp_retrievals = 0
    recent_sessions: list[dict] = []

    try:
        conn = init_db(DB_PATH)
        total_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        total_facts    = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
        total_insights = conn.execute("SELECT COUNT(*) AS c FROM insights").fetchone()["c"]
        chunks_indexed = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]

        inj = conn.execute(
            "SELECT COUNT(*) AS c, SUM(result_size) AS t FROM retrievals WHERE tool = 'wake_up_injection'"
        ).fetchone()
        injections         = inj["c"] or 0
        tokens_from_memory = inj["t"] or 0
        mcp_retrievals     = conn.execute(
            "SELECT COUNT(*) AS c FROM retrievals WHERE tool != 'wake_up_injection'"
        ).fetchone()["c"]

        for s in conn.execute(
            "SELECT session_id, turn_count, started_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 3"
        ).fetchall():
            recent_sessions.append(dict(s))

        conn.close()
    except Exception:
        pass

    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    return {
        "dashboard_server": {"running": True, "port": PORT},
        "ingest_server":    {"port": 7747},
        "daemon":  {"running": daemon_running, "last_run": last_run_iso, "facts_extracted": facts_extracted_total},
        "ollama":  {"running": ollama_running, "model": ollama_model},
        "memory":  {"total_sessions": total_sessions, "total_facts": total_facts,
                    "total_insights": total_insights, "chunks_indexed": chunks_indexed,
                    "db_size_bytes": db_size_bytes},
        "retrieval": {"injections": injections, "tokens_from_memory": tokens_from_memory,
                      "mcp_retrievals": mcp_retrievals},
        "recent_sessions": recent_sessions,
    }


@ui_router.get("/status")
def get_status() -> dict:
    """Quick health check — server up, session count, DB size."""
    try:
        conn = init_db(DB_PATH)
        row = conn.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS newest FROM sessions").fetchone()
        total_sessions = row["total"] if row else 0
        newest_session = row["newest"] if row else None
        conn.close()
    except Exception:
        total_sessions, newest_session = 0, None
    return {
        "status": "ok",
        "total_sessions": total_sessions,
        "newest_session": newest_session,
        "db_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
    }


@ui_router.get("/stats/charts")
def get_chart_stats() -> dict:
    """Time-series data for the landing-page charts (last 14 days)."""
    try:
        conn = init_db(DB_PATH)
        sessions_by_day = [dict(r) for r in conn.execute(
            """
            SELECT DATE(started_at) AS day, COUNT(*) AS count
            FROM sessions WHERE started_at >= DATE('now', '-14 days')
            GROUP BY day ORDER BY day
            """
        ).fetchall()]
        token_economics = [dict(r) for r in conn.execute(
            """
            SELECT DATE(queried_at) AS day,
                   COALESCE(SUM(CASE WHEN tool = 'wake_up_injection' THEN result_size END), 0) AS memory_tokens,
                   COALESCE(COUNT(CASE WHEN tool = 'wake_up_injection' THEN 1 END), 0)         AS injections
            FROM retrievals WHERE queried_at >= DATE('now', '-14 days')
            GROUP BY day ORDER BY day
            """
        ).fetchall()]
        facts_by_day = [dict(r) for r in conn.execute(
            """
            SELECT DATE(created_at) AS day, COUNT(*) AS count
            FROM facts WHERE created_at >= DATE('now', '-14 days')
            GROUP BY day ORDER BY day
            """
        ).fetchall()]
        conn.close()
        return {"sessions_by_day": sessions_by_day, "token_economics": token_economics, "facts_by_day": facts_by_day}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Memory router — data store reads ─────────────────────────────────────────

memory_router = APIRouter(prefix="/memory")


@memory_router.get("/sessions")
def get_memory_sessions() -> dict:
    """All sessions with turn counts, dates, and first-user-message preview."""
    rows: list[dict] = []
    try:
        conn = init_db(DB_PATH)
        for s in conn.execute(
            "SELECT session_id, agent, turn_count, started_at, updated_at, transcript FROM sessions ORDER BY updated_at DESC"
        ).fetchall():
            preview = ""
            try:
                turns   = json.loads(s["transcript"])
                first   = next((t["content"] for t in turns if t.get("role") == "user"), "")
                preview = first[:200].replace("\n", " ")
            except Exception:
                pass
            rows.append({"session_id": s["session_id"], "agent": s["agent"],
                         "turn_count": s["turn_count"], "started_at": s["started_at"],
                         "updated_at": s["updated_at"], "preview": preview})
        conn.close()
    except Exception:
        pass
    return {"sessions": rows, "total": len(rows)}


@memory_router.get("/chunks")
def get_memory_chunks() -> dict:
    """All semantic search chunks (text trimmed to 300 chars for display)."""
    rows: list[dict] = []
    try:
        conn = init_db(DB_PATH)
        for c in conn.execute(
            "SELECT id, session_id, chunk_index, text, created_at FROM chunks ORDER BY created_at DESC"
        ).fetchall():
            rows.append({"id": c["id"], "session_id": c["session_id"],
                         "chunk_index": c["chunk_index"],
                         "text": (c["text"] or "")[:300], "created_at": c["created_at"]})
        conn.close()
    except Exception:
        pass
    return {"chunks": rows, "total": len(rows)}


@memory_router.get("/facts")
def get_memory_facts() -> dict:
    """All extracted facts with tags and source session."""
    rows: list[dict] = []
    try:
        conn = init_db(DB_PATH)
        for f in conn.execute(
            "SELECT id, content, tags, source, session_id, created_at FROM facts ORDER BY created_at DESC"
        ).fetchall():
            try:
                tags = json.loads(f["tags"]) if f["tags"] else []
            except Exception:
                tags = []
            rows.append({"id": f["id"], "content": f["content"], "tags": tags,
                         "source": f["source"], "session_id": f["session_id"],
                         "created_at": f["created_at"]})
        conn.close()
    except Exception:
        pass
    return {"facts": rows, "total": len(rows)}


@memory_router.get("/insights")
def get_memory_insights() -> dict:
    """All cross-session insights generated by the daemon."""
    rows: list[dict] = []
    try:
        conn = init_db(DB_PATH)
        for i in conn.execute(
            "SELECT id, insight_type, content, confidence, created_at, updated_at FROM insights ORDER BY updated_at DESC"
        ).fetchall():
            rows.append({"id": i["id"], "insight_type": i["insight_type"],
                         "content": i["content"],
                         "confidence": round(float(i["confidence"] or 0), 2),
                         "created_at": i["created_at"], "updated_at": i["updated_at"]})
        conn.close()
    except Exception:
        pass
    return {"insights": rows, "total": len(rows)}


# ── Ops router — logs, restarts, compression ──────────────────────────────────

ops_router = APIRouter()


@ops_router.get("/logs/{service}")
def get_logs(service: str, lines: int = 200) -> dict:
    """Return the last N lines of a service log file."""
    if service not in _LOG_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    path = _LOG_FILES[service]
    if not os.path.exists(path):
        return {"lines": [], "service": service, "exists": False}
    try:
        result = subprocess.run(["tail", f"-{lines}", path],
                                capture_output=True, text=True, timeout=5)
        return {"lines": result.stdout.splitlines(), "service": service, "exists": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@ops_router.post("/restart/{service}")
def post_restart(service: str) -> dict:
    """Restart a memory service via launchctl kickstart, or open Ollama."""
    if service == "ollama":
        ollama_bin = None
        for candidate in ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"]:
            if os.path.isfile(candidate):
                ollama_bin = candidate
                break
        if ollama_bin is None:
            raise HTTPException(status_code=500, detail="ollama binary not found")
        try:
            subprocess.Popen([ollama_bin, "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"ok": True, "service": service}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    if service not in _LAUNCHCTL_LABELS:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    uid = os.getuid()
    try:
        result = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{_LAUNCHCTL_LABELS[service]}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500,
                                detail=result.stderr.strip() or "launchctl kickstart failed")
        return {"ok": True, "service": service}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@ops_router.post("/compress")
def post_compress() -> dict:
    """
    Compress all sessions into a structured memory document and delete them.

    Lives here (dashboard server) so the dashboard can call it same-origin
    without a cross-origin POST to the ingest server.
    """
    from memory.compress import compress_memory  # noqa: PLC0415

    try:
        conn = init_db(DB_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    try:
        result = compress_memory(conn)
        return {"ok": True, "sessions_compressed": result.sessions_compressed,
                "model": result.model, "created_at": result.created_at,
                "content": result.content}
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    finally:
        conn.close()


# ── App assembly ──────────────────────────────────────────────────────────────

app = FastAPI(title="Memory Dashboard Server", version="1.0.0")
app.include_router(ui_router)
app.include_router(memory_router)
app.include_router(ops_router)

if __name__ == "__main__":
    import logging
    import setproctitle
    setproctitle.setproctitle("AgenticMemoryDashboard")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
