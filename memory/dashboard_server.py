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
#   memory_router — GET /memory/sessions|facts|insights|working|episodic|compacted|procedural
#                   GET /memory/chunks (implementation detail / debug view)
#   ops_router    — GET /logs/{service}   tail last N lines of a service log
#                   POST /restart/{service}  launchctl kickstart or open Ollama
#                   POST /compress           trigger daemon to process unprocessed sessions
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
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

from memory.db import bootstrap_db, open_db
from memory.debug import enable_debug

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH      = os.path.expanduser("~/.memory/memory.db")
DAEMON_LOG   = os.path.expanduser("~/.memory/daemon.log")
PORT         = int(os.environ.get("MEMORY_QUERY_PORT", "7748"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dashboard-log sources. Some have multiple candidate paths depending on install style.
_LOG_SOURCES: dict[str, dict[str, object]] = {
    "query": {
        "label": "Query Server",
        "description": "Dashboard + query API stdout/stderr",
        "kind": "service",
        "paths": [os.path.expanduser("~/.memory/query.log")],
    },
    "daemon": {
        "label": "Daemon",
        "description": "Background consolidation stdout/stderr",
        "kind": "service",
        "paths": [os.path.expanduser("~/.memory/daemon.log")],
    },
    "ingest": {
        "label": "Ingest Server",
        "description": "Write API stdout/stderr",
        "kind": "service",
        "paths": [os.path.expanduser("~/.memory/ingest.log")],
    },
    "ollama": {
        "label": "Ollama",
        "description": "Local LLM runtime logs",
        "kind": "service",
        "paths": [
            os.environ.get("OLLAMA_LOG_PATH", ""),
            "/opt/homebrew/var/log/ollama.log",
            "/usr/local/var/log/ollama.log",
            os.path.expanduser("~/Library/Logs/Ollama/server.log"),
            os.path.expanduser("~/Library/Logs/Ollama/ollama.log"),
            os.path.expanduser("~/.ollama/logs/server.log"),
        ],
    },
    "activity": {
        "label": "Activity Log",
        "description": "Structured memory operations across the app",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/activity.log")],
    },
    "error": {
        "label": "Error Log",
        "description": "Structured errors across all components",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/error.log")],
    },
    "wake_up": {
        "label": "Wake Up Hook",
        "description": "Prompt-injection hook log",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/wake_up.log")],
    },
    "save_hook": {
        "label": "Save Hook",
        "description": "Session-save hook log",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/save_hook.log")],
    },
    "debug": {
        "label": "Debug Log",
        "description": "Optional debugger attach log",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/debug.log")],
    },
    "facts_debug": {
        "label": "Facts Debug",
        "description": "Prompt/session/raw-output debug log for fact extraction",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/facts_debug.log")],
    },
    "compress_debug": {
        "label": "Compress Debug",
        "description": "Before/after log for each compression LLM call",
        "kind": "app_log",
        "paths": [os.path.expanduser("~/.memory/compress_debug.log")],
    },
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


def _resolve_log_path(name: str) -> str | None:
    """Return the first existing log path for a configured source, if any."""
    cfg = _LOG_SOURCES.get(name) or {}
    for candidate in cfg.get("paths", []):
        if candidate and os.path.exists(str(candidate)):
            return str(candidate)
    return None


def _describe_log_source(name: str) -> dict:
    """Metadata for a log source, including resolved file path if present."""
    cfg = _LOG_SOURCES[name]
    resolved = _resolve_log_path(name)
    candidates = [str(p) for p in cfg.get("paths", []) if p]
    path = resolved or (candidates[0] if candidates else None)
    exists = bool(resolved)
    size_bytes = os.path.getsize(resolved) if resolved and os.path.exists(resolved) else 0
    return {
        "name": name,
        "label": cfg["label"],
        "description": cfg["description"],
        "kind": cfg["kind"],
        "path": path,
        "exists": exists,
        "size_bytes": size_bytes,
        "candidates": candidates,
    }


def _list_logs(kind: str | None = None) -> list[dict]:
    logs = [_describe_log_source(name) for name in _LOG_SOURCES]
    if kind is not None:
        logs = [log for log in logs if log["kind"] == kind]
    return logs


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


def _check_launchd_service(label: str) -> tuple[bool, int | None]:
    """Ask launchctl if a launchd service is running. Returns (running, pid)."""
    try:
        result = subprocess.run(
            ["launchctl", "list", label],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            m = re.search(r'"PID"\s*=\s*(\d+)', result.stdout)
            if m:
                return True, int(m.group(1))
    except Exception:
        pass
    return False, None


def _check_daemon() -> tuple[bool, int | None]:
    """Ask launchctl if com.memory.daemon is running. Returns (running, pid)."""
    return _check_launchd_service("com.memory.daemon")


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
    ingest_running, _ = _check_launchd_service("com.memory.ingest")
    last_run_iso, facts_extracted_total = _parse_daemon_log()
    ollama_running, ollama_model = _check_ollama()

    total_sessions = total_facts = total_insights = chunks_indexed = 0
    total_working_memory = active_working_memory = 0
    total_episodic = total_compacted = total_procedural = 0
    injections = tokens_from_memory = mcp_retrievals = 0
    recent_sessions: list[dict] = []

    try:
        conn = open_db(DB_PATH)
        try:
            total_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
            total_facts    = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
            total_insights = conn.execute("SELECT COUNT(*) AS c FROM insights").fetchone()["c"]
            chunks_indexed = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
            total_working_memory = conn.execute("SELECT COUNT(*) AS c FROM working_memory").fetchone()["c"]
            active_working_memory = conn.execute(
                "SELECT COUNT(*) AS c FROM working_memory WHERE closed_at IS NULL"
            ).fetchone()["c"]
            total_episodic = conn.execute("SELECT COUNT(*) AS c FROM episodic_memory").fetchone()["c"]
            total_compacted = conn.execute("SELECT COUNT(*) AS c FROM compacted_sessions").fetchone()["c"]
            total_procedural = conn.execute("SELECT COUNT(*) AS c FROM procedural_memory").fetchone()["c"]

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
        finally:
            conn.close()
    except Exception:
        pass

    db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    return {
        "dashboard_server": {"running": True, "port": PORT},
        "ingest_server":    {"running": ingest_running, "port": 7747},
        "daemon":  {"running": daemon_running, "last_run": last_run_iso, "facts_extracted": facts_extracted_total},
        "ollama":  {"running": ollama_running, "model": ollama_model},
        "logs": {
            "services": _list_logs("service"),
            "app": _list_logs("app_log"),
        },
        "memory":  {
            "total_sessions": total_sessions,
            "total_facts": total_facts,
            "total_insights": total_insights,
            "chunks_indexed": chunks_indexed,
            "total_working_memory": total_working_memory,
            "active_working_memory": active_working_memory,
            "total_episodic": total_episodic,
            "total_compacted": total_compacted,
            "total_procedural": total_procedural,
            "db_size_bytes": db_size_bytes,
        },
        "retrieval": {"injections": injections, "tokens_from_memory": tokens_from_memory,
                      "mcp_retrievals": mcp_retrievals},
        "recent_sessions": recent_sessions,
    }


@ui_router.get("/status")
def get_status() -> dict:
    """Quick health check — server up, session count, DB size."""
    try:
        conn = open_db(DB_PATH)
        try:
            row = conn.execute("SELECT COUNT(*) AS total, MAX(updated_at) AS newest FROM sessions").fetchone()
            total_sessions = row["total"] if row else 0
            newest_session = row["newest"] if row else None
        finally:
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
        conn = open_db(DB_PATH)
        try:
            sessions_by_day = [dict(r) for r in conn.execute(
                """
                SELECT DATE(started_at) AS day, COUNT(*) AS count
                FROM sessions WHERE started_at >= DATE('now', '-14 days')
                GROUP BY day ORDER BY day
                """
            ).fetchall()]
            token_economics = [dict(r) for r in conn.execute(
                """
                SELECT DATE(called_at) AS day,
                       COALESCE(SUM(CASE WHEN tool = 'wake_up_injection' THEN result_size END), 0) AS memory_tokens,
                       COALESCE(COUNT(CASE WHEN tool = 'wake_up_injection' THEN 1 END), 0)         AS injections
                FROM retrievals WHERE called_at >= DATE('now', '-14 days')
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
        finally:
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
        conn = open_db(DB_PATH)
        try:
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
        finally:
            conn.close()
    except Exception:
        pass
    return {"sessions": rows, "total": len(rows)}


@memory_router.get("/sessions/{session_id}/transcript")
def get_session_transcript(session_id: str) -> dict:
    """Return the full transcript JSON for one session."""
    try:
        conn = open_db(DB_PATH)
        try:
            row = conn.execute(
                "SELECT session_id, agent, turn_count, started_at, updated_at, transcript FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return {"error": "session not found"}
            # transcript is stored as a JSON string; parse it so the client gets an array.
            try:
                turns = json.loads(row["transcript"] or "[]")
            except Exception:
                turns = []
            return {
                "session_id": row["session_id"],
                "agent": row["agent"],
                "turn_count": row["turn_count"],
                "started_at": row["started_at"],
                "updated_at": row["updated_at"],
                "transcript": turns,
            }
        finally:
            conn.close()
    except Exception as exc:
        return {"error": str(exc)}


@memory_router.get("/compressed")
def get_compressed_memory() -> dict:
    """Return all compressed_memory rows, newest first."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for r in conn.execute(
                "SELECT id, content, sessions_compressed, model, created_at FROM compressed_memory ORDER BY id DESC"
            ).fetchall():
                rows.append({
                    "id": r["id"],
                    "content": r["content"],
                    "sessions_compressed": r["sessions_compressed"],
                    "model": r["model"],
                    "created_at": r["created_at"],
                })
        finally:
            conn.close()
    except Exception:
        pass
    return {"compressed": rows, "total": len(rows)}


@memory_router.get("/chunks")
def get_memory_chunks() -> dict:
    """All semantic search chunks (text trimmed to 300 chars for display)."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for c in conn.execute(
                "SELECT id, session_id, chunk_index, text, created_at FROM chunks ORDER BY created_at DESC"
            ).fetchall():
                rows.append({"id": c["id"], "session_id": c["session_id"],
                             "chunk_index": c["chunk_index"],
                             "text": (c["text"] or "")[:300], "created_at": c["created_at"]})
        finally:
            conn.close()
    except Exception:
        pass
    return {"chunks": rows, "total": len(rows)}


@memory_router.get("/facts")
def get_memory_facts() -> dict:
    """All extracted facts with tags and source session."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
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
        finally:
            conn.close()
    except Exception:
        pass
    return {"facts": rows, "total": len(rows)}


@memory_router.get("/working")
def get_working_memory_entries() -> dict:
    """All working-memory entries, newest first, with active/closed state."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for wm in conn.execute(
                """
                SELECT id, cluster_id, summary, session_ids, created_at, updated_at, closed_at
                FROM working_memory
                ORDER BY updated_at DESC
                """
            ).fetchall():
                try:
                    session_ids = json.loads(wm["session_ids"]) if wm["session_ids"] else []
                except Exception:
                    session_ids = []
                rows.append({
                    "id": wm["id"],
                    "cluster_id": wm["cluster_id"],
                    "summary": wm["summary"],
                    "session_ids": session_ids,
                    "session_count": len(session_ids),
                    "created_at": wm["created_at"],
                    "updated_at": wm["updated_at"],
                    "closed_at": wm["closed_at"],
                    "status": "closed" if wm["closed_at"] else "active",
                })
        finally:
            conn.close()
    except Exception:
        pass
    return {"working": rows, "total": len(rows)}


@memory_router.get("/episodic")
def get_episodic_memory_entries() -> dict:
    """All episodic-memory entries generated by the daemon."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for ep in conn.execute(
                "SELECT id, session_id, title, abstract, happened_at FROM episodic_memory ORDER BY happened_at DESC"
            ).fetchall():
                rows.append({
                    "id": ep["id"],
                    "session_id": ep["session_id"],
                    "title": ep["title"],
                    "abstract": ep["abstract"],
                    "happened_at": ep["happened_at"],
                })
        finally:
            conn.close()
    except Exception:
        pass
    return {"episodic": rows, "total": len(rows)}


@memory_router.get("/compacted")
def get_compacted_memory_entries() -> dict:
    """All compacted cluster summaries used for semantic cache / enrichment."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for cs in conn.execute(
                """
                SELECT id, cluster_id, content, source_session_ids, hit_count, created_at, updated_at
                FROM compacted_sessions
                ORDER BY updated_at DESC
                """
            ).fetchall():
                try:
                    source_session_ids = json.loads(cs["source_session_ids"]) if cs["source_session_ids"] else []
                except Exception:
                    source_session_ids = []
                rows.append({
                    "id": cs["id"],
                    "cluster_id": cs["cluster_id"],
                    "content": cs["content"],
                    "source_session_ids": source_session_ids,
                    "source_session_count": len(source_session_ids),
                    "hit_count": cs["hit_count"] or 0,
                    "created_at": cs["created_at"],
                    "updated_at": cs["updated_at"],
                })
        finally:
            conn.close()
    except Exception:
        pass
    return {"compacted": rows, "total": len(rows)}


@memory_router.get("/procedural")
def get_procedural_memory_entries() -> dict:
    """All reusable how-to patterns extracted from past sessions."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for proc in conn.execute(
                """
                SELECT id, title, steps, confidence, observation_count, created_at, updated_at
                FROM procedural_memory
                ORDER BY confidence DESC, observation_count DESC, updated_at DESC
                """
            ).fetchall():
                rows.append({
                    "id": proc["id"],
                    "title": proc["title"],
                    "steps": proc["steps"],
                    "confidence": round(float(proc["confidence"] or 0), 2),
                    "observation_count": proc["observation_count"] or 0,
                    "created_at": proc["created_at"],
                    "updated_at": proc["updated_at"],
                })
        finally:
            conn.close()
    except Exception:
        pass
    return {"procedural": rows, "total": len(rows)}


@memory_router.get("/insights")
def get_memory_insights() -> dict:
    """All cross-session insights generated by the daemon."""
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for i in conn.execute(
                "SELECT id, insight_type, content, confidence, created_at, updated_at FROM insights ORDER BY updated_at DESC"
            ).fetchall():
                rows.append({"id": i["id"], "insight_type": i["insight_type"],
                             "content": i["content"],
                             "confidence": round(float(i["confidence"] or 0), 2),
                             "created_at": i["created_at"], "updated_at": i["updated_at"]})
        finally:
            conn.close()
    except Exception:
        pass
    return {"insights": rows, "total": len(rows)}


# ── Ops router — logs, restarts, compression ──────────────────────────────────

ops_router = APIRouter()


@ops_router.get("/logs/{service}")
def get_logs(service: str, lines: int = 200) -> dict:
    """Return the last N lines of a dashboard-visible log source."""
    if service not in _LOG_SOURCES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    meta = _describe_log_source(service)
    path = meta["path"]
    if not meta["exists"] or not path:
        return {
            "lines": [],
            "service": service,
            "exists": False,
            "path": path,
            "candidates": meta["candidates"],
        }
    try:
        result = subprocess.run(["tail", f"-{lines}", path],
                                capture_output=True, text=True, timeout=5)
        return {
            "lines": result.stdout.splitlines(),
            "service": service,
            "exists": True,
            "path": path,
            "candidates": meta["candidates"],
        }
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
    Trigger the daemon to run one complete pass immediately.
    
    This processes all unprocessed sessions (episodic entries, clustering,
    compaction, insights, and procedural patterns) in a background thread.
    Returns immediately without waiting for completion.
    """
    from memory.daemon import run  # noqa: PLC0415

    def _run_daemon_once():
        """Run daemon in background thread."""
        try:
            run(once=True)
        except Exception as exc:
            from memory.logger import error_log  # noqa: PLC0415
            error_log("dashboard_compress", "daemon_run_failed", error=str(exc))

    thread = threading.Thread(target=_run_daemon_once, daemon=True)
    thread.start()
    
    return {
        "ok": True,
        "message": "Daemon processing triggered. Check logs for details."
    }


# ── App assembly ──────────────────────────────────────────────────────────────

app = FastAPI(title="Agentic Memory Dashboard", version="1.0.0")


@app.on_event("startup")
def _bootstrap_database() -> None:
    """Create or migrate the DB once when the dashboard server starts."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = bootstrap_db(DB_PATH)
    conn.close()

app.include_router(ui_router)
app.include_router(memory_router)
app.include_router(ops_router)

if __name__ == "__main__":
    import logging
    import setproctitle
    setproctitle.setproctitle("AgenticMemoryDashboard")
    enable_debug("query")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
