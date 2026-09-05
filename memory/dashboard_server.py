# dashboard_server.py — simplified dashboard API for sessions, facts, episodes, and logs.

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

from memory.db import bootstrap_db, open_db
from memory.debug import enable_debug


DB_PATH = os.path.expanduser("~/.memory/memory.db")
DAEMON_LOG = os.path.expanduser("~/.memory/daemon.log")
PORT = int(os.environ.get("MEMORY_QUERY_PORT", "7748"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_LOG_SOURCES: dict[str, dict[str, object]] = {
    "daemon": {
        "label": "Daemon",
        "description": "Background extraction log",
        "paths": [os.path.expanduser("~/.memory/daemon.log")],
    },
    "facts": {
        "label": "Facts",
        "description": "Fact extraction log",
        "paths": [os.path.expanduser("~/.memory/facts.log")],
    },
    "episodic": {
        "label": "Episodes",
        "description": "Episode extraction and retrieval log",
        "paths": [os.path.expanduser("~/.memory/episodic.log")],
    },
}

ui_router = APIRouter()
memory_router = APIRouter(prefix="/memory")
ops_router = APIRouter()


def _check_daemon() -> tuple[bool, int | None]:
    """Ask launchctl if com.memory.daemon is running. Returns (running, pid)."""
    try:
        result = subprocess.run(
            ["launchctl", "list", "com.memory.daemon"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if '"PID"' in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        return True, int(parts[1].strip().rstrip(";"))
            return True, None
    except Exception:
        pass
    return False, None


def _check_ollama() -> tuple[bool, str | None]:
    for path in ("/api/ps", "/api/tags"):
        try:
            with urllib.request.urlopen(f"http://localhost:11434{path}", timeout=2) as response:
                if response.status == 200:
                    models = json.loads(response.read()).get("models", [])
                    model = models[0].get("name") if models else None
                    return True, model
        except Exception:
            pass
    return False, None


def _resolve_log_path(name: str) -> str | None:
    config = _LOG_SOURCES.get(name) or {}
    for candidate in config.get("paths", []):
        if candidate and os.path.exists(str(candidate)):
            return str(candidate)
    return None


def _describe_log_source(name: str) -> dict:
    config = _LOG_SOURCES[name]
    resolved = _resolve_log_path(name)
    candidates = [str(path) for path in config.get("paths", []) if path]
    path = resolved or (candidates[0] if candidates else None)
    exists = bool(resolved)
    size_bytes = os.path.getsize(resolved) if resolved and os.path.exists(resolved) else 0
    return {
        "name": name,
        "label": config["label"],
        "description": config["description"],
        "path": path,
        "exists": exists,
        "size_bytes": size_bytes,
        "candidates": candidates,
    }


def _list_logs() -> list[dict]:
    return [_describe_log_source(name) for name in _LOG_SOURCES]


def _parse_daemon_log() -> tuple[str | None, int]:
    last_run_iso, facts_total = None, 0
    if not os.path.exists(DAEMON_LOG):
        return last_run_iso, facts_total
    try:
        with open(DAEMON_LOG) as handle:
            for line in handle:
                line = line.strip()
                if "extracted" in line and "facts" in line:
                    parts = line.split()
                    for index, part in enumerate(parts):
                        if part == "extracted" and index + 1 < len(parts):
                            try:
                                facts_total += int(parts[index + 1])
                                last_run_iso = parts[0]
                            except ValueError:
                                pass
    except Exception:
        pass
    return last_run_iso, facts_total


@ui_router.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    path = os.path.join(PROJECT_ROOT, "dashboard.html")
    try:
        with open(path) as handle:
            html = handle.read()
    except FileNotFoundError:
        html = f"<h1>dashboard.html not found</h1><p>Expected at: {path}</p>"
    return HTMLResponse(content=html)


@ui_router.get("/services")
def get_services() -> dict:
    daemon_running, daemon_pid = _check_daemon()
    ollama_running, ollama_model = _check_ollama()
    last_run_iso, facts_extracted_total = _parse_daemon_log()

    total_sessions = 0
    total_facts = 0
    total_episodes = 0

    try:
        conn = open_db(DB_PATH)
        try:
            total_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
            total_facts = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
            total_episodes = conn.execute("SELECT COUNT(*) AS c FROM episodic_memory").fetchone()["c"]
        finally:
            conn.close()
    except Exception:
        pass

    return {
        "daemon": {
            "running": daemon_running,
            "pid": daemon_pid,
            "last_run": last_run_iso,
            "facts_extracted": facts_extracted_total,
        },
        "ollama": {
            "running": ollama_running,
            "model": ollama_model,
        },
        "logs": _list_logs(),
        "memory": {
            "total_sessions": total_sessions,
            "total_facts": total_facts,
            "total_episodes": total_episodes,
            "db_size_bytes": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
        },
    }


@ui_router.get("/status")
def get_status() -> dict:
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
    try:
        conn = open_db(DB_PATH)
        try:
            sessions_by_day = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT DATE(started_at) AS day, COUNT(*) AS count
                    FROM sessions
                    WHERE started_at >= DATE('now', '-14 days')
                    GROUP BY day
                    ORDER BY day
                    """
                ).fetchall()
            ]
            facts_by_day = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT DATE(created_at) AS day, COUNT(*) AS count
                    FROM facts
                    WHERE created_at >= DATE('now', '-14 days')
                    GROUP BY day
                    ORDER BY day
                    """
                ).fetchall()
            ]
            episodes_by_day = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT DATE(happened_at) AS day, COUNT(*) AS count
                    FROM episodic_memory
                    WHERE happened_at >= DATE('now', '-14 days')
                    GROUP BY day
                    ORDER BY day
                    """
                ).fetchall()
            ]
        finally:
            conn.close()
        return {
            "sessions_by_day": sessions_by_day,
            "facts_by_day": facts_by_day,
            "episodes_by_day": episodes_by_day,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@memory_router.get("/sessions")
def get_memory_sessions() -> dict:
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for row in conn.execute(
                "SELECT session_id, agent, turn_count, started_at, updated_at, transcript FROM sessions ORDER BY updated_at DESC"
            ).fetchall():
                preview = ""
                try:
                    turns = json.loads(row["transcript"] or "[]")
                    preview = next(
                        (
                            str(turn.get("content", "")).replace("\n", " ")[:200]
                            for turn in turns
                            if turn.get("role") == "user"
                        ),
                        "",
                    )
                except Exception:
                    pass
                rows.append(
                    {
                        "session_id": row["session_id"],
                        "agent": row["agent"],
                        "turn_count": row["turn_count"],
                        "started_at": row["started_at"],
                        "updated_at": row["updated_at"],
                        "preview": preview,
                    }
                )
        finally:
            conn.close()
    except Exception:
        pass
    return {"sessions": rows, "total": len(rows)}


@memory_router.get("/sessions/{session_id}/transcript")
def get_session_transcript(session_id: str) -> dict:
    try:
        conn = open_db(DB_PATH)
        try:
            row = conn.execute(
                "SELECT session_id, agent, turn_count, started_at, updated_at, transcript FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return {"error": "session not found"}
            try:
                transcript = json.loads(row["transcript"] or "[]")
            except Exception:
                transcript = []
            return {
                "session_id": row["session_id"],
                "agent": row["agent"],
                "turn_count": row["turn_count"],
                "started_at": row["started_at"],
                "updated_at": row["updated_at"],
                "transcript": transcript,
            }
        finally:
            conn.close()
    except Exception as exc:
        return {"error": str(exc)}


@memory_router.get("/facts")
def get_memory_facts() -> dict:
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for row in conn.execute(
                "SELECT id, content, tags, source, session_id, created_at, updated_at FROM facts ORDER BY created_at DESC"
            ).fetchall():
                try:
                    tags = json.loads(row["tags"]) if row["tags"] else []
                except Exception:
                    tags = []
                rows.append(
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "tags": tags,
                        "source": row["source"],
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                )
        finally:
            conn.close()
    except Exception:
        pass
    return {"facts": rows, "total": len(rows)}


@memory_router.get("/episodic")
@memory_router.get("/episodes")
def get_memory_episodes() -> dict:
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for row in conn.execute(
                "SELECT id, session_id, title, abstract, happened_at, details FROM episodic_memory ORDER BY happened_at DESC"
            ).fetchall():
                try:
                    details = json.loads(row["details"] or "{}")
                except Exception:
                    details = {}
                rows.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "title": row["title"],
                        "abstract": row["abstract"],
                        "happened_at": row["happened_at"],
                        "participants": details.get("participants", []),
                        "decisions": details.get("decisions", []),
                        "outcomes": details.get("outcomes", []),
                        "follow_ups": details.get("follow_ups", []),
                        "confidence": details.get("confidence"),
                        "source_quote": details.get("source_quote"),
                        "source": details.get("source", "episodic_extractor"),
                    }
                )
        finally:
            conn.close()
    except Exception:
        pass
    return {"episodes": rows, "episodic": rows, "total": len(rows)}


@ops_router.get("/logs/{service}")
def get_logs(service: str, lines: int = 200) -> dict:
    if service not in _LOG_SOURCES:
        raise HTTPException(status_code=404, detail=f"Unknown log: {service}")

    meta = _describe_log_source(service)
    path = meta["path"]
    if not meta["exists"] or not path:
        return {
            "service": service,
            "exists": False,
            "path": path,
            "candidates": meta["candidates"],
            "lines": [],
        }

    try:
        result = subprocess.run(
            ["tail", f"-{lines}", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return {
            "service": service,
            "exists": True,
            "path": path,
            "candidates": meta["candidates"],
            "lines": result.stdout.splitlines(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = bootstrap_db(DB_PATH)
    conn.close()
    yield


app = FastAPI(title="Agentic Memory Dashboard", version="2.0.0", lifespan=_lifespan)


app.include_router(ui_router)
app.include_router(memory_router)
app.include_router(ops_router)


if __name__ == "__main__":
    enable_debug("query")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
