# dashboard_server.py — simplified dashboard API for sessions, facts, episodes, working memory, session memory, and logs.

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

from memory.db import bootstrap_db, open_db
from memory.facts.text import build_canonical_fact_content
from memory.utils.debug import enable_debug
from memory.llm.ollama import is_ollama_running, start_ollama_if_needed


DB_PATH = os.path.expanduser("~/.memory/memory.db")
DAEMON_LOG = os.path.expanduser("~/.memory/daemon.log")
PORT = int(os.environ.get("MEMORY_QUERY_PORT", "7748"))
RECALL_PORT = int(os.environ.get("MEMORY_INGEST_PORT", "7747"))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECALL_SERVER_URL = f"http://127.0.0.1:{RECALL_PORT}"
MEMORY_API_LABEL = "Memory Search Engine"
_ACTIVITY_LOG_PATH = os.path.expanduser("~/.memory/activity.log")
_ACTIVITY_STATS_CACHE = {
    "mtime": None,
    "size": None,
    "stats": {"llm_calls_avoided": 0, "est_tokens_saved": 0},
}

_LOG_SOURCES: dict[str, dict[str, object]] = {
    "daemon": {
        "label": "Daemon",
        "description": "Background extraction log",
        "paths": [os.path.expanduser("~/.memory/daemon.log")],
    },
    "memory_search_engine": {
        "label": "Memory Search Engine",
        "description": "Memory search engine log",
        "paths": [os.path.expanduser("~/.memory/ingest.log")],
    },
    "wake_up": {
        "label": "Wake Up",
        "description": "Claude wake-up recall log",
        "paths": [os.path.expanduser("~/.memory/wake_up.log")],
    },
    "save_hook": {
        "label": "Save Hook",
        "description": "Claude save-hook ingest log",
        "paths": [os.path.expanduser("~/.memory/save_hook.log")],
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
    "procedural": {
        "label": "Procedural",
        "description": "Procedural extraction log",
        "paths": [os.path.expanduser("~/.memory/procedural.log")],
    },
    "working_memory": {
        "label": "Working Memory",
        "description": "Working-memory extraction and retrieval log",
        "paths": [os.path.expanduser("~/.memory/working_memory.log")],
    },
    "session_memory": {
        "label": "Session Memory",
        "description": "Session-memory extraction and retrieval log",
        "paths": [os.path.expanduser("~/.memory/session_memory.log")],
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


def _check_ollama() -> tuple[bool, str | None, list[str]]:
    active_model: str | None = None
    installed_models: list[str] = []

    try:
        with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=2) as response:
            if response.status == 200:
                models = json.loads(response.read()).get("models", [])
                active_model = models[0].get("name") if models else None
    except Exception:
        pass

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as response:
            if response.status == 200:
                models = json.loads(response.read()).get("models", [])
                installed_models = [model.get("name") for model in models if model.get("name")]
    except Exception:
        pass

    running = is_ollama_running()
    if running and active_model is None and installed_models:
        active_model = installed_models[0]
    return running, active_model, installed_models


def _recall_server_script() -> str:
    return os.path.join(PROJECT_ROOT, "memory", "ingest_server.py")


def _check_recall_server() -> dict:
    try:
        with urllib.request.urlopen(f"{RECALL_SERVER_URL}/status", timeout=2) as response:
            if response.status != 200:
                return {
                    "running": False,
                    "status": "unreachable",
                    "port": RECALL_PORT,
                    "embed_model_ready": False,
                    "embed_model_name": None,
                    "embed_model_error": f"HTTP {response.status}",
                }
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except Exception as exc:
        return {
            "running": False,
            "status": "stopped",
            "port": RECALL_PORT,
            "embed_model_ready": False,
            "embed_model_name": None,
            "embed_model_error": str(exc),
        }

    embed_ready = bool(payload.get("embed_model_ready", False))
    embed_name = payload.get("embed_model_name")
    embed_error = payload.get("embed_model_error") or ""
    status = "running" if embed_ready else "degraded"
    return {
        "running": True,
        "status": status,
        "port": RECALL_PORT,
        "embed_model_ready": embed_ready,
        "embed_model_name": embed_name,
        "embed_model_error": embed_error,
        "db_path": payload.get("db_path"),
        "db_exists": payload.get("db_exists"),
    }


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


def _parse_activity_log() -> dict:
    if not os.path.exists(_ACTIVITY_LOG_PATH):
        return {"llm_calls_avoided": 0, "est_tokens_saved": 0}

    try:
        mtime = os.path.getmtime(_ACTIVITY_LOG_PATH)
        size = os.path.getsize(_ACTIVITY_LOG_PATH)
    except OSError:
        return {"llm_calls_avoided": 0, "est_tokens_saved": 0}

    if _ACTIVITY_STATS_CACHE["mtime"] == mtime and _ACTIVITY_STATS_CACHE["size"] == size:
        return dict(_ACTIVITY_STATS_CACHE["stats"])

    stats = {"llm_calls_avoided": 0, "est_tokens_saved": 0}
    try:
        with open(_ACTIVITY_LOG_PATH) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if payload.get("action") != "memory_answer":
                    continue
                stats["llm_calls_avoided"] += 1
                try:
                    stats["est_tokens_saved"] += int(payload.get("tokens_saved_estimate") or 0)
                except (TypeError, ValueError):
                    pass
    except Exception:
        return {"llm_calls_avoided": 0, "est_tokens_saved": 0}

    _ACTIVITY_STATS_CACHE["mtime"] = mtime
    _ACTIVITY_STATS_CACHE["size"] = size
    _ACTIVITY_STATS_CACHE["stats"] = dict(stats)
    return stats


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
    ollama_running, ollama_model, ollama_installed_models = _check_ollama()
    memory_search_engine = _check_recall_server()
    last_run_iso, facts_extracted_total = _parse_daemon_log()
    efficiency = _parse_activity_log()

    total_sessions = 0
    total_facts = 0
    total_episodes = 0
    total_procedures = 0
    total_working_memory = 0
    total_session_memory = 0
    unprocessed_sessions = 0

    try:
        conn = open_db(DB_PATH)
        try:
            total_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
            total_facts = conn.execute("SELECT COUNT(*) AS c FROM facts").fetchone()["c"]
            total_episodes = conn.execute("SELECT COUNT(*) AS c FROM episodic_memory").fetchone()["c"]
            total_procedures = conn.execute("SELECT COUNT(*) AS c FROM procedural_memory").fetchone()["c"]
            total_working_memory = conn.execute("SELECT COUNT(*) AS c FROM working_memory").fetchone()["c"]
            total_session_memory = conn.execute("SELECT COUNT(*) AS c FROM session_memory").fetchone()["c"]
            unprocessed_sessions = conn.execute("SELECT COUNT(*) AS c FROM sessions WHERE daemon_processed_at IS NULL").fetchone()["c"]
        finally:
            conn.close()
    except Exception:
        pass

    default_model = os.environ.get("MEMORY_OLLAMA_MODEL", "qwen2.5:7b")
    only_small_model_installed = ollama_installed_models == ["qwen2.5:3b"]

    return {
        "daemon": {
            "running": daemon_running,
            "pid": daemon_pid,
            "last_run": last_run_iso,
            "facts_extracted": facts_extracted_total,
            "unprocessed_sessions": unprocessed_sessions,
            "process_one_endpoint": "/ops/daemon/process-one",
        },
        "ollama": {
            "running": ollama_running,
            "model": ollama_model,
            "installed_models": ollama_installed_models,
            "default_model": default_model,
            "only_small_model_installed": only_small_model_installed,
            "warning": "Only qwen2.5:3b is installed; consider pulling qwen2.5:7b for stronger extraction and rendering." if only_small_model_installed else None,
            "pull_recommendation": "ollama pull qwen2.5:7b",
            "start_command": "ollama serve",
            "start_endpoint": "/ops/ollama/start",
        },
        "memory_search_engine": {
            **memory_search_engine,
            "display_name": MEMORY_API_LABEL,
            "restart_command": f"{sys.executable} memory/ingest_server.py",
            "restart_endpoint": "/ops/memory-search-engine/restart",
        },
        "logs": _list_logs(),
        "efficiency": {
            **efficiency,
            "metric": "estimated_tokens_saved_via_memory_answers",
            "method": "prompt_tokens + answer_tokens using ~chars/4 heuristic for direct memory answers",
        },
        "memory": {
            "total_sessions": total_sessions,
            "total_facts": total_facts,
            "total_episodes": total_episodes,
            "total_procedures": total_procedures,
            "total_working_memory": total_working_memory,
            "total_session_memory": total_session_memory,
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
                "SELECT id, semantic_content, entity, attribute, value, tags, source, session_id, created_at, updated_at FROM facts ORDER BY created_at DESC"
            ).fetchall():
                try:
                    tags = json.loads(row["tags"]) if row["tags"] else []
                except Exception:
                    tags = []
                rows.append(
                    {
                        "id": row["id"],
                        "content": build_canonical_fact_content(row["entity"], row["attribute"], row["value"]),
                        "semantic_content": row["semantic_content"],
                        "entity": row["entity"],
                        "attribute": row["attribute"],
                        "value": row["value"],
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


@memory_router.get("/procedural")
def get_memory_procedural() -> dict:
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for row in conn.execute(
                "SELECT id, session_id, title, summary, updated_at, details FROM procedural_memory ORDER BY updated_at DESC"
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
                        "summary": row["summary"],
                        "updated_at": row["updated_at"],
                        "steps": details.get("steps", []),
                        "tools": details.get("tools", []),
                        "trigger_phrases": details.get("trigger_phrases", []),
                        "source": details.get("source", "procedural_extractor"),
                    }
                )
        finally:
            conn.close()
    except Exception:
        pass
    return {"procedural": rows, "total": len(rows)}


@memory_router.get("/working-memory")
@memory_router.get("/working")
def get_memory_working_memory() -> dict:
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for row in conn.execute(
                "SELECT id, session_id, current_goal, current_focus, next_step, status, updated_at, details FROM working_memory ORDER BY updated_at DESC"
            ).fetchall():
                try:
                    details = json.loads(row["details"] or "{}")
                except Exception:
                    details = {}
                rows.append(
                    {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "current_goal": row["current_goal"],
                        "current_focus": row["current_focus"],
                        "next_step": row["next_step"],
                        "status": row["status"],
                        "updated_at": row["updated_at"],
                        "active_tasks": details.get("active_tasks", []),
                        "constraints": details.get("constraints", []),
                        "confidence": details.get("confidence"),
                        "source_quote": details.get("source_quote"),
                        "source": details.get("source", "working_memory_extractor"),
                    }
                )
        finally:
            conn.close()
    except Exception:
        pass
    return {"working_memory": rows, "total": len(rows)}


@memory_router.get("/session-memory")
@memory_router.get("/session-memory/compacted")
@memory_router.get("/sessions/compacted")
def get_memory_session_memory() -> dict:
    rows: list[dict] = []
    try:
        conn = open_db(DB_PATH)
        try:
            for row in conn.execute(
                "SELECT id, session_id, title, summary, left_off_at, updated_at, details FROM session_memory ORDER BY updated_at DESC"
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
                        "summary": row["summary"],
                        "left_off_at": row["left_off_at"],
                        "updated_at": row["updated_at"],
                        "what_was_tried": details.get("what_was_tried", []),
                        "outcomes": details.get("outcomes", []),
                        "next_steps": details.get("next_steps", []),
                        "confidence": details.get("confidence"),
                        "source_quote": details.get("source_quote"),
                        "source": details.get("source", "session_memory_extractor"),
                    }
                )
        finally:
            conn.close()
    except Exception:
        pass
    return {"session_memory": rows, "total": len(rows)}


@ops_router.post("/ops/ollama/start")
def start_ollama() -> dict:
    proc = start_ollama_if_needed()
    running, model, installed_models = _check_ollama()
    return {
        "ok": running,
        "running": running,
        "model": model,
        "installed_models": installed_models,
        "default_model": os.environ.get("MEMORY_OLLAMA_MODEL", "qwen2.5:7b"),
        "started_here": proc is not None,
        "start_command": "ollama serve",
    }


@ops_router.post("/ops/daemon/process-one")
def process_one_daemon_session() -> dict:
    from memory.daemon import process_one_unprocessed_session

    result = process_one_unprocessed_session()
    if result.get("processed") or result.get("ok"):
        try:
            conn = open_db(DB_PATH)
            try:
                result["unprocessed_sessions"] = conn.execute(
                    "SELECT COUNT(*) AS c FROM sessions WHERE daemon_processed_at IS NULL"
                ).fetchone()["c"]
            finally:
                conn.close()
        except Exception:
            result["unprocessed_sessions"] = None
    return result


@ops_router.post("/ops/memory-search-engine/restart")
def restart_recall_server() -> dict:
    script = _recall_server_script()
    command = [sys.executable, script]
    plist = os.path.expanduser("~/Library/LaunchAgents/com.memory.ingest.plist")
    pid = None
    started_here = False

    try:
        if os.path.exists(plist):
            label = "com.memory.ingest"
            subprocess.run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"], capture_output=True, text=True, timeout=5)
        else:
            try:
                subprocess.run(["pkill", "-f", script], capture_output=True, text=True, timeout=5)
            except Exception:
                pass
            time.sleep(0.25)
            proc = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            pid = proc.pid
            started_here = True
    except Exception as exc:
        status = _check_recall_server()
        return {
            "ok": False,
            "started_here": started_here,
            "pid": pid,
            "error": str(exc),
            **status,
            "restart_command": f"{sys.executable} memory/ingest_server.py",
        }

    for _ in range(32):
        time.sleep(0.25)
        status = _check_recall_server()
        if status.get("running"):
            return {
                "ok": True,
                "started_here": started_here,
                "pid": pid,
                **status,
                "restart_command": f"{sys.executable} memory/ingest_server.py",
            }

    status = _check_recall_server()
    return {
        "ok": bool(status.get("running")),
        "started_here": started_here,
        "pid": pid,
        **status,
        "restart_command": f"{sys.executable} memory/ingest_server.py",
    }


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
