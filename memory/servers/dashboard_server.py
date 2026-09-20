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


def _empty_activity_stats() -> dict[str, int | float]:
    return {
        "llm_calls_avoided": 0,
        "hard_tokens_saved": 0,
        "est_tokens_saved": 0,
        "context_recalls": 0,
        "context_tokens_recovered": 0,
        "injection_tokens": 0,
        "compression_gain_tokens": 0,
        "compression_ratio": 0.0,
    }


_ACTIVITY_STATS_CACHE = {
    "mtime": None,
    "size": None,
    "stats": _empty_activity_stats(),
}
_PERFORMANCE_STATS_CACHE = {
    "mtime": None,
    "size": None,
    "stats": {
        "recall_embedding_ms_recent": [],
        "memory_search_ms_recent": [],
        "daemon_session_ms_recent": [],
        "summary": {
            "recall_embedding_avg_ms": 0.0,
            "memory_search_avg_ms": 0.0,
            "daemon_session_avg_ms": 0.0,
            "recall_embedding_p95_ms": 0.0,
            "memory_search_p95_ms": 0.0,
            "daemon_session_p95_ms": 0.0,
        },
    },
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


def _parse_pid_lines(text: str) -> int | None:
    for line in text.splitlines():
        value = line.strip().rstrip(";")
        if not value:
            continue
        if '"PID"' in line and "=" in line:
            _, _, value = line.partition("=")
            value = value.strip().rstrip(";")
        try:
            return int(value)
        except ValueError:
            continue
    return None



def _check_daemon() -> tuple[bool, int | None]:
    """Return daemon status from launchctl, with a process-title fallback."""
    try:
        result = subprocess.run(
            ["launchctl", "list", "com.memory.daemon"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, _parse_pid_lines(result.stdout)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["pgrep", "-x", "AgenticMemoryDaemon"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, _parse_pid_lines(result.stdout)
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
    return os.path.join(PROJECT_ROOT, "memory", "servers", "ingest_server.py")


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


def _with_log_partitions(lines: list[str]) -> list[str]:
    if not lines:
        return []
    partitioned: list[str] = []
    for line in lines:
        partitioned.append(line)
        if line.strip() != "=" * 60:
            partitioned.append("=" * 60)
    return partitioned


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
        return _empty_activity_stats()

    try:
        mtime = os.path.getmtime(_ACTIVITY_LOG_PATH)
        size = os.path.getsize(_ACTIVITY_LOG_PATH)
    except OSError:
        return _empty_activity_stats()

    if _ACTIVITY_STATS_CACHE["mtime"] == mtime and _ACTIVITY_STATS_CACHE["size"] == size:
        return dict(_ACTIVITY_STATS_CACHE["stats"])

    stats = _empty_activity_stats()
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
                action = payload.get("action")
                if action == "memory_answer":
                    stats["llm_calls_avoided"] += 1
                    try:
                        saved = int(payload.get("tokens_saved_estimate") or 0)
                    except (TypeError, ValueError):
                        saved = 0
                    stats["hard_tokens_saved"] += saved
                    stats["est_tokens_saved"] += saved
                    continue
                if action == "memory_injection":
                    stats["context_recalls"] += 1
                    try:
                        stats["context_tokens_recovered"] += int(payload.get("recalled_context_tokens_estimate") or 0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        stats["injection_tokens"] += int(payload.get("injection_tokens_estimate") or 0)
                    except (TypeError, ValueError):
                        pass
                    try:
                        stats["compression_gain_tokens"] += int(payload.get("compression_gain_tokens_estimate") or 0)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        return _empty_activity_stats()

    if stats["injection_tokens"] > 0:
        stats["compression_ratio"] = round(stats["context_tokens_recovered"] / stats["injection_tokens"], 2)

    _ACTIVITY_STATS_CACHE["mtime"] = mtime
    _ACTIVITY_STATS_CACHE["size"] = size
    _ACTIVITY_STATS_CACHE["stats"] = dict(stats)
    return stats


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)



def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return round(float(ordered[index]), 3)



def _sample_label(timestamp: str | None, fallback: str) -> str:
    if not timestamp:
        return fallback
    if "T" not in timestamp:
        return timestamp[-8:]
    time_part = timestamp.split("T", 1)[1]
    time_part = time_part.replace("Z", "")
    return time_part[:8] or fallback



def _parse_performance_activity_log() -> dict:
    empty = {
        "recall_embedding_ms_recent": [],
        "memory_search_ms_recent": [],
        "daemon_session_ms_recent": [],
        "summary": {
            "recall_embedding_avg_ms": 0.0,
            "memory_search_avg_ms": 0.0,
            "daemon_session_avg_ms": 0.0,
            "recall_embedding_p95_ms": 0.0,
            "memory_search_p95_ms": 0.0,
            "daemon_session_p95_ms": 0.0,
        },
    }
    if not os.path.exists(_ACTIVITY_LOG_PATH):
        return empty

    try:
        mtime = os.path.getmtime(_ACTIVITY_LOG_PATH)
        size = os.path.getsize(_ACTIVITY_LOG_PATH)
    except OSError:
        return empty

    if _PERFORMANCE_STATS_CACHE["mtime"] == mtime and _PERFORMANCE_STATS_CACHE["size"] == size:
        return dict(_PERFORMANCE_STATS_CACHE["stats"])

    recall_embedding: list[dict] = []
    memory_search: list[dict] = []
    daemon_session: list[dict] = []

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
                action = payload.get("action")
                component = payload.get("component")
                timestamp = payload.get("timestamp")
                if component == "retrieval" and action == "recall_timing":
                    embed_ms = payload.get("prompt_embedding_ms")
                    search_ms = payload.get("memory_search_ms")
                    if isinstance(embed_ms, (int, float)):
                        recall_embedding.append({
                            "timestamp": timestamp,
                            "label": _sample_label(timestamp, str(len(recall_embedding) + 1)),
                            "value": round(float(embed_ms), 3),
                        })
                    if isinstance(search_ms, (int, float)):
                        memory_search.append({
                            "timestamp": timestamp,
                            "label": _sample_label(timestamp, str(len(memory_search) + 1)),
                            "value": round(float(search_ms), 3),
                        })
                if component == "daemon" and action == "session_timing":
                    duration_ms = payload.get("duration_ms")
                    if isinstance(duration_ms, (int, float)):
                        daemon_session.append({
                            "timestamp": timestamp,
                            "label": _sample_label(timestamp, str(len(daemon_session) + 1)),
                            "value": round(float(duration_ms), 3),
                            "session": payload.get("session"),
                        })
    except Exception:
        return empty

    recall_embedding = recall_embedding[-20:]
    memory_search = memory_search[-20:]
    daemon_session = daemon_session[-20:]

    stats = {
        "recall_embedding_ms_recent": recall_embedding,
        "memory_search_ms_recent": memory_search,
        "daemon_session_ms_recent": daemon_session,
        "summary": {
            "recall_embedding_avg_ms": _avg([item["value"] for item in recall_embedding]),
            "memory_search_avg_ms": _avg([item["value"] for item in memory_search]),
            "daemon_session_avg_ms": _avg([item["value"] for item in daemon_session]),
            "recall_embedding_p95_ms": _p95([item["value"] for item in recall_embedding]),
            "memory_search_p95_ms": _p95([item["value"] for item in memory_search]),
            "daemon_session_p95_ms": _p95([item["value"] for item in daemon_session]),
        },
    }

    _PERFORMANCE_STATS_CACHE["mtime"] = mtime
    _PERFORMANCE_STATS_CACHE["size"] = size
    _PERFORMANCE_STATS_CACHE["stats"] = dict(stats)
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
    efficiency = {**_empty_activity_stats(), **_parse_activity_log()}
    performance = _parse_performance_activity_log()

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
            "restart_command": f"{sys.executable} memory/servers/ingest_server.py",
            "restart_endpoint": "/ops/memory-search-engine/restart",
        },
        "logs": _list_logs(),
        "efficiency": {
            **efficiency,
            "metric": "hard_token_savings_and_context_compression",
            "method": "Direct answers count prompt_tokens + answer_tokens; injections estimate recalled context, injected context, and compression gain using a ~chars/4 heuristic.",
        },
        "performance": performance.get("summary", {}),
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
        performance = _parse_performance_activity_log()
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
            "recall_embedding_ms_recent": performance.get("recall_embedding_ms_recent", []),
            "memory_search_ms_recent": performance.get("memory_search_ms_recent", []),
            "daemon_session_ms_recent": performance.get("daemon_session_ms_recent", []),
            "performance_summary": performance.get("summary", {}),
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

    def _spawn_manually() -> tuple[int | None, bool]:
        try:
            subprocess.run(["pkill", "-f", script], capture_output=True, text=True, timeout=5)
        except Exception:
            pass
        time.sleep(0.25)
        ingest_log_path = os.path.expanduser("~/.memory/ingest.log")
        os.makedirs(os.path.dirname(ingest_log_path), exist_ok=True)
        with open(ingest_log_path, "a") as ingest_log:
            proc = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=ingest_log,
                stderr=ingest_log,
                start_new_session=True,
            )
        return proc.pid, True

    try:
        should_spawn_manually = True
        if os.path.exists(plist):
            label = "com.memory.ingest"
            result = subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            should_spawn_manually = result.returncode != 0
        if should_spawn_manually:
            pid, started_here = _spawn_manually()
    except Exception as exc:
        status = _check_recall_server()
        return {
            "ok": False,
            "started_here": started_here,
            "pid": pid,
            "error": str(exc),
            **status,
            "restart_command": f"{sys.executable} memory/servers/ingest_server.py",
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
                "restart_command": f"{sys.executable} memory/servers/ingest_server.py",
            }

    if os.path.exists(plist) and not started_here:
        try:
            pid, started_here = _spawn_manually()
        except Exception as exc:
            status = _check_recall_server()
            return {
                "ok": False,
                "started_here": started_here,
                "pid": pid,
                "error": str(exc),
                **status,
                "restart_command": f"{sys.executable} memory/servers/ingest_server.py",
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
                    "restart_command": f"{sys.executable} memory/servers/ingest_server.py",
                }

    status = _check_recall_server()
    return {
        "ok": bool(status.get("running")),
        "started_here": started_here,
        "pid": pid,
        **status,
        "restart_command": f"{sys.executable} memory/servers/ingest_server.py",
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
            "lines": _with_log_partitions(result.stdout.splitlines()),
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
