from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from memory.db._utils import _json_loads, _utc_now


_PROJECT_CONTEXT_FIELDS = ("project_id", "repo_root", "cwd", "git_remote", "git_branch")


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _normalize_bool(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _project_context_values(project_context: dict | None) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    project_context = project_context if isinstance(project_context, dict) else {}
    return tuple(project_context.get(field) for field in _PROJECT_CONTEXT_FIELDS)


def _event_dict(row: sqlite3.Row | dict) -> dict:
    payload = dict(row)
    if "include_working_memory" in payload and payload["include_working_memory"] is not None:
        payload["include_working_memory"] = bool(payload["include_working_memory"])
    if "details" in payload:
        payload["details"] = _json_loads(payload.get("details"), None)
    return payload


def insert_recall_event(
    conn: sqlite3.Connection,
    *,
    request_id: str,
    event_type: str,
    session_id: str | None = None,
    agent: str | None = None,
    prompt_chars: int | None = None,
    prompt_tokens_estimate: int | None = None,
    include_working_memory: bool | None = None,
    project_context: dict | None = None,
    action: str | None = None,
    warnings_count: int | None = None,
    facts_count: int | None = None,
    episodic_count: int | None = None,
    procedural_count: int | None = None,
    session_memory_count: int | None = None,
    working_memory_count: int | None = None,
    answer_tokens_estimate: int | None = None,
    injection_tokens_estimate: int | None = None,
    recalled_context_tokens_estimate: int | None = None,
    compression_gain_tokens_estimate: int | None = None,
    tokens_saved_estimate: int | None = None,
    error_message: str | None = None,
    details: dict | list | None = None,
    created_at: str | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO telemetry_recall_events (
          id, request_id, event_type, created_at, session_id, agent,
          prompt_chars, prompt_tokens_estimate, include_working_memory,
          project_id, repo_root, cwd, git_remote, git_branch,
          action, warnings_count, facts_count, episodic_count, procedural_count,
          session_memory_count, working_memory_count,
          answer_tokens_estimate, injection_tokens_estimate,
          recalled_context_tokens_estimate, compression_gain_tokens_estimate,
          tokens_saved_estimate, error_message, details
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            request_id,
            event_type,
            created_at or _utc_now(),
            session_id,
            agent,
            prompt_chars,
            prompt_tokens_estimate,
            _normalize_bool(include_working_memory),
            *_project_context_values(project_context),
            action,
            warnings_count,
            facts_count,
            episodic_count,
            procedural_count,
            session_memory_count,
            working_memory_count,
            answer_tokens_estimate,
            injection_tokens_estimate,
            recalled_context_tokens_estimate,
            compression_gain_tokens_estimate,
            tokens_saved_estimate,
            error_message,
            _json_dumps(details),
        ),
    )
    conn.commit()
    return event_id


def list_recall_events(
    conn: sqlite3.Connection,
    *,
    request_id: str | None = None,
    event_type: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM telemetry_recall_events
        {where}
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_event_dict(row) for row in rows]


def insert_retrieval_lane_metric(
    conn: sqlite3.Connection,
    *,
    request_id: str,
    lane: str,
    candidate_count: int | None = None,
    filtered_count: int | None = None,
    selected_count: int | None = None,
    hit_count: int | None = None,
    duration_ms: float | None = None,
    details: dict | list | None = None,
    created_at: str | None = None,
) -> str:
    metric_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO telemetry_retrieval_lane_metrics (
          id, request_id, lane, created_at, candidate_count, filtered_count,
          selected_count, hit_count, duration_ms, details
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metric_id,
            request_id,
            lane,
            created_at or _utc_now(),
            candidate_count,
            filtered_count,
            selected_count,
            hit_count,
            duration_ms,
            _json_dumps(details),
        ),
    )
    conn.commit()
    return metric_id


def list_retrieval_lane_metrics(
    conn: sqlite3.Connection,
    *,
    request_id: str | None = None,
    lane: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if request_id:
        clauses.append("request_id = ?")
        params.append(request_id)
    if lane:
        clauses.append("lane = ?")
        params.append(lane)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM telemetry_retrieval_lane_metrics
        {where}
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_event_dict(row) for row in rows]


def insert_latency_breakdown(
    conn: sqlite3.Connection,
    *,
    component: str,
    operation: str,
    stage: str,
    duration_ms: float,
    request_id: str | None = None,
    session_id: str | None = None,
    details: dict | list | None = None,
    created_at: str | None = None,
) -> str:
    latency_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO telemetry_latency_breakdowns (
          id, request_id, created_at, component, operation, stage,
          duration_ms, session_id, details
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            latency_id,
            request_id,
            created_at or _utc_now(),
            component,
            operation,
            stage,
            float(duration_ms),
            session_id,
            _json_dumps(details),
        ),
    )
    conn.commit()
    return latency_id


def list_recent_latency_breakdowns(
    conn: sqlite3.Connection,
    *,
    component: str | None = None,
    operation: str | None = None,
    stage: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if component:
        clauses.append("component = ?")
        params.append(component)
    if operation:
        clauses.append("operation = ?")
        params.append(operation)
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM telemetry_latency_breakdowns
        {where}
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_event_dict(row) for row in rows]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95)))
    return round(float(ordered[index]), 3)


def summarize_latency_breakdowns(
    conn: sqlite3.Connection,
    *,
    component: str | None = None,
    operation: str | None = None,
    limit: int = 500,
) -> dict[str, dict[str, float | int]]:
    rows = list_recent_latency_breakdowns(conn, component=component, operation=operation, limit=limit)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        stage = str(row.get("stage") or "")
        duration = row.get("duration_ms")
        if not stage or not isinstance(duration, (int, float)):
            continue
        grouped.setdefault(stage, []).append(round(float(duration), 3))

    summary: dict[str, dict[str, float | int]] = {}
    for stage, values in grouped.items():
        summary[stage] = {
            "count": len(values),
            "avg_ms": round(sum(values) / len(values), 3) if values else 0.0,
            "p95_ms": _p95(values),
        }
    return summary


def insert_system_stat(
    conn: sqlite3.Connection,
    *,
    component: str,
    metric_name: str,
    metric_value: float,
    unit: str | None = None,
    process_id: int | None = None,
    session_id: str | None = None,
    details: dict | list | None = None,
    created_at: str | None = None,
) -> str:
    stat_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO telemetry_system_stats (
          id, created_at, component, process_id, session_id,
          metric_name, metric_value, unit, details
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stat_id,
            created_at or _utc_now(),
            component,
            process_id,
            session_id,
            metric_name,
            float(metric_value),
            unit,
            _json_dumps(details),
        ),
    )
    conn.commit()
    return stat_id


def list_recent_system_stats(
    conn: sqlite3.Connection,
    *,
    component: str | None = None,
    metric_name: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if component:
        clauses.append("component = ?")
        params.append(component)
    if metric_name:
        clauses.append("metric_name = ?")
        params.append(metric_name)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM telemetry_system_stats
        {where}
        ORDER BY created_at DESC, rowid DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    return [_event_dict(row) for row in rows]


def get_latest_system_stats(conn: sqlite3.Connection, *, component: str | None = None) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in list_recent_system_stats(conn, component=component, limit=500):
        metric_name = row.get("metric_name")
        if isinstance(metric_name, str) and metric_name not in latest:
            latest[metric_name] = row
    return latest


def get_telemetry_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "recall_events": int(conn.execute("SELECT COUNT(*) AS c FROM telemetry_recall_events").fetchone()["c"]),
        "retrieval_lane_metrics": int(conn.execute("SELECT COUNT(*) AS c FROM telemetry_retrieval_lane_metrics").fetchone()["c"]),
        "latency_breakdowns": int(conn.execute("SELECT COUNT(*) AS c FROM telemetry_latency_breakdowns").fetchone()["c"]),
        "system_stats": int(conn.execute("SELECT COUNT(*) AS c FROM telemetry_system_stats").fetchone()["c"]),
    }
