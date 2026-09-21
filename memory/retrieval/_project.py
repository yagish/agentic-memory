from __future__ import annotations

import os
import sqlite3

from memory.db.sessions import _normalize_project_context
from memory.facts.scope import classify_fact_scope_from_mapping

_PROJECT_CONTEXT_FIELDS = ("project_id", "repo_root", "cwd", "git_remote", "git_branch")
_SAME_PROJECT_ONLY_KINDS = {"working_memory", "session_memory", "episodic"}


def normalize_project_context(project_context: dict | None = None) -> dict[str, str]:
    return _normalize_project_context(project_context, None)


def _nested_workspace_paths(left: str, right: str) -> bool:
    left = os.path.abspath(os.path.expanduser(left))
    right = os.path.abspath(os.path.expanduser(right))
    if left == right:
        return True
    left_prefix = left.rstrip(os.sep) + os.sep
    right_prefix = right.rstrip(os.sep) + os.sep
    return left.startswith(right_prefix) or right.startswith(left_prefix)


def project_match_details(ambient_project: dict | None, candidate_project: dict | None) -> tuple[bool, float, str | None]:
    ambient = normalize_project_context(ambient_project)
    candidate = normalize_project_context(candidate_project)
    if not ambient or not candidate:
        return False, 0.0, None

    if ambient.get("project_id") and candidate.get("project_id"):
        matched = ambient["project_id"] == candidate["project_id"]
        return matched, (1.0 if matched else 0.0), ("project_id" if matched else None)

    if ambient.get("git_remote") and candidate.get("git_remote"):
        matched = ambient["git_remote"] == candidate["git_remote"]
        return matched, (1.0 if matched else 0.0), ("git_remote" if matched else None)

    if ambient.get("repo_root") and candidate.get("repo_root"):
        matched = ambient["repo_root"] == candidate["repo_root"]
        return matched, (0.9 if matched else 0.0), ("repo_root" if matched else None)

    if ambient.get("cwd") and candidate.get("cwd"):
        matched = _nested_workspace_paths(ambient["cwd"], candidate["cwd"])
        return matched, (0.75 if matched else 0.0), ("cwd" if matched else None)

    return False, 0.0, None


def _coerce_row_mapping(row) -> dict | None:
    if isinstance(row, dict):
        return row
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return None


def resolve_ambient_project_context(
    conn,
    *,
    project_context: dict | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    normalized = normalize_project_context(project_context)
    if normalized or not session_id:
        return normalized
    try:
        row = conn.execute(
            "SELECT project_id, repo_root, cwd, git_remote, git_branch FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    except Exception:
        return normalized
    mapping = _coerce_row_mapping(row)
    if not mapping:
        return normalized
    return normalize_project_context({field: mapping.get(field) for field in _PROJECT_CONTEXT_FIELDS})


def load_session_project_contexts(
    conn,
    *,
    session_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, str]]:
    params: tuple = ()
    sql = "SELECT session_id, project_id, repo_root, cwd, git_remote, git_branch FROM sessions"
    if session_ids is not None:
        unique_ids = tuple(sorted({session_id for session_id in session_ids if session_id}))
        if not unique_ids:
            return {}
        placeholders = ", ".join("?" for _ in unique_ids)
        sql += f" WHERE session_id IN ({placeholders})"
        params = unique_ids
    rows = conn.execute(sql, params).fetchall()
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        mapping = _coerce_row_mapping(row)
        if not mapping:
            continue
        normalized = normalize_project_context({field: mapping.get(field) for field in _PROJECT_CONTEXT_FIELDS})
        result[mapping["session_id"]] = normalized
    return result


def same_project_session_ids(conn, ambient_project: dict | None) -> set[str] | None:
    ambient = normalize_project_context(ambient_project)
    if not ambient:
        return None
    matches: set[str] = set()
    for session_id, session_project in load_session_project_contexts(conn).items():
        matched, _, _ = project_match_details(ambient, session_project)
        if matched:
            matches.add(session_id)
    return matches


def enrich_rows_with_project_metadata(
    rows: list[dict],
    *,
    ambient_project: dict | None = None,
    session_projects: dict[str, dict[str, str]] | None = None,
    kind: str,
) -> list[dict]:
    ambient = normalize_project_context(ambient_project)
    session_projects = session_projects or {}
    enriched: list[dict] = []
    for row in rows:
        item = dict(row)
        session_id = item.get("session_id")
        session_project = session_projects.get(session_id, {}) if session_id else {}
        for field in _PROJECT_CONTEXT_FIELDS:
            if session_project.get(field) and not item.get(field):
                item[field] = session_project[field]
        if ambient and session_project:
            same_project, match_score, match_basis = project_match_details(ambient, session_project)
            item["same_project"] = same_project
            item["project_match_score"] = match_score
            item["project_match_basis"] = match_basis
        else:
            item.setdefault("same_project", None)
            item.setdefault("project_match_score", 0.0)
            item.setdefault("project_match_basis", None)
        if kind == "facts":
            item["fact_scope"] = classify_fact_scope_from_mapping(item)
        enriched.append(item)
    return enriched


def is_row_allowed_for_project_policy(row: dict, kind: str, ambient_project: dict | None) -> bool:
    ambient = normalize_project_context(ambient_project)
    if kind == "facts":
        scope = row.get("fact_scope") or classify_fact_scope_from_mapping(row)
        if scope == "global":
            return True
        if not ambient:
            return False
        return bool(row.get("same_project"))

    if kind in _SAME_PROJECT_ONLY_KINDS:
        if not ambient:
            return True
        return bool(row.get("same_project"))

    return True
