"""Lightweight offline retrieval eval harness.

This module is intentionally test-local for now. It provides a deterministic,
fully offline way to seed labeled retrieval corpora into an in-memory SQLite DB,
run current retrieval behavior against labeled prompts, and assert expectations.

Fixture format
--------------
JSON object with:
- schema_version: integer (currently 1)
- embedding_space:
  - type: "anchor_terms"
  - anchors: ["token", ...]
- sessions: list of seeded sessions. Each session may include:
  - session_id, agent, started_at, updated_at
  - project_context: {project_id, repo_root, cwd, git_remote, git_branch}
  - turns: transcript turns (optional)
  - working_memory: object
  - session_memory: object
  - episodic: [object, ...]
  - procedural: [object, ...]
  - facts: [object, ...]

Memory rows can include either:
- embedding_terms: ["anchor", ...]
- embedding_text: "free text"
If neither is present, the harness falls back to a deterministic text derived from
that memory row's main fields.

Cases:
- id
- prompt
- include_working_memory
- session_id (optional)
- project_context (optional)
- expect:
  - action or allowed_actions
  - min_counts: {facts, episodic, procedural, session_memory, working_memory}
  - required_memory_types: ["procedural", ...]
  - required_session_ids / forbidden_session_ids
  - allowed_project_ids / forbidden_project_ids
  - max_wrong_project_hits
  - answer_must_contain / injection_must_contain
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any

from integrations.common import decide_prompt_memory_action, retrieve_prompt_memory
from memory.db import (
    init_db,
    insert_episodic,
    insert_fact,
    insert_procedural,
    pack_vector,
    upsert_session,
    upsert_session_memory,
    upsert_working_memory,
)


MEMORY_TYPES = ("facts", "episodic", "procedural", "session_memory", "working_memory")
DEFAULT_SUITE_PATH = Path(__file__).parent / "fixtures" / "retrieval_eval_cases.json"
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class EvalHit:
    memory_type: str
    session_id: str | None
    project_id: str | None
    label: str


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    passed: bool
    action: str
    counts: dict[str, int]
    projects_hit: list[str]
    session_ids_hit: list[str]
    wrong_project_hits: int
    errors: list[str] = field(default_factory=list)
    answer: str = ""
    injection: str = ""


@dataclass(frozen=True)
class EvalSuiteResult:
    total: int
    passed: int
    failed: int
    cases: list[EvalCaseResult]


def load_eval_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    suite_path = Path(path)
    data = json.loads(suite_path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported eval suite schema_version: {data.get('schema_version')!r}")
    embedding_space = data.get("embedding_space") or {}
    if embedding_space.get("type") != "anchor_terms":
        raise ValueError("Eval suite embedding_space.type must be 'anchor_terms'")
    if not isinstance(embedding_space.get("anchors"), list):
        raise ValueError("Eval suite embedding_space.anchors must be a list")
    if not isinstance(data.get("sessions"), list):
        raise ValueError("Eval suite sessions must be a list")
    if not isinstance(data.get("cases"), list):
        raise ValueError("Eval suite cases must be a list")
    return data


def build_anchor_embed_fn(anchors: list[str]):
    normalized = [str(anchor).strip().lower() for anchor in anchors if str(anchor).strip()]
    index = {anchor: position for position, anchor in enumerate(normalized)}

    def embed_fn(text: str) -> list[float]:
        vector = [0.0] * len(normalized)
        for token in _TOKEN_RE.findall((text or "").lower()):
            position = index.get(token)
            if position is not None:
                vector[position] += 1.0
        return vector

    return embed_fn


def _memory_embedding_vector(memory: dict[str, Any], embed_fn, fallback_text: str) -> list[float]:
    if isinstance(memory.get("embedding_terms"), list):
        return embed_fn(" ".join(str(item) for item in memory["embedding_terms"]))
    if isinstance(memory.get("embedding_text"), str):
        return embed_fn(memory["embedding_text"])
    return embed_fn(fallback_text)


def _seed_fact(conn, *, session_id: str, fact: dict[str, Any], embed_fn) -> None:
    semantic_content = fact.get("semantic_content") or (
        f"{fact['entity']} {fact['attribute']} {fact['value']}"
    )
    fact_id = insert_fact(
        conn,
        entity=fact["entity"],
        attribute=fact["attribute"],
        value=fact["value"],
        semantic_content=semantic_content,
        tags=fact.get("tags") or [],
        source="offline_eval",
        session_id=session_id,
    )
    embedding = _memory_embedding_vector(fact, embed_fn, semantic_content)
    conn.execute(
        "UPDATE facts SET embedding = ? WHERE id = ?",
        (pack_vector(embedding), fact_id),
    )
    conn.commit()


def seed_eval_suite(conn, suite: dict[str, Any], *, embed_fn=None) -> None:
    if embed_fn is None:
        embed_fn = build_anchor_embed_fn((suite.get("embedding_space") or {}).get("anchors") or [])

    for session in suite.get("sessions", []):
        turns = session.get("turns") or [
            {"role": "user", "content": f"Seed transcript for {session['session_id']}"},
            {"role": "assistant", "content": "Seeded."},
        ]
        upsert_session(
            conn,
            session["session_id"],
            session.get("agent", "claude"),
            turns,
            session.get("started_at", "2026-01-01T00:00:00Z"),
            session.get("updated_at", "2026-01-01T00:00:00Z"),
            metadata=session.get("metadata"),
            project_context=session.get("project_context"),
        )

        working = session.get("working_memory")
        if isinstance(working, dict):
            upsert_working_memory(
                conn,
                session_id=session["session_id"],
                current_goal=working["current_goal"],
                current_focus=working.get("current_focus", ""),
                next_step=working["next_step"],
                status=working.get("status", "ready_to_resume"),
                updated_at=working.get("updated_at", session.get("updated_at", "2026-01-01T00:00:00Z")),
                details={
                    "active_tasks": working.get("active_tasks", []),
                    "constraints": working.get("constraints", []),
                    "source": "offline_eval",
                },
                embedding=_memory_embedding_vector(
                    working,
                    embed_fn,
                    ". ".join(
                        part for part in [
                            working.get("current_goal", ""),
                            working.get("current_focus", ""),
                            working.get("next_step", ""),
                            "; ".join(working.get("active_tasks", []) or []),
                        ]
                        if part
                    ),
                ),
            )

        session_memory = session.get("session_memory")
        if isinstance(session_memory, dict):
            upsert_session_memory(
                conn,
                session_id=session["session_id"],
                title=session_memory["title"],
                summary=session_memory["summary"],
                left_off_at=session_memory["left_off_at"],
                updated_at=session_memory.get("updated_at", session.get("updated_at", "2026-01-01T00:00:00Z")),
                details={
                    "next_steps": session_memory.get("next_steps", []),
                    "what_was_tried": session_memory.get("what_was_tried", []),
                    "outcomes": session_memory.get("outcomes", []),
                    "source": "offline_eval",
                },
                embedding=_memory_embedding_vector(
                    session_memory,
                    embed_fn,
                    ". ".join(
                        part for part in [
                            session_memory.get("title", ""),
                            session_memory.get("summary", ""),
                            session_memory.get("left_off_at", ""),
                            "; ".join(session_memory.get("next_steps", []) or []),
                        ]
                        if part
                    ),
                ),
            )

        for episode in session.get("episodic", []):
            insert_episodic(
                conn,
                session_id=session["session_id"],
                title=episode["title"],
                abstract=episode["abstract"],
                happened_at=episode.get("happened_at", session.get("updated_at", "2026-01-01T00:00:00Z")),
                details={
                    "decisions": episode.get("decisions", []),
                    "outcomes": episode.get("outcomes", []),
                    "follow_ups": episode.get("follow_ups", []),
                    "source": "offline_eval",
                },
                embedding=_memory_embedding_vector(
                    episode,
                    embed_fn,
                    ". ".join(
                        part for part in [
                            episode.get("title", ""),
                            episode.get("abstract", ""),
                            "; ".join(episode.get("decisions", []) or []),
                            "; ".join(episode.get("outcomes", []) or []),
                            "; ".join(episode.get("follow_ups", []) or []),
                        ]
                        if part
                    ),
                ),
            )

        for procedure in session.get("procedural", []):
            insert_procedural(
                conn,
                session_id=session["session_id"],
                title=procedure["title"],
                summary=procedure["summary"],
                updated_at=procedure.get("updated_at", session.get("updated_at", "2026-01-01T00:00:00Z")),
                details={
                    "steps": procedure.get("steps", []),
                    "trigger_phrases": procedure.get("trigger_phrases", []),
                    "tools": procedure.get("tools", []),
                    "source": "offline_eval",
                },
                embedding=_memory_embedding_vector(
                    procedure,
                    embed_fn,
                    ". ".join(
                        part for part in [
                            procedure.get("title", ""),
                            procedure.get("summary", ""),
                            "; ".join(procedure.get("steps", []) or []),
                            "; ".join(procedure.get("trigger_phrases", []) or []),
                        ]
                        if part
                    ),
                ),
            )

        for fact in session.get("facts", []):
            _seed_fact(conn, session_id=session["session_id"], fact=fact, embed_fn=embed_fn)


def _session_projects(conn) -> dict[str, str | None]:
    rows = conn.execute("SELECT session_id, project_id FROM sessions").fetchall()
    return {row["session_id"]: row["project_id"] for row in rows}


def _collect_hits(context, session_projects: dict[str, str | None]) -> list[EvalHit]:
    hits: list[EvalHit] = []

    if context.working_mem:
        session_id = context.working_mem.get("session_id")
        hits.append(
            EvalHit(
                memory_type="working_memory",
                session_id=session_id,
                project_id=session_projects.get(session_id),
                label=context.working_mem.get("current_goal") or context.working_mem.get("id") or "working_memory",
            )
        )

    for memory_type, rows, label_field in (
        ("session_memory", context.session_memory, "title"),
        ("episodic", context.episodic, "title"),
        ("procedural", context.procedural, "title"),
        ("facts", context.facts, "content"),
    ):
        for row in rows:
            session_id = row.get("session_id")
            hits.append(
                EvalHit(
                    memory_type=memory_type,
                    session_id=session_id,
                    project_id=session_projects.get(session_id),
                    label=row.get(label_field) or row.get("id") or memory_type,
                )
            )
    return hits


def _counts_from_context(context) -> dict[str, int]:
    return {
        "facts": len(context.facts),
        "episodic": len(context.episodic),
        "procedural": len(context.procedural),
        "session_memory": len(context.session_memory),
        "working_memory": 1 if context.working_mem else 0,
    }


def _wrong_project_hits(
    hits: list[EvalHit],
    *,
    case_project_id: str | None,
    allowed_project_ids: list[str] | None,
) -> int:
    if allowed_project_ids:
        allowed = {project_id for project_id in allowed_project_ids if project_id}
        return sum(1 for hit in hits if hit.project_id and hit.project_id not in allowed)
    if case_project_id:
        return sum(1 for hit in hits if hit.project_id and hit.project_id != case_project_id)
    return 0


def evaluate_case(conn, case: dict[str, Any], *, embed_fn=None, session_projects=None) -> EvalCaseResult:
    if embed_fn is None:
        raise ValueError("evaluate_case requires embed_fn")
    if session_projects is None:
        session_projects = _session_projects(conn)

    context = retrieve_prompt_memory(
        conn,
        case["prompt"],
        include_working_memory=bool(case.get("include_working_memory", False)),
        session_id=case.get("session_id"),
        project_context=case.get("project_context"),
        embed_fn=embed_fn,
    )
    outcome = decide_prompt_memory_action(case["prompt"], context)
    hits = _collect_hits(context, session_projects)
    counts = _counts_from_context(context)
    observed_projects = sorted({hit.project_id for hit in hits if hit.project_id})
    observed_session_ids = sorted({hit.session_id for hit in hits if hit.session_id})

    expect = case.get("expect") or {}
    case_project_id = ((case.get("project_context") or {}).get("project_id"))
    wrong_hits = _wrong_project_hits(
        hits,
        case_project_id=case_project_id,
        allowed_project_ids=expect.get("allowed_project_ids"),
    )

    errors: list[str] = []
    expected_action = expect.get("action")
    if expected_action is not None and outcome.action != expected_action:
        errors.append(f"action expected {expected_action!r} but observed {outcome.action!r}")

    allowed_actions = expect.get("allowed_actions")
    if allowed_actions is not None and outcome.action not in set(allowed_actions):
        errors.append(f"action {outcome.action!r} not in allowed_actions={allowed_actions!r}")

    for memory_type, minimum in (expect.get("min_counts") or {}).items():
        observed = counts.get(memory_type, 0)
        if observed < int(minimum):
            errors.append(f"min_counts[{memory_type}] expected >= {minimum}, observed {observed}")

    for memory_type in expect.get("required_memory_types") or []:
        if counts.get(memory_type, 0) <= 0:
            errors.append(f"required memory type missing: {memory_type}")

    required_session_ids = set(expect.get("required_session_ids") or [])
    missing_sessions = sorted(required_session_ids - set(observed_session_ids))
    if missing_sessions:
        errors.append(f"required_session_ids missing: {missing_sessions}")

    forbidden_session_ids = set(expect.get("forbidden_session_ids") or [])
    present_forbidden_sessions = sorted(forbidden_session_ids & set(observed_session_ids))
    if present_forbidden_sessions:
        errors.append(f"forbidden_session_ids present: {present_forbidden_sessions}")

    allowed_project_ids = set(expect.get("allowed_project_ids") or [])
    if allowed_project_ids:
        disallowed = sorted(project_id for project_id in observed_projects if project_id not in allowed_project_ids)
        if disallowed:
            errors.append(f"disallowed project_ids observed: {disallowed}")

    forbidden_project_ids = set(expect.get("forbidden_project_ids") or [])
    present_forbidden_projects = sorted(project_id for project_id in observed_projects if project_id in forbidden_project_ids)
    if present_forbidden_projects:
        errors.append(f"forbidden project_ids observed: {present_forbidden_projects}")

    if "max_wrong_project_hits" in expect and wrong_hits > int(expect["max_wrong_project_hits"]):
        errors.append(
            f"max_wrong_project_hits expected <= {expect['max_wrong_project_hits']}, observed {wrong_hits}"
        )

    answer_fragment = expect.get("answer_must_contain")
    if answer_fragment and answer_fragment not in outcome.answer:
        errors.append(f"answer missing required fragment: {answer_fragment!r}")

    injection_fragment = expect.get("injection_must_contain")
    if injection_fragment and injection_fragment not in outcome.injection:
        errors.append(f"injection missing required fragment: {injection_fragment!r}")

    return EvalCaseResult(
        case_id=case["id"],
        passed=not errors,
        action=outcome.action,
        counts=counts,
        projects_hit=observed_projects,
        session_ids_hit=observed_session_ids,
        wrong_project_hits=wrong_hits,
        errors=errors,
        answer=outcome.answer,
        injection=outcome.injection,
    )


def run_eval_suite(conn, suite: dict[str, Any]) -> EvalSuiteResult:
    embed_fn = build_anchor_embed_fn((suite.get("embedding_space") or {}).get("anchors") or [])
    seed_eval_suite(conn, suite, embed_fn=embed_fn)
    session_projects = _session_projects(conn)
    results = [
        evaluate_case(conn, case, embed_fn=embed_fn, session_projects=session_projects)
        for case in suite.get("cases", [])
    ]
    passed = sum(1 for result in results if result.passed)
    return EvalSuiteResult(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        cases=results,
    )


def run_eval_suite_from_path(path: str | Path = DEFAULT_SUITE_PATH) -> EvalSuiteResult:
    conn = init_db(":memory:")
    try:
        return run_eval_suite(conn, load_eval_suite(path))
    finally:
        conn.close()
