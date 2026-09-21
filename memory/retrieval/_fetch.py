from __future__ import annotations

import time

from memory.retrieval._models import (
    EPISODIC_FETCH_LIMIT,
    EPISODIC_LIMIT,
    EPISODIC_MIN_SIMILARITY,
    EPISODIC_RECENT_LIMIT,
    FACT_FETCH_LIMIT,
    FACT_LIMIT,
    FACT_MIN_SIMILARITY,
    PROCEDURAL_FETCH_LIMIT,
    PROCEDURAL_LIMIT,
    PROCEDURAL_MIN_SIMILARITY,
    PROJECT_SCOPED_FETCH_MULTIPLIER,
    SESSION_FTS_FETCH_LIMIT,
    SESSION_MEMORY_FETCH_LIMIT,
    SESSION_MEMORY_LIMIT,
    SESSION_MEMORY_MIN_SIMILARITY,
    MemoryRow,
    RetrievalWarning,
    WakeUpContext,
)
from memory.retrieval._hybrid import fuse_ranked_lanes
from memory.retrieval._text import _append_warning
from memory.retrieval._project import (
    enrich_rows_with_project_metadata,
    is_row_allowed_for_project_policy,
    load_session_project_contexts,
    resolve_ambient_project_context,
    same_project_session_ids,
)
from memory.retrieval._rank import (
    _filter_by_similarity,
    _is_substantive_episode,
    _looks_like_missing_memory_episode,
    _rank_rows,
)
from memory.retrieval._intent import _prompt_requests_recent_episode_summary

from memory.db import (
    search_episodic_fts,
    search_facts_semantic,
    search_procedural_fts,
    search_session_fts,
    search_session_memory_fts,
)
from memory.episodic.repository import list_recent_episodic_memories, list_session_episodes, retrieve_episodic_memories
from memory.llm.inference import embed_text
from memory.procedural.repository import list_session_procedures, retrieve_procedural_memories
from memory.session.repository import list_session_memory_rows, retrieve_session_memories
from memory.working_memory.repository import retrieve_working_memory as _retrieve_wm


def _search_session_hits(
    conn,
    prompt: str,
    warnings: list[RetrievalWarning],
    *,
    session_id: str | None,
    same_project_session_ids: set[str] | None,
) -> list[dict]:
    try:
        return [
            row
            for row in search_session_fts(
                conn,
                prompt,
                limit=SESSION_FTS_FETCH_LIMIT,
                session_ids=same_project_session_ids,
            )
            if row.get("session_id") != session_id
        ]
    except Exception as exc:
        _append_warning(warnings, "session_keyword", exc)
        return []


def _session_hit_rows(rows: list[MemoryRow], session_hits: list[dict]) -> list[MemoryRow]:
    hit_by_session_id = {row.get("session_id"): row for row in session_hits if row.get("session_id")}
    expanded: list[MemoryRow] = []
    for row in rows:
        session_id = row.get("session_id")
        session_hit = hit_by_session_id.get(session_id)
        if not session_hit:
            continue
        payload = dict(row)
        payload["session_hit"] = True
        payload["session_hit_score"] = float(session_hit.get("keyword_score", 0.0) or 0.0)
        payload["session_hit_rank"] = session_hit.get("keyword_rank")
        expanded.append(payload)
    return expanded


def _expand_session_hits(
    conn,
    session_hits: list[dict],
    warnings: list[RetrievalWarning],
) -> dict[str, list[MemoryRow]]:
    session_ids = [row.get("session_id") for row in session_hits if row.get("session_id")]
    if not session_ids:
        return {"episodic": [], "procedural": [], "session_memory": []}
    expanded = {"episodic": [], "procedural": [], "session_memory": []}
    try:
        episodic_rows: list[MemoryRow] = []
        for hit_session_id in session_ids:
            episodic_rows.extend(list_session_episodes(conn, session_id=hit_session_id))
        expanded["episodic"] = _session_hit_rows(episodic_rows, session_hits)
    except Exception as exc:
        _append_warning(warnings, "episodic_session_keyword", exc)
    try:
        procedural_rows: list[MemoryRow] = []
        for hit_session_id in session_ids:
            procedural_rows.extend(list_session_procedures(conn, session_id=hit_session_id))
        expanded["procedural"] = _session_hit_rows(procedural_rows, session_hits)
    except Exception as exc:
        _append_warning(warnings, "procedural_session_keyword", exc)
    try:
        session_memory_rows: list[MemoryRow] = []
        for hit_session_id in session_ids:
            session_memory_rows.extend(list_session_memory_rows(conn, session_id=hit_session_id))
        expanded["session_memory"] = _session_hit_rows(session_memory_rows, session_hits)
    except Exception as exc:
        _append_warning(warnings, "session_memory_session_keyword", exc)
    return expanded


def _retrieve_episodic(
    conn,
    prompt: str,
    prompt_vec,
    embed_fn,
    warnings: list[RetrievalWarning],
    *,
    same_project_session_ids: set[str] | None,
    session_rows: list[MemoryRow] | None = None,
) -> list[MemoryRow]:
    if same_project_session_ids is not None and not same_project_session_ids:
        return []
    semantic_rows: list[MemoryRow] = []
    keyword_rows: list[MemoryRow] = []
    try:
        semantic_rows = retrieve_episodic_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=EPISODIC_MIN_SIMILARITY,
            limit=EPISODIC_FETCH_LIMIT,
            source="wake_up",
            session_ids=same_project_session_ids,
        )
    except Exception as exc:
        _append_warning(warnings, "episodic", exc)
    try:
        keyword_rows = search_episodic_fts(
            conn,
            prompt,
            limit=EPISODIC_FETCH_LIMIT * PROJECT_SCOPED_FETCH_MULTIPLIER,
            session_ids=same_project_session_ids,
        )
    except Exception as exc:
        _append_warning(warnings, "episodic_keyword", exc)
    return fuse_ranked_lanes(
        ("semantic", semantic_rows),
        ("keyword", keyword_rows),
        ("session", session_rows or []),
    )


def _retrieve_recent_episodic(
    conn,
    warnings: list[RetrievalWarning],
    *,
    ambient_project: dict | None,
    session_projects: dict[str, dict[str, str]],
) -> list[MemoryRow]:
    try:
        recent = list_recent_episodic_memories(conn, limit=EPISODIC_RECENT_LIMIT, source="wake_up_recent")
        recent = enrich_rows_with_project_metadata(
            recent,
            ambient_project=ambient_project,
            session_projects=session_projects,
            kind="episodic",
        )
        substantive = [item for item in recent if _is_substantive_episode(item)]
        allowed = [item for item in substantive if is_row_allowed_for_project_policy(item, "episodic", ambient_project)]
        return [item for item in allowed if not _looks_like_missing_memory_episode(item)][:EPISODIC_FETCH_LIMIT]
    except Exception as exc:
        _append_warning(warnings, "episodic_recent", exc)
        return []


def _retrieve_facts(
    conn,
    prompt: str,
    prompt_vec,
    warnings: list[RetrievalWarning],
    *,
    ambient_project: dict | None,
    session_projects: dict[str, dict[str, str]],
) -> list[MemoryRow]:
    del prompt
    try:
        fact_results = search_facts_semantic(
            conn,
            prompt_vec,
            limit=FACT_FETCH_LIMIT * PROJECT_SCOPED_FETCH_MULTIPLIER,
        )
        fact_results = enrich_rows_with_project_metadata(
            fact_results,
            ambient_project=ambient_project,
            session_projects=session_projects,
            kind="facts",
        )
        fact_results = [
            row for row in fact_results if is_row_allowed_for_project_policy(row, "facts", ambient_project)
        ]
        return _filter_by_similarity(fact_results, FACT_MIN_SIMILARITY)
    except Exception as exc:
        _append_warning(warnings, "facts", exc)
        return []


def _retrieve_procedural(
    conn,
    prompt: str,
    prompt_vec,
    embed_fn,
    warnings: list[RetrievalWarning],
    *,
    session_rows: list[MemoryRow] | None = None,
) -> list[MemoryRow]:
    semantic_rows: list[MemoryRow] = []
    keyword_rows: list[MemoryRow] = []
    try:
        semantic_rows = retrieve_procedural_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=PROCEDURAL_MIN_SIMILARITY,
            limit=PROCEDURAL_FETCH_LIMIT,
            source="wake_up",
        )
    except Exception as exc:
        _append_warning(warnings, "procedural", exc)
    try:
        keyword_rows = search_procedural_fts(
            conn,
            prompt,
            limit=PROCEDURAL_FETCH_LIMIT * 3,
        )
    except Exception as exc:
        _append_warning(warnings, "procedural_keyword", exc)
    return fuse_ranked_lanes(
        ("semantic", semantic_rows),
        ("keyword", keyword_rows),
        ("session", session_rows or []),
    )


def _retrieve_working_memory(
    conn,
    session_id: str | None,
    warnings: list[RetrievalWarning],
    *,
    ambient_project: dict | None,
    session_projects: dict[str, dict[str, str]],
) -> MemoryRow | None:
    if not session_id:
        return None
    try:
        working_memory = _retrieve_wm(conn, session_id=session_id, source="wake_up")
        if working_memory is None:
            return None
        enriched = enrich_rows_with_project_metadata(
            [working_memory],
            ambient_project=ambient_project,
            session_projects=session_projects,
            kind="working_memory",
        )[0]
        if not is_row_allowed_for_project_policy(enriched, "working_memory", ambient_project):
            return None
        return enriched
    except Exception as exc:
        _append_warning(warnings, "working_memory", exc)
        return None


def _retrieve_session_memory(
    conn,
    prompt: str,
    prompt_vec,
    embed_fn,
    warnings: list[RetrievalWarning],
    *,
    session_id: str | None,
    same_project_session_ids: set[str] | None,
    session_rows: list[MemoryRow] | None = None,
) -> list[MemoryRow]:
    if same_project_session_ids is not None and not same_project_session_ids:
        return []
    semantic_rows: list[MemoryRow] = []
    keyword_rows: list[MemoryRow] = []
    try:
        semantic_rows = retrieve_session_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=SESSION_MEMORY_MIN_SIMILARITY,
            limit=SESSION_MEMORY_FETCH_LIMIT,
            source="wake_up",
            exclude_session_id=session_id,
            session_ids=same_project_session_ids,
        )
    except Exception as exc:
        _append_warning(warnings, "session_memory", exc)
    try:
        keyword_rows = [
            row
            for row in search_session_memory_fts(
                conn,
                prompt,
                limit=SESSION_MEMORY_FETCH_LIMIT * PROJECT_SCOPED_FETCH_MULTIPLIER,
                session_ids=same_project_session_ids,
            )
            if row.get("session_id") != session_id
        ]
    except Exception as exc:
        _append_warning(warnings, "session_memory_keyword", exc)
    return fuse_ranked_lanes(
        ("semantic", semantic_rows),
        ("keyword", keyword_rows),
        ("session", session_rows or []),
    )


def retrieve_wake_up_context(
    conn,
    prompt: str,
    *,
    include_working_memory: bool,
    session_id: str | None = None,
    project_context: dict | None = None,
    embed_fn=embed_text,
) -> WakeUpContext:
    """Retrieve wake-up context for one user prompt.

    Retrieval is intentionally broad: all memory types are searched with one
    shared query embedding. Narrowing happens via per-type ranking and result caps.
    """
    ambient_project = resolve_ambient_project_context(
        conn,
        project_context=project_context,
        session_id=session_id,
    )

    warnings: list[RetrievalWarning] = []
    overall_started = time.perf_counter()
    embed_started = overall_started
    prompt_vec = embed_fn(prompt)
    embed_ms = round((time.perf_counter() - embed_started) * 1000, 3)

    search_started = time.perf_counter()
    scoped_session_ids = same_project_session_ids(conn, ambient_project)
    session_hits = _search_session_hits(
        conn,
        prompt,
        warnings,
        session_id=session_id,
        same_project_session_ids=scoped_session_ids,
    )
    session_hit_rows = _expand_session_hits(conn, session_hits, warnings)

    project_session_ids: set[str] = set(scoped_session_ids or set())
    if session_id:
        project_session_ids.add(session_id)
    project_session_ids.update({row.get("session_id") for row in session_hits if row.get("session_id")})
    session_projects = load_session_project_contexts(conn, session_ids=project_session_ids) if project_session_ids else {}

    working_mem = _retrieve_working_memory(
        conn,
        session_id if (include_working_memory or session_id) else None,
        warnings,
        ambient_project=ambient_project,
        session_projects=session_projects,
    )
    episodic = enrich_rows_with_project_metadata(
        _retrieve_episodic(
            conn,
            prompt,
            prompt_vec,
            embed_fn,
            warnings,
            same_project_session_ids=scoped_session_ids,
            session_rows=session_hit_rows["episodic"],
        ),
        ambient_project=ambient_project,
        session_projects=session_projects,
        kind="episodic",
    )
    facts = enrich_rows_with_project_metadata(
        _retrieve_facts(
            conn,
            prompt,
            prompt_vec,
            warnings,
            ambient_project=ambient_project,
            session_projects=session_projects,
        ),
        ambient_project=ambient_project,
        session_projects=session_projects,
        kind="facts",
    )
    procedural_raw = _retrieve_procedural(
        conn,
        prompt,
        prompt_vec,
        embed_fn,
        warnings,
        session_rows=session_hit_rows["procedural"],
    )
    procedural_projects = load_session_project_contexts(
        conn,
        session_ids={row.get("session_id") for row in procedural_raw if row.get("session_id")},
    ) if procedural_raw else {}
    procedural = enrich_rows_with_project_metadata(
        procedural_raw,
        ambient_project=ambient_project,
        session_projects=procedural_projects,
        kind="procedural",
    )
    session_memory = enrich_rows_with_project_metadata(
        _retrieve_session_memory(
            conn,
            prompt,
            prompt_vec,
            embed_fn,
            warnings,
            session_id=session_id,
            same_project_session_ids=scoped_session_ids,
            session_rows=session_hit_rows["session_memory"],
        ),
        ambient_project=ambient_project,
        session_projects=session_projects,
        kind="session_memory",
    )

    episodic = _rank_rows(episodic, "episodic", prompt, limit=EPISODIC_LIMIT, project_context=ambient_project)
    facts = _rank_rows(facts, "facts", prompt, limit=FACT_LIMIT, project_context=ambient_project)
    procedural = _rank_rows(procedural, "procedural", prompt, limit=PROCEDURAL_LIMIT, project_context=ambient_project)
    session_memory = _rank_rows(
        session_memory,
        "session_memory",
        prompt,
        limit=SESSION_MEMORY_LIMIT,
        project_context=ambient_project,
    )

    if not episodic and not session_memory and _prompt_requests_recent_episode_summary(prompt_vec):
        episodic = _rank_rows(
            _retrieve_recent_episodic(
                conn,
                warnings,
                ambient_project=ambient_project,
                session_projects=session_projects,
            ),
            "episodic",
            prompt,
            limit=EPISODIC_LIMIT,
            project_context=ambient_project,
        )

    search_ms = round((time.perf_counter() - search_started) * 1000, 3)
    total_ms = round((time.perf_counter() - overall_started) * 1000, 3)

    return WakeUpContext(
        cache_hit=None,
        working_mem=working_mem,
        enrichment=[],
        episodic=episodic,
        facts=facts,
        procedural=procedural,
        session_memory=session_memory,
        warnings=warnings,
        prompt_vec=prompt_vec,
        timings={
            "prompt_embedding_ms": embed_ms,
            "memory_search_ms": search_ms,
            "retrieval_total_ms": total_ms,
        },
    )
