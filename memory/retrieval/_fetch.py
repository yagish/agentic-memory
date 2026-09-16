from __future__ import annotations

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
    SESSION_MEMORY_FETCH_LIMIT,
    SESSION_MEMORY_LIMIT,
    SESSION_MEMORY_MIN_SIMILARITY,
    MemoryRow,
    RetrievalWarning,
    WakeUpContext,
)
from memory.retrieval._text import _append_warning
from memory.retrieval._rank import (
    _filter_by_similarity,
    _is_substantive_episode,
    _looks_like_missing_memory_episode,
    _rank_rows,
)
from memory.retrieval._intent import _prompt_requests_recent_episode_summary

from memory.db import search_facts_semantic
from memory.episodic.repository import list_recent_episodic_memories, retrieve_episodic_memories
from memory.llm.inference import embed_text
from memory.procedural.repository import retrieve_procedural_memories
from memory.session.repository import retrieve_session_memories
from memory.working_memory.repository import retrieve_working_memory as _retrieve_wm


def _retrieve_episodic(conn, prompt: str, prompt_vec, embed_fn, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    try:
        return retrieve_episodic_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=EPISODIC_MIN_SIMILARITY,
            limit=EPISODIC_FETCH_LIMIT,
            source="wake_up",
        )
    except Exception as exc:
        _append_warning(warnings, "episodic", exc)
        return []


def _retrieve_recent_episodic(conn, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    try:
        recent = list_recent_episodic_memories(conn, limit=EPISODIC_RECENT_LIMIT, source="wake_up_recent")
        substantive = [item for item in recent if _is_substantive_episode(item)]
        return [item for item in substantive if not _looks_like_missing_memory_episode(item)][:EPISODIC_FETCH_LIMIT]
    except Exception as exc:
        _append_warning(warnings, "episodic_recent", exc)
        return []


def _retrieve_facts(conn, prompt: str, prompt_vec, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    del prompt
    try:
        fact_results = search_facts_semantic(conn, prompt_vec, limit=FACT_FETCH_LIMIT)
        return _filter_by_similarity(fact_results, FACT_MIN_SIMILARITY)
    except Exception as exc:
        _append_warning(warnings, "facts", exc)
        return []


def _retrieve_procedural(conn, prompt: str, prompt_vec, embed_fn, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    try:
        return retrieve_procedural_memories(
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
        return []


def _retrieve_working_memory(conn, session_id: str | None, warnings: list[RetrievalWarning]) -> MemoryRow | None:
    if not session_id:
        return None
    try:
        return _retrieve_wm(conn, session_id=session_id, source="wake_up")
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
) -> list[MemoryRow]:
    try:
        return retrieve_session_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=SESSION_MEMORY_MIN_SIMILARITY,
            limit=SESSION_MEMORY_FETCH_LIMIT,
            source="wake_up",
            exclude_session_id=session_id,
        )
    except Exception as exc:
        _append_warning(warnings, "session_memory", exc)
        return []


def retrieve_wake_up_context(
    conn,
    prompt: str,
    *,
    include_working_memory: bool,
    session_id: str | None = None,
    embed_fn=embed_text,
) -> WakeUpContext:
    """Retrieve wake-up context for one user prompt.

    Retrieval is intentionally broad: all memory types are searched with one
    shared query embedding. Narrowing happens via per-type ranking and result caps.
    """
    warnings: list[RetrievalWarning] = []
    prompt_vec = embed_fn(prompt)

    working_mem = _retrieve_working_memory(
        conn,
        session_id if (include_working_memory or session_id) else None,
        warnings,
    )
    episodic = _rank_rows(_retrieve_episodic(conn, prompt, prompt_vec, embed_fn, warnings), "episodic", prompt, limit=EPISODIC_LIMIT)
    facts = _rank_rows(_retrieve_facts(conn, prompt, prompt_vec, warnings), "facts", prompt, limit=FACT_LIMIT)
    procedural = _rank_rows(_retrieve_procedural(conn, prompt, prompt_vec, embed_fn, warnings), "procedural", prompt, limit=PROCEDURAL_LIMIT)
    session_memory = _rank_rows(
        _retrieve_session_memory(conn, prompt, prompt_vec, embed_fn, warnings, session_id=session_id),
        "session_memory",
        prompt,
        limit=SESSION_MEMORY_LIMIT,
    )

    if not episodic and not session_memory and _prompt_requests_recent_episode_summary(prompt_vec):
        episodic = _rank_rows(_retrieve_recent_episodic(conn, warnings), "episodic", prompt, limit=EPISODIC_LIMIT)

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
    )
