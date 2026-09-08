"""Wake-up retrieval focused on durable structured memories.

Adapters should call these functions instead of assembling retrieval policy
inline.

Current runtime mode keeps:
- working memory
- session memory
- episodic memory
- facts
- procedural memory

The extra fields on ``WakeUpContext`` remain for caller compatibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.db import search_facts_semantic
from memory.episodic_repository import list_recent_episodic_memories, retrieve_episodic_memories
from memory.inference import embed_text
from memory.procedural_repository import retrieve_procedural_memories
from memory.session_memory_repository import retrieve_session_memories
from memory.working_memory_repository import retrieve_working_memory

MemoryRow = dict[str, Any]

# 500-token hard budget (estimated as chars / 4).
TOKEN_BUDGET = 500
CHARS_BUDGET = TOKEN_BUDGET * 4

EPISODIC_MIN_SIMILARITY = 0.72
FACT_MIN_SIMILARITY = 0.38
EPISODIC_LIMIT = 3
EPISODIC_RECENT_LIMIT = 5
FACT_LIMIT = 5
PROCEDURAL_MIN_SIMILARITY = 0.74
PROCEDURAL_LIMIT = 2
SESSION_MEMORY_MIN_SIMILARITY = 0.78
SESSION_MEMORY_LIMIT = 2


@dataclass(frozen=True)
class RetrievalWarning:
    """A non-fatal issue encountered while assembling retrieval context."""

    stage: str
    message: str


@dataclass(frozen=True)
class WakeUpContext:
    """Structured retrieval output for the wake-up hook adapter."""

    cache_hit: MemoryRow | None
    working_mem: MemoryRow | None
    enrichment: list[MemoryRow]
    episodic: list[MemoryRow]
    facts: list[MemoryRow]
    procedural: list[MemoryRow]
    session_memory: list[MemoryRow] = field(default_factory=list)
    warnings: list[RetrievalWarning] = field(default_factory=list)



def _normalize_text(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()



def _as_sentence(content: str) -> str:
    content = _normalize_text(content)
    if not content:
        return ""
    return content if content.endswith((".", "!", "?")) else f"{content}."



def _naturalize_fact(content: str) -> str:
    content = _normalize_text(content)
    match = re.fullmatch(r"([a-z0-9_]+)\.([a-z0-9_]+)\s*=\s*(.+)", content, re.IGNORECASE)
    if match:
        entity, attribute, value = match.groups()
        return _as_sentence(f"Remembered fact: {entity}.{attribute} = {value}")
    return _as_sentence(content)



def _append_warning(warnings: list[RetrievalWarning], stage: str, exc: Exception) -> None:
    warnings.append(RetrievalWarning(stage, str(exc)))



def _prompt_requests_recent_episode_summary(prompt: str) -> bool:
    normalized = prompt.lower()
    time_words = {"last", "latest", "recent", "recently", "previous", "before"}
    topic_words = {"work", "working", "worked", "doing", "did", "task", "project", "session", "chat", "conversation", "discuss", "discussed", "talk", "talked"}
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return bool(tokens & time_words) and bool(tokens & topic_words)



def _looks_like_missing_memory_episode(item: MemoryRow) -> bool:
    haystack = _normalize_text(f"{item.get('title', '')} {item.get('abstract', '')}").lower()
    return any(
        phrase in haystack
        for phrase in (
            "no memory",
            "not stored in",
            "no specific information",
            "not recalled",
            "nothing was recalled",
            "nothing recalled",
        )
    )



def _is_substantive_episode(item: MemoryRow) -> bool:
    details_count = sum(len(item.get(key, []) or []) for key in ("decisions", "outcomes", "follow_ups"))
    return details_count >= 2 and not _looks_like_missing_memory_episode(item)



def _filter_by_similarity(rows: list[MemoryRow], threshold: float | None) -> list[MemoryRow]:
    if threshold is None:
        return rows
    return [row for row in rows if row.get("similarity", 0.0) >= threshold]



def _format_episode(item: MemoryRow) -> str:
    title = _normalize_text(item.get("title", ""))
    abstract = _as_sentence(item.get("abstract", ""))
    decisions = [_normalize_text(value) for value in item.get("decisions", []) if _normalize_text(value)]
    outcomes = [_normalize_text(value) for value in item.get("outcomes", []) if _normalize_text(value)]
    follow_ups = [_normalize_text(value) for value in item.get("follow_ups", []) if _normalize_text(value)]

    fragments: list[str] = []
    if title and abstract:
        fragments.append(f"Recent related episode: {title}. {abstract}")
    elif abstract:
        fragments.append(f"Recent related episode: {abstract}")
    elif title:
        fragments.append(f"Recent related episode: {title}.")

    if decisions:
        fragments.append(f"Decision: {'; '.join(decisions[:2])}.")
    if outcomes:
        fragments.append(f"Outcome: {'; '.join(outcomes[:2])}.")
    if follow_ups:
        fragments.append(f"Follow-up: {'; '.join(follow_ups[:2])}.")

    return " ".join(fragments)



def _format_episodic(episodic: list[MemoryRow]) -> str:
    return " ".join(_format_episode(item) for item in episodic[:2] if _format_episode(item))



def _format_working_memory(item: MemoryRow | None) -> str:
    if not item:
        return ""
    goal = _normalize_text(item.get("current_goal", ""))
    focus = _normalize_text(item.get("current_focus", ""))
    next_step = _normalize_text(item.get("next_step", ""))
    status = _normalize_text(item.get("status", ""))
    active_tasks = [_normalize_text(value) for value in item.get("active_tasks", []) if _normalize_text(value)]
    constraints = [_normalize_text(value) for value in item.get("constraints", []) if _normalize_text(value)]

    fragments: list[str] = []
    if goal:
        fragments.append(f"Current working goal: {goal}.")
    if focus:
        fragments.append(f"Current focus: {focus}.")
    if active_tasks:
        fragments.append(f"Active tasks: {'; '.join(active_tasks[:3])}.")
    if constraints:
        fragments.append(f"Constraints: {'; '.join(constraints[:2])}.")
    if next_step:
        fragments.append(f"Next step: {next_step}.")
    if status:
        fragments.append(f"Status: {status}.")
    return " ".join(fragments)



def _format_facts(facts: list[MemoryRow]) -> str:
    lines: list[str] = []
    for fact in facts[:3]:
        content = _normalize_text(fact.get("content", ""))
        if content:
            lines.append(_naturalize_fact(content))
    return " ".join(lines)



def _format_procedural(procedural: list[MemoryRow]) -> str:
    fragments: list[str] = []
    for item in procedural[:2]:
        title = _normalize_text(item.get("title", ""))
        summary = _as_sentence(item.get("summary", ""))
        steps = [_normalize_text(value) for value in item.get("steps", []) if _normalize_text(value)]
        if title and summary:
            fragments.append(f"Relevant how-to pattern: {title}. {summary}")
        elif summary:
            fragments.append(f"Relevant how-to pattern: {summary}")
        elif title:
            fragments.append(f"Relevant how-to pattern: {title}.")
        if steps:
            fragments.append(f"Steps: {'; '.join(steps[:4])}.")
    return " ".join(fragments)



def _format_session_memory(items: list[MemoryRow]) -> str:
    fragments: list[str] = []
    for item in items[:2]:
        title = _normalize_text(item.get("title", ""))
        summary = _as_sentence(item.get("summary", ""))
        left_off_at = _normalize_text(item.get("left_off_at", ""))
        next_steps = [_normalize_text(value) for value in item.get("next_steps", []) if _normalize_text(value)]
        if title and summary:
            fragments.append(f"Relevant prior session: {title}. {summary}")
        elif summary:
            fragments.append(f"Relevant prior session: {summary}")
        elif title:
            fragments.append(f"Relevant prior session: {title}.")
        if left_off_at:
            fragments.append(f"Left off at: {left_off_at}.")
        if next_steps:
            fragments.append(f"Next session: {'; '.join(next_steps[:2])}.")
    return " ".join(fragments)



def _context_sections(context: WakeUpContext) -> list[str]:
    return [
        _format_working_memory(context.working_mem),
        _format_session_memory(context.session_memory),
        _format_episodic(context.episodic),
        _format_procedural(context.procedural),
        _format_facts(context.facts),
    ]



def _build_context_body(context: WakeUpContext) -> str:
    return _normalize_text(" ".join(section for section in _context_sections(context) if section))



def _build_context_envelope(body: str) -> str:
    body = _normalize_text(body)
    if not body:
        return ""
    return f"[Memory context: {body}]"



def _fit_context_to_budget(sections: list[str], *, char_budget: int) -> str:
    available = max(1, char_budget - len("[Memory context: ]"))
    kept: list[str] = []

    for section in (_normalize_text(section) for section in sections):
        if not section:
            continue
        candidate = _normalize_text(" ".join(kept + [section]))
        if len(candidate) <= available:
            kept.append(section)
            continue
        if not kept:
            return section[:available].rstrip()
        break

    return _normalize_text(" ".join(kept))



def build_wake_up_injection(context: WakeUpContext) -> str:
    """Assemble the wake-up memory block within the char budget.

    This path is intentionally deterministic and model-free.
    """
    body = _fit_context_to_budget(_context_sections(context), char_budget=CHARS_BUDGET)
    if not body:
        return ""
    return _build_context_envelope(body)



def _retrieve_episodic(conn, prompt: str, prompt_vec, embed_fn, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    try:
        return retrieve_episodic_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=EPISODIC_MIN_SIMILARITY,
            limit=EPISODIC_LIMIT,
            source="wake_up",
        )
    except Exception as exc:
        _append_warning(warnings, "episodic", exc)
        return []



def _retrieve_recent_episodic(conn, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    try:
        recent = list_recent_episodic_memories(conn, limit=EPISODIC_RECENT_LIMIT, source="wake_up_recent")
        substantive = [item for item in recent if _is_substantive_episode(item)]
        filtered = [item for item in substantive if not _looks_like_missing_memory_episode(item)]
        return filtered[:EPISODIC_LIMIT]
    except Exception as exc:
        _append_warning(warnings, "episodic_recent", exc)
        return []



def _retrieve_facts(conn, prompt: str, prompt_vec, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    del prompt
    try:
        fact_results = search_facts_semantic(conn, prompt_vec, limit=FACT_LIMIT)
        return _filter_by_similarity(fact_results, FACT_MIN_SIMILARITY)
    except Exception as exc:
        _append_warning(warnings, "facts", exc)
        return []



def _prompt_requests_procedural_help(prompt: str) -> bool:
    normalized = prompt.lower().strip()
    if normalized.startswith((
        "how do i",
        "how to",
        "remind me how",
        "what is the deploy process",
        "what's the deploy process",
    )):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return bool(tokens & {"how", "steps", "workflow", "procedure", "process", "deploy", "release", "rollback", "restart", "setup", "configure", "install"})



def _prompt_requests_resume_context(prompt: str) -> bool:
    normalized = prompt.lower().strip()
    phrases = (
        "pick up where i left off",
        "pick up where we left off",
        "where did i leave off",
        "where did we leave off",
        "what was i working on",
        "what were we working on",
        "resume the work",
        "continue the work",
        "continue where i left off",
        "continue where we left off",
        "remind me what i was doing",
    )
    if any(phrase in normalized for phrase in phrases):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return bool(tokens & {"resume", "continue", "left", "leftoff", "previous", "last", "recent", "working", "handoff"}) and bool(tokens & {"work", "task", "session", "project", "doing"})



def _retrieve_procedural(conn, prompt: str, prompt_vec, embed_fn, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    if not _prompt_requests_procedural_help(prompt):
        return []
    try:
        return retrieve_procedural_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=PROCEDURAL_MIN_SIMILARITY,
            limit=PROCEDURAL_LIMIT,
            source="wake_up",
        )
    except Exception as exc:
        _append_warning(warnings, "procedural", exc)
        return []



def _retrieve_working_memory(conn, session_id: str | None, warnings: list[RetrievalWarning]) -> MemoryRow | None:
    if not session_id:
        return None
    try:
        return retrieve_working_memory(conn, session_id=session_id, source="wake_up")
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
    if not _prompt_requests_resume_context(prompt):
        return []
    try:
        return retrieve_session_memories(
            conn,
            prompt,
            query_vector=prompt_vec,
            embed_fn=embed_fn,
            min_similarity=SESSION_MEMORY_MIN_SIMILARITY,
            limit=SESSION_MEMORY_LIMIT,
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
    """Retrieve wake-up context for one user prompt."""

    warnings: list[RetrievalWarning] = []
    prompt_vec = embed_fn(prompt)

    working_mem = _retrieve_working_memory(
        conn,
        session_id if (include_working_memory or session_id) else None,
        warnings,
    )
    episodic = _retrieve_episodic(conn, prompt, prompt_vec, embed_fn, warnings)
    if not episodic and _prompt_requests_recent_episode_summary(prompt):
        episodic = _retrieve_recent_episodic(conn, warnings)
    facts = _retrieve_facts(conn, prompt, prompt_vec, warnings)
    procedural = _retrieve_procedural(conn, prompt, prompt_vec, embed_fn, warnings)
    session_memory = _retrieve_session_memory(
        conn,
        prompt,
        prompt_vec,
        embed_fn,
        warnings,
        session_id=session_id,
    )

    return WakeUpContext(
        cache_hit=None,
        working_mem=working_mem,
        enrichment=[],
        episodic=episodic,
        facts=facts,
        procedural=procedural,
        session_memory=session_memory,
        warnings=warnings,
    )
