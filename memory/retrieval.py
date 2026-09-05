"""Wake-up retrieval focused on episodic memory and facts.

Adapters should call these functions instead of assembling retrieval policy
inline.

Current runtime mode keeps only:
- episodic memory
- facts

The extra fields on ``WakeUpContext`` remain for caller compatibility, but they
are intentionally left empty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.db import search_facts_semantic
from memory.episodic_repository import retrieve_episodic_memories
from memory.inference import embed_text

MemoryRow = dict[str, Any]

# 500-token hard budget (estimated as chars / 4).
TOKEN_BUDGET = 500
CHARS_BUDGET = TOKEN_BUDGET * 4

EPISODIC_MIN_SIMILARITY = 0.72
FACT_MIN_SIMILARITY = 0.76
EPISODIC_LIMIT = 3
FACT_LIMIT = 5


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
    warnings: list[RetrievalWarning] = field(default_factory=list)


class _BudgetedSectionBuilder:
    """Collect normalized text sections without exceeding the wake-up budget."""

    def __init__(self, char_budget: int) -> None:
        self._char_budget = char_budget
        self._chars_used = 0
        self._sections: list[str] = []

    def add(self, block: str) -> None:
        block = _normalize_text(block)
        if not block:
            return
        if self._chars_used + len(block) > self._char_budget:
            return
        self._sections.append(block)
        self._chars_used += len(block)

    def render(self) -> str:
        if not self._sections:
            return ""
        return f"[Memory context: {' '.join(self._sections)}]"


def build_fact_query(prompt: str) -> str:
    """Build a punctuation-safe FTS query for fact lookup."""
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    keywords = [token for token in tokens if len(token) > 2 or token in {"my"}]

    deduped: list[str] = []
    for token in keywords:
        if token not in deduped:
            deduped.append(token)

    return " OR ".join(deduped)


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


def _filter_by_similarity(rows: list[MemoryRow], threshold: float) -> list[MemoryRow]:
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


def _format_facts(facts: list[MemoryRow]) -> str:
    lines: list[str] = []
    for fact in facts[:3]:
        content = _normalize_text(fact.get("content", ""))
        if content:
            lines.append(_naturalize_fact(content))
    return " ".join(lines)


def build_wake_up_injection(context: WakeUpContext) -> str:
    """Assemble the wake-up memory block within the char budget."""
    builder = _BudgetedSectionBuilder(CHARS_BUDGET)
    builder.add(_format_episodic(context.episodic))
    builder.add(_format_facts(context.facts))
    return builder.render()


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


def _retrieve_facts(conn, prompt_vec, warnings: list[RetrievalWarning]) -> list[MemoryRow]:
    try:
        fact_results = search_facts_semantic(conn, prompt_vec, limit=FACT_LIMIT)
        return _filter_by_similarity(fact_results, FACT_MIN_SIMILARITY)
    except Exception as exc:
        _append_warning(warnings, "facts", exc)
        return []


def retrieve_wake_up_context(
    conn,
    prompt: str,
    *,
    include_working_memory: bool,
    embed_fn=embed_text,
) -> WakeUpContext:
    """Retrieve wake-up context for one user prompt.

    ``include_working_memory`` is retained for API compatibility but ignored.
    """
    del include_working_memory

    warnings: list[RetrievalWarning] = []
    prompt_vec = embed_fn(prompt)

    episodic = _retrieve_episodic(conn, prompt, prompt_vec, embed_fn, warnings)
    facts = _retrieve_facts(conn, prompt_vec, warnings)

    return WakeUpContext(
        cache_hit=None,
        working_mem=None,
        enrichment=[],
        episodic=episodic,
        facts=facts,
        procedural=[],
        warnings=warnings,
    )
