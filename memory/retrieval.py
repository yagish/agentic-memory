"""Deep retrieval module for wake-up and agent recall policy.

Adapters should call these functions instead of assembling retrieval layers
and ranking policy directly.

Current runtime mode supports the full memory stack:
- session cache / enrichment from compacted sessions
- working memory
- episodic memory
- facts
- procedural memory
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memory.db import (
    increment_compacted_hit,
    search_compacted_sessions,
    search_episodic_semantic,
    search_facts_semantic,
    search_procedural_semantic,
    search_working_memory_semantic,
)
from memory.inference import embed_text

# 500-token hard budget (estimated as chars / 4).
TOKEN_BUDGET = 500
CHARS_BUDGET = TOKEN_BUDGET * 4

# Retrieval thresholds. These are intentionally centralized so wake-up policy
# can be tuned in one place instead of scattering magic numbers across callers.
SESSION_CACHE_HIT_SIMILARITY = 0.96
ENRICHMENT_MIN_SIMILARITY = 0.70
WORKING_MEMORY_MIN_SIMILARITY = 0.72
EPISODIC_MIN_SIMILARITY = 0.72
FACT_MIN_SIMILARITY = 0.76
PROCEDURAL_MIN_SIMILARITY = 0.72

_PROCEDURAL_MARKERS = (
    "how",
    "steps",
    "workflow",
    "procedure",
    "runbook",
    "deploy",
    "setup",
    "install",
    "configure",
    "fix",
    "debug",
)


@dataclass(frozen=True)
class RetrievalWarning:
    """A non-fatal issue encountered while assembling retrieval context."""

    stage: str
    message: str


@dataclass(frozen=True)
class WakeUpContext:
    """Structured retrieval output for the wake-up hook adapter."""

    cache_hit: dict | None
    working_mem: dict | None
    enrichment: list[dict]
    episodic: list[dict]
    facts: list[dict]
    procedural: list[dict]
    warnings: list[RetrievalWarning] = field(default_factory=list)



def build_fact_query(prompt: str) -> str:
    """Build a punctuation-safe FTS query for fact lookup."""
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    keywords = [t for t in tokens if len(t) > 2 or t in {"my"}]

    deduped: list[str] = []
    for token in keywords:
        if token not in deduped:
            deduped.append(token)

    return " OR ".join(deduped)



def _looks_like_procedural_prompt(prompt: str) -> bool:
    """Return True when the prompt appears to ask for a how-to or workflow."""
    lowered = prompt.lower()
    return any(marker in lowered for marker in _PROCEDURAL_MARKERS)



def build_wake_up_injection(context: WakeUpContext) -> str:
    """Assemble the wake-up memory block within the char budget."""
    sections: list[str] = []
    chars_used = 0

    def _normalize(content: str) -> str:
        return re.sub(r"\s+", " ", content).strip()

    def _sentence(content: str) -> str:
        content = _normalize(content)
        if not content:
            return ""
        return content if content.endswith((".", "!", "?")) else f"{content}."

    def _naturalize_fact(content: str) -> str:
        content = _normalize(content)
        match = re.fullmatch(r"([a-z0-9_]+)\.([a-z0-9_]+)\s*=\s*(.+)", content, re.IGNORECASE)
        if match:
            entity, attribute, value = match.groups()
            return _sentence(f"Remembered fact: {entity}.{attribute} = {value}")
        return _sentence(content)

    def _add(block: str) -> bool:
        nonlocal chars_used
        block = _normalize(block)
        if not block:
            return True
        if chars_used + len(block) > CHARS_BUDGET:
            return False
        sections.append(block)
        chars_used += len(block)
        return True

    if context.cache_hit:
        sim = context.cache_hit.get("similarity", 0)
        content = context.cache_hit.get("content", "")
        _add(
            f"Relevant prior session ({sim:.0%} match): {_sentence(content)}"
        )

    if context.working_mem:
        _add(f"Current task context: {_sentence(context.working_mem.get('summary', ''))}")

    if context.enrichment:
        lines = []
        for item in context.enrichment[:2]:
            content = _sentence(item.get("content", ""))
            if content:
                lines.append(f"Related prior session ({item.get('similarity', 0):.0%} match): {content}")
        if lines:
            _add(" ".join(lines))

    if context.episodic:
        lines = []
        for item in context.episodic[:2]:
            title = _normalize(item.get("title", ""))
            abstract = _sentence(item.get("abstract", ""))
            if title and abstract:
                lines.append(f"Recent related episode: {title}. {abstract}")
            elif abstract:
                lines.append(f"Recent related episode: {abstract}")
        if lines:
            _add(" ".join(lines))

    if context.facts:
        lines = []
        for fact in context.facts[:3]:
            content = _normalize(fact.get("content", ""))
            if content:
                lines.append(_naturalize_fact(content))
        if lines:
            _add(" ".join(lines))

    if context.procedural:
        lines = []
        for item in context.procedural[:2]:
            title = _normalize(item.get("title", ""))
            steps = _sentence(item.get("steps", ""))
            if title and steps:
                lines.append(f"Relevant how-to pattern: {title}. {steps}")
            elif steps:
                lines.append(f"Relevant how-to pattern: {steps}")
        if lines:
            _add(" ".join(lines))

    if not sections:
        return ""

    return f"[Memory context: {' '.join(sections)}]"



def retrieve_wake_up_context(
    conn,
    prompt: str,
    *,
    include_working_memory: bool,
    embed_fn=embed_text,
) -> WakeUpContext:
    """Retrieve multi-layer wake-up context for one user prompt."""
    warnings: list[RetrievalWarning] = []

    prompt_vec = embed_fn(prompt)

    cache_hit: dict | None = None
    working_mem: dict | None = None
    enrichment: list[dict] = []
    episodic: list[dict] = []
    facts: list[dict] = []
    procedural: list[dict] = []

    try:
        compacted = search_compacted_sessions(conn, prompt_vec, limit=4)
        if compacted:
            top = compacted[0]
            if top.get("similarity", 0.0) >= SESSION_CACHE_HIT_SIMILARITY:
                cache_hit = top
                try:
                    increment_compacted_hit(conn, top["id"])
                except Exception as exc:
                    warnings.append(RetrievalWarning("cache_hit_metrics", str(exc)))
            enrichment = [
                row for row in compacted
                if row.get("similarity", 0.0) >= ENRICHMENT_MIN_SIMILARITY
                and (cache_hit is None or row.get("id") != cache_hit.get("id"))
            ]
    except Exception as exc:
        warnings.append(RetrievalWarning("compacted_sessions", str(exc)))

    if include_working_memory:
        try:
            working_results = search_working_memory_semantic(conn, prompt_vec, limit=1)
            if working_results and working_results[0].get("similarity", 0.0) >= WORKING_MEMORY_MIN_SIMILARITY:
                working_mem = working_results[0]
        except Exception as exc:
            warnings.append(RetrievalWarning("working_memory", str(exc)))

    try:
        episodic_results = search_episodic_semantic(conn, prompt_vec, limit=3)
        episodic = [
            row for row in episodic_results
            if row.get("similarity", 0.0) >= EPISODIC_MIN_SIMILARITY
        ]
    except Exception as exc:
        warnings.append(RetrievalWarning("episodic", str(exc)))

    try:
        fact_results = search_facts_semantic(conn, prompt_vec, limit=5)
        facts = [
            row for row in fact_results
            if row.get("similarity", 0.0) >= FACT_MIN_SIMILARITY
        ]
    except Exception as exc:
        warnings.append(RetrievalWarning("facts", str(exc)))

    if _looks_like_procedural_prompt(prompt):
        try:
            procedural_results = search_procedural_semantic(conn, prompt_vec, limit=3)
            procedural = [
                row for row in procedural_results
                if row.get("similarity", 0.0) >= PROCEDURAL_MIN_SIMILARITY
            ]
        except Exception as exc:
            warnings.append(RetrievalWarning("procedural", str(exc)))

    return WakeUpContext(
        cache_hit=cache_hit,
        working_mem=working_mem,
        enrichment=enrichment,
        episodic=episodic,
        facts=facts,
        procedural=procedural,
        warnings=warnings,
    )
