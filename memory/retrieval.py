"""Deep retrieval module for wake-up and agent recall policy.

Adapters should call these functions instead of assembling retrieval layers
and ranking policy directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memory.db import (
    get_active_working_memory,
    increment_compacted_hit,
    search_compacted_sessions,
    search_facts_semantic,
    search_procedural_semantic,
)
from memory.inference import embed_text

# Semantic thresholds for compacted-session retrieval.
SEMANTIC_CACHE_THRESHOLD = 0.96
ENRICHMENT_THRESHOLD = 0.70
WORKING_MEM_THRESHOLD = 0.50

# How-to markers — enable procedural memory injection when present.
HOW_TO_MARKERS = {
    "how", "steps", "step by", "best way", "should i",
    "approach", "workflow", "procedure", "guide", "tutorial",
}

# 500-token hard budget (estimated as chars / 4).
TOKEN_BUDGET = 500
CHARS_BUDGET = TOKEN_BUDGET * 4


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
    facts: list[dict]
    procedural: list[dict]
    warnings: list[RetrievalWarning] = field(default_factory=list)


def is_how_to(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(marker in lowered for marker in HOW_TO_MARKERS)


def build_fact_query(prompt: str) -> str:
    """Build a punctuation-safe FTS query for fact lookup."""
    tokens = re.findall(r"[a-z0-9]+", prompt.lower())
    keywords = [t for t in tokens if len(t) > 2 or t in {"my"}]

    deduped: list[str] = []
    for token in keywords:
        if token not in deduped:
            deduped.append(token)

    return " OR ".join(deduped)


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
        content = _sentence(content)
        if content.startswith("User's "):
            return "The user's " + content[len("User's "):]
        return content

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
            f"You answered this question before ({sim:.0%} match). Previous answer: {_sentence(content)}"
        )

    if context.working_mem:
        _add(
            f"Current task context: {context.working_mem['summary']}"
        )

    if context.enrichment:
        lines = []
        for compacted in context.enrichment:
            sim = compacted.get("similarity", 0)
            lines.append(f"Relevant prior conversation ({sim:.0%} match): {_sentence(compacted.get('content', ''))}")
        _add(" ".join(lines))

    if context.facts:
        top_fact = _normalize(context.facts[0].get("content", ""))
        if top_fact:
            _add(_naturalize_fact(top_fact))

    if context.procedural:
        lines = []
        for procedural in context.procedural:
            title = procedural.get("title", "")
            steps = procedural.get("steps", "")
            lines.append(f"Relevant how-to pattern: {_sentence(f'{title}: {steps}')}")
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
    """Retrieve wake-up context across compacted, working, fact, and procedural layers."""
    prompt_vec = embed_fn(prompt)
    warnings: list[RetrievalWarning] = []

    cache_hit: dict | None = None
    enrichment: list[dict] = []
    cluster_id_for_wm: str | None = None

    try:
        results = search_compacted_sessions(conn, prompt_vec, limit=5)
        for compacted in results:
            sim = compacted.get("similarity", 0)
            if cluster_id_for_wm is None and sim >= WORKING_MEM_THRESHOLD:
                cluster_id_for_wm = compacted.get("cluster_id")
            if sim >= SEMANTIC_CACHE_THRESHOLD:
                cache_hit = compacted
                try:
                    increment_compacted_hit(conn, compacted["id"])
                except Exception as exc:
                    warnings.append(RetrievalWarning("compacted_hit_count", str(exc)))
                break
            if ENRICHMENT_THRESHOLD <= sim < SEMANTIC_CACHE_THRESHOLD:
                enrichment.append(compacted)
    except Exception as exc:
        warnings.append(RetrievalWarning("compacted_search", str(exc)))

    working_mem: dict | None = None
    if include_working_memory and cluster_id_for_wm:
        try:
            working_mem = get_active_working_memory(conn, cluster_id_for_wm)
        except Exception as exc:
            warnings.append(RetrievalWarning("working_memory", str(exc)))

    facts: list[dict] = []
    try:
        # Use semantic search to find facts similar to the prompt.
        # This handles queries like "what everyone calls me" → "User's name is Yagish"
        facts = search_facts_semantic(conn, prompt_vec, limit=5)
    except Exception as exc:
        warnings.append(RetrievalWarning("facts", str(exc)))

    procedural: list[dict] = []
    if is_how_to(prompt):
        try:
            # Use semantic search for procedural patterns.
            procedural = search_procedural_semantic(conn, prompt_vec, min_confidence=0.6, limit=3)
        except Exception as exc:
            warnings.append(RetrievalWarning("procedural", str(exc)))

    return WakeUpContext(
        cache_hit=cache_hit,
        working_mem=working_mem,
        enrichment=enrichment,
        facts=facts,
        procedural=procedural,
        warnings=warnings,
    )
