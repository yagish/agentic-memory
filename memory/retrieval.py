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
    search_facts,
    search_procedural,
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

    def _add(block: str) -> bool:
        nonlocal chars_used
        if chars_used + len(block) > CHARS_BUDGET:
            return False
        sections.append(block)
        chars_used += len(block)
        return True

    if context.cache_hit:
        sim = context.cache_hit.get("similarity", 0)
        content = context.cache_hit.get("content", "")
        _add(
            f"[Cached Session — {sim:.0%} match]\n"
            f"{content}\n\n"
            "[System] A cached session summary closely matches this prompt. "
            "Respond from this memory, prefix your answer with [From Memory].\n"
        )

    if context.working_mem:
        _add(
            "[Working Memory — current task context]\n"
            f"{context.working_mem['summary']}\n"
        )

    if context.enrichment:
        lines = ["[Relevant Past Work]"]
        for compacted in context.enrichment:
            sim = compacted.get("similarity", 0)
            lines.append(f"({sim:.0%} match)")
            lines.append(compacted.get("content", ""))
        _add("\n".join(lines) + "\n")

    if context.facts:
        lines = ["[Relevant Facts — authoritative]"]
        for fact in context.facts:
            lines.append(f"• {fact.get('content', '')}")
        _add("\n".join(lines) + "\n")

    if context.procedural:
        lines = ["[How-To Patterns]"]
        for procedural in context.procedural:
            lines.append(f"**{procedural.get('title', '')}**")
            lines.append(procedural.get("steps", ""))
        _add("\n".join(lines) + "\n")

    if not sections:
        return ""

    header = (
        "[I have a persistent memory system that retrieved the following context for this prompt. "
        "Please use it to answer my question:]\n\n"
    )
    return header + "\n".join(sections)


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
        fact_query = build_fact_query(prompt)
        if fact_query:
            facts = search_facts(conn, fact_query, limit=5)
    except Exception as exc:
        warnings.append(RetrievalWarning("facts", str(exc)))

    procedural: list[dict] = []
    if is_how_to(prompt):
        try:
            procedural = search_procedural(conn, prompt, min_confidence=0.6, limit=3)
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
