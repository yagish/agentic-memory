"""Deep retrieval module for wake-up and agent recall policy.

Adapters should call these functions instead of assembling retrieval layers
and ranking policy directly.

Current runtime mode is intentionally facts-only. Wake-up retrieval now ignores
response cache and every non-fact memory layer, and searches only the facts
store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memory.db import search_facts_semantic
from memory.inference import embed_text

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
        content = context.cache_hit.get("response", "")
        _add(
            f"You answered this question before ({sim:.0%} match). Previous answer: {_sentence(content)}"
        )

    if context.facts:
        lines = []
        for fact in context.facts[:3]:
            content = _normalize(fact.get("content", ""))
            if content:
                lines.append(_naturalize_fact(content))
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
    """Retrieve wake-up context.

    Facts-only mode searches only the structured facts store. If no fact is
    found, callers can decide how to surface that miss.
    """
    warnings: list[RetrievalWarning] = []

    # Kept in the function signature for adapter compatibility while non-fact
    # wake-up integrations are intentionally disabled in facts-only mode.
    _ = include_working_memory

    facts: list[dict] = []
    try:
        prompt_vec = embed_fn(prompt)
        facts = search_facts_semantic(conn, prompt_vec, limit=5)
    except Exception as exc:
        warnings.append(RetrievalWarning("facts", str(exc)))

    return WakeUpContext(
        cache_hit=None,
        working_mem=None,
        enrichment=[],
        facts=facts,
        procedural=[],
        warnings=warnings,
    )
