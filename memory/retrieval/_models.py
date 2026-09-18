from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MemoryRow = dict[str, Any]

TOKEN_BUDGET = 500
CHARS_BUDGET = TOKEN_BUDGET * 4

EPISODIC_MIN_SIMILARITY = 0.58
EPISODIC_FETCH_LIMIT = 8
EPISODIC_LIMIT = 2
EPISODIC_RECENT_LIMIT = 5
FACT_MIN_SIMILARITY = 0.38
FACT_FETCH_LIMIT = 5
FACT_LIMIT = 3
PROCEDURAL_MIN_SIMILARITY = 0.58
PROCEDURAL_FETCH_LIMIT = 4
PROCEDURAL_LIMIT = 2
SESSION_MEMORY_MIN_SIMILARITY = 0.60
SESSION_MEMORY_FETCH_LIMIT = 4
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
    prompt_vec: list[float] | None = field(default=None)
    timings: dict[str, float] = field(default_factory=dict)
