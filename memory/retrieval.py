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

import functools
import math
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
    # The prompt's embedding vector, stored here so build_wake_up_injection can
    # classify intent for adaptive budget without re-embedding. None in tests or
    # when called from an older adapter that doesn't pass embeddings.
    prompt_vec: list[float] | None = field(default=None)


WORD_RE = re.compile(r"[a-z0-9_]{2,}")


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


# ── Section 1: Semantic intent detection ──────────────────────────────────
#
# These exemplar phrases represent "the user wants to resume or review recent
# work." We compare the prompt's embedding against these phrases using cosine
# similarity instead of a fragile keyword list.
#
# functools.cache stores the computed vectors for the lifetime of the process.
# The ingest server is long-lived, so these vectors are computed exactly once
# at first call and reused on every subsequent prompt — zero extra embedding
# cost after the first wake-up call.

_RESUME_INTENT_EXEMPLARS = [
    "what were we working on last time",
    "continue where we left off",
    "remind me what we were doing",
    "what happened in the last session",
    "resume our previous work",
    "where did we leave off",
    "catch me up on recent progress",
    "what did we discuss before",
    "pick up from last time",
    "summarize recent work",
    "what is the current state of the project",
    "fill me in on what was done",
]

# Threshold for resume-intent detection. 0.45 is intentionally loose — we
# prefer to fetch recent episodes when the user might be resuming, since the
# episode will simply be ignored if it is not relevant.
_RESUME_SIMILARITY_THRESHOLD = 0.45


@functools.cache
def _get_resume_exemplar_vecs() -> tuple[list[float], ...]:
    """Embed the resume-intent exemplar phrases and cache the result forever.

    Returns a tuple (hashable, so functools.cache works on it) of embedding
    vectors. Called once on the first wake-up invocation; every later call
    returns the same cached tuple instantly.
    """
    # embed_text returns list[float]; we convert to tuple so functools.cache
    # can store the result (lists are not hashable and cannot be cached).
    return tuple(embed_text(phrase) for phrase in _RESUME_INTENT_EXEMPLARS)


def _cosine_sim(a: list[float], b) -> float:
    """Compute cosine similarity between two embedding vectors.

    Cosine similarity measures directional closeness: 1.0 = identical direction
    (semantically very similar), 0.0 = orthogonal (unrelated), -1.0 = opposite.

    We use plain Python arithmetic here rather than numpy because this function
    runs in the fast-path of every prompt and importing numpy adds startup cost
    for the wake-up hook subprocess. The math is trivial for 384-dim vectors.

    Args:
        a — first vector (list or tuple of floats)
        b — second vector (list or tuple of floats)

    Returns:
        Cosine similarity as a float.
    """
    dot = sum(x * y for x, y in zip(a, b))      # element-wise product sum
    mag_a = math.sqrt(sum(x * x for x in a))     # magnitude of a
    mag_b = math.sqrt(sum(y * y for y in b))     # magnitude of b
    if mag_a == 0 or mag_b == 0:
        return 0.0  # zero vector has no direction — treat as dissimilar
    return dot / (mag_a * mag_b)


def _prompt_requests_recent_episode_summary(prompt: str, prompt_vec: list[float] | None = None) -> bool:
    """Return True when the prompt asks to resume or review recent work.

    Uses semantic exemplars when a prompt embedding is available, but keeps a
    lightweight lexical fallback so tests and degraded embedding paths still
    behave sensibly.
    """
    normalized = prompt.lower()
    time_words = {"last", "latest", "recent", "recently", "previous", "before"}
    topic_words = {"work", "working", "worked", "doing", "did", "task", "project", "session", "chat", "conversation", "discuss", "discussed", "talk", "talked"}
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if bool(tokens & time_words) and bool(tokens & topic_words):
        return True

    if prompt_vec is None:
        return False

    try:
        exemplar_vecs = _get_resume_exemplar_vecs()
        return any(
            _cosine_sim(prompt_vec, ex_vec) >= _RESUME_SIMILARITY_THRESHOLD
            for ex_vec in exemplar_vecs
        )
    except Exception:
        return False


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


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall(_normalize_text(text).lower()))


def _join_clean(values: list[str] | tuple[str, ...] | None) -> str:
    return "; ".join(_normalize_text(value) for value in (values or []) if _normalize_text(value))


def _episodic_semantic_text(item: MemoryRow) -> str:
    stored = _normalize_text(str(item.get("semantic_text", "")))
    if stored:
        return stored

    parts = [_normalize_text(item.get("title", "")), _normalize_text(item.get("abstract", ""))]
    participants = _join_clean(item.get("participants", []))
    decisions = _join_clean(item.get("decisions", []))
    outcomes = _join_clean(item.get("outcomes", []))
    follow_ups = _join_clean(item.get("follow_ups", []))
    if participants:
        parts.append(f"Participants: {participants}")
    if decisions:
        parts.append(f"Decisions: {decisions}")
    if outcomes:
        parts.append(f"Outcomes: {outcomes}")
    if follow_ups:
        parts.append(f"Follow-ups: {follow_ups}")
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."


def _procedural_semantic_text(item: MemoryRow) -> str:
    stored = _normalize_text(str(item.get("semantic_text", "")))
    if stored:
        return stored

    parts = [_normalize_text(item.get("title", "")), _normalize_text(item.get("summary", ""))]
    steps = _join_clean(item.get("steps", []))
    triggers = _join_clean(item.get("trigger_phrases", []))
    tools = _join_clean(item.get("tools", []))
    if steps:
        parts.append(f"Steps: {steps}")
    if triggers:
        parts.append(f"Useful for: {triggers}")
    if tools:
        parts.append(f"Tools: {tools}")
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."


def _session_memory_semantic_text(item: MemoryRow) -> str:
    stored = _normalize_text(str(item.get("semantic_text", "")))
    if stored:
        return stored

    parts = [
        _normalize_text(item.get("title", "")),
        _normalize_text(item.get("summary", "")),
        f"Left off at: {_normalize_text(item.get('left_off_at', ''))}",
    ]
    tried = _join_clean(item.get("what_was_tried", []))
    outcomes = _join_clean(item.get("outcomes", []))
    next_steps = _join_clean(item.get("next_steps", []))
    if tried:
        parts.append(f"Tried: {tried}")
    if outcomes:
        parts.append(f"Outcomes: {outcomes}")
    if next_steps:
        parts.append(f"Next session: {next_steps}")
    return ". ".join(part.rstrip(". ") for part in parts if part.strip()) + "."


def _memory_text(item: MemoryRow, kind: str) -> str:
    if kind == "episodic":
        return _episodic_semantic_text(item)
    if kind == "procedural":
        return _procedural_semantic_text(item)
    if kind == "session_memory":
        return _session_memory_semantic_text(item)
    if kind == "facts":
        return _naturalize_fact(str(item.get("content", "")))
    return _normalize_text(str(item))


# ── Section 2: Recency weighting ──────────────────────────────────────────
#
# Without recency, a 6-month-old episode with similarity 0.62 beats a
# 3-day-old episode with similarity 0.60. Recency decay fixes this by adding
# a small bonus that favors newer memories when similarity scores are close.
#
# Decay rate: a memory from today scores 1.0; one from 90 days ago scores
# ~0.37 (= e^-1, the natural exponential half-life). After ~270 days the
# bonus is negligible (<0.05), so truly old memories are not unfairly penalized
# when they're the only matching content.

_RECENCY_DECAY_DAYS = 90.0


def _recency_score(item: MemoryRow) -> float:
    """Compute an exponential recency score (0.0–1.0) for a memory row.

    Returns 1.0 for a memory created today, decaying toward 0.0 for old ones.
    Falls back to 0.5 (neutral) when no parseable timestamp is found.

    Uses the most granular timestamp available: updated_at > happened_at >
    started_at, in that order. All timestamps are expected to be ISO-8601 strings
    (with or without a 'Z' suffix).
    """
    from datetime import datetime, timezone  # local import avoids module-level circular risk

    # Pick the best available timestamp from the memory row
    timestamp_str = (
        item.get("updated_at")
        or item.get("happened_at")
        or item.get("started_at")
        or ""
    )
    if not timestamp_str:
        return 0.5  # no timestamp available — neutral recency, no penalty or bonus

    try:
        # Normalize the "Z" UTC suffix to "+00:00" which fromisoformat() understands.
        # Python < 3.11 does not parse the trailing "Z" natively.
        if isinstance(timestamp_str, str) and timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        ts = datetime.fromisoformat(str(timestamp_str))
        # If the stored timestamp has no timezone info, assume UTC
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_old = max(0.0, (now - ts).total_seconds() / 86400.0)
        # Exponential decay: e^(-days_old / DECAY_DAYS)
        return math.exp(-days_old / _RECENCY_DECAY_DAYS)
    except Exception:
        return 0.5  # any parse error → neutral


def _row_score(item: MemoryRow, kind: str, prompt_tokens: set[str]) -> float:
    """Compute a composite ranking score for one memory row.

    Combines three signals:
    - similarity (0–1): cosine similarity from the embedding search — dominant signal
    - overlap (0–1): fraction of prompt tokens that appear in the memory text — lexical bonus
    - recency (0–1): exponential decay based on memory age — tie-breaker for equal similarity

    Weights: similarity is ~10× more important than overlap, and ~20× more
    important than recency. Recency only matters when similarity scores are close.
    """
    similarity = float(item.get("similarity", 0.0) or 0.0)
    # Lexical overlap: gives a small bonus when specific terms (e.g. project names,
    # file names) appear in both the prompt and the memory text
    memory_tokens = _tokenize(_memory_text(item, kind))
    overlap = len(prompt_tokens & memory_tokens) / max(len(prompt_tokens), 1)
    # Recency bonus: recent memories rank slightly higher when similarity is equal
    recency = _recency_score(item)
    score = similarity + (0.10 * overlap) + (0.05 * recency)
    if kind == "episodic" and _is_substantive_episode(item):
        score += 0.03  # small bonus for well-structured episodes
    if kind == "episodic" and _looks_like_missing_memory_episode(item):
        score -= 0.25  # large penalty for LLM-generated "no memory" placeholders
    return score


def _rank_rows(rows: list[MemoryRow], kind: str, prompt: str, *, limit: int) -> list[MemoryRow]:
    prompt_tokens = _tokenize(prompt)
    filtered: list[MemoryRow] = []
    for row in rows:
        if kind == "episodic" and _looks_like_missing_memory_episode(row):
            continue
        filtered.append(row)
    return sorted(
        filtered,
        key=lambda row: (
            _row_score(row, kind, prompt_tokens),
            float(row.get("similarity", 0.0) or 0.0),
            str(row.get("updated_at", row.get("happened_at", ""))),
        ),
        reverse=True,
    )[:limit]


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


# ── Section 3: Adaptive token budget ──────────────────────────────────────
#
# A flat 500-token budget is wasteful for simple questions and insufficient for
# "resume where we left off" requests that benefit from the full session handoff.
# We classify the prompt intent semantically and pick an appropriate budget.
#
# Budget tiers:
#   quick  → ~200 tokens: single-answer questions don't need much context
#   task   → ~500 tokens: standard continuation (previous default)
#   resume → ~1500 tokens: full handoff — session summary, last steps, next steps

_QUICK_INTENT_EXEMPLARS = [
    "what is",
    "define this",
    "what does this mean",
    "quick question",
    "just tell me",
    "what language is this written in",
    "how do you spell",
    "what is the syntax for",
    "can you explain briefly",
    "in one sentence",
]

# Higher threshold for "quick" intent — we want confidence before we truncate context.
_QUICK_SIMILARITY_THRESHOLD = 0.55

# Budget in characters (tokens * 4) for each intent class
_BUDGET_BY_INTENT = {
    "quick": 200 * 4,    # ~200 tokens = 800 chars
    "task": 500 * 4,     # ~500 tokens = 2000 chars (the previous hardcoded default)
    "resume": 1500 * 4,  # ~1500 tokens = 6000 chars — full handoff context
}


@functools.cache
def _get_quick_exemplar_vecs() -> tuple[list[float], ...]:
    """Embed the quick-question exemplar phrases and cache the result forever.

    Works the same way as _get_resume_exemplar_vecs — computed once on first
    call, reused on every subsequent call in the same process.
    """
    return tuple(embed_text(phrase) for phrase in _QUICK_INTENT_EXEMPLARS)


def _classify_prompt_intent(prompt_vec: list[float]) -> str:
    """Classify the prompt as 'resume', 'quick', or 'task' for budget selection.

    Uses semantic similarity against two sets of exemplar phrases. 'resume' is
    checked before 'quick' because a resume request benefits more from a larger
    budget than a quick question suffers from a smaller one.

    Args:
        prompt_vec — the prompt's embedding vector (already computed during retrieval)

    Returns:
        One of: 'resume', 'quick', 'task' (default when neither matches).
    """
    try:
        # Check resume intent first — it overrides quick even if both match
        resume_vecs = _get_resume_exemplar_vecs()
        if any(_cosine_sim(prompt_vec, v) >= _RESUME_SIMILARITY_THRESHOLD for v in resume_vecs):
            return "resume"
        # Check if this is a short, simple question
        quick_vecs = _get_quick_exemplar_vecs()
        if any(_cosine_sim(prompt_vec, v) >= _QUICK_SIMILARITY_THRESHOLD for v in quick_vecs):
            return "quick"
    except Exception:
        pass  # exemplar embedding failed — fall through to default
    return "task"


def build_wake_up_injection(context: WakeUpContext) -> str:
    """Assemble the wake-up memory block within an adaptive char budget.

    This path is intentionally deterministic and model-free. The only "smart"
    step is the intent classification, which is pure cosine similarity arithmetic
    against pre-embedded exemplar phrases — no LLM call.

    When context.prompt_vec is available (set by retrieve_wake_up_context),
    classifies the prompt intent and picks the right budget:
    - 'resume' requests get 1500 tokens — enough for a full session handoff
    - 'quick' questions get 200 tokens — avoids padding simple answers with noise
    - 'task' (default) gets 500 tokens — the original hardcoded budget

    Falls back to the standard 500-token budget when prompt_vec is None
    (e.g. in tests or when called from an older adapter).
    """
    if context.prompt_vec is not None:
        intent = _classify_prompt_intent(context.prompt_vec)
        char_budget = _BUDGET_BY_INTENT[intent]
    else:
        char_budget = CHARS_BUDGET  # standard fallback — matches the old hardcoded value

    body = _fit_context_to_budget(_context_sections(context), char_budget=char_budget)
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
        filtered = [item for item in substantive if not _looks_like_missing_memory_episode(item)]
        return filtered[:EPISODIC_FETCH_LIMIT]
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

    Retrieval is intentionally broad: durable memory types are searched for every
    prompt with one shared query embedding. Narrowing happens after candidate
    search via per-type ranking and tight result caps.
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
        _retrieve_session_memory(
            conn,
            prompt,
            prompt_vec,
            embed_fn,
            warnings,
            session_id=session_id,
        ),
        "session_memory",
        prompt,
        limit=SESSION_MEMORY_LIMIT,
    )

    # Fallback: if semantic search found no episodes or session memories, check
    # whether the prompt is semantically asking to resume/review recent work.
    # We pass the already-computed prompt_vec — no extra embedding call needed.
    if not episodic and not session_memory and _prompt_requests_recent_episode_summary(prompt, prompt_vec):
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
        # Store the prompt's embedding vector so build_wake_up_injection can
        # classify prompt intent for adaptive budget without re-embedding.
        prompt_vec=prompt_vec,
    )
