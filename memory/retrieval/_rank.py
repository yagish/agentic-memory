from __future__ import annotations

import math
from datetime import datetime, timezone

from memory.retrieval._models import (
    KEYWORD_HIT_BONUS,
    KEYWORD_SCORE_WEIGHT,
    PROJECT_GLOBAL_MISMATCH_PENALTY,
    PROJECT_MATCH_BONUS,
    SESSION_HIT_BONUS,
    SESSION_HIT_SCORE_WEIGHT,
)
from memory.retrieval._project import is_row_allowed_for_project_policy
from memory.retrieval._text import WORD_RE, _memory_text, _normalize_text

_RECENCY_DECAY_DAYS = 90.0


def _filter_by_similarity(rows: list[dict], threshold: float | None) -> list[dict]:
    if threshold is None:
        return rows
    return [row for row in rows if row.get("similarity", 0.0) >= threshold]


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall(_normalize_text(text).lower()))


def _looks_like_missing_memory_episode(item: dict) -> bool:
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


def _is_substantive_episode(item: dict) -> bool:
    details_count = sum(len(item.get(key, []) or []) for key in ("decisions", "outcomes", "follow_ups"))
    return details_count >= 2 and not _looks_like_missing_memory_episode(item)


def _recency_score(item: dict) -> float:
    """Exponential recency score (0–1): 1.0 today, decaying to ~0 over 270 days."""
    timestamp_str = (
        item.get("updated_at")
        or item.get("happened_at")
        or item.get("started_at")
        or ""
    )
    if not timestamp_str:
        return 0.5
    try:
        if isinstance(timestamp_str, str) and timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        ts = datetime.fromisoformat(str(timestamp_str))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_old = max(0.0, (now - ts).total_seconds() / 86400.0)
        return math.exp(-days_old / _RECENCY_DECAY_DAYS)
    except Exception:
        return 0.5


def _project_rank_adjustment(item: dict, kind: str) -> float:
    same_project = item.get("same_project")
    if same_project is True:
        return PROJECT_MATCH_BONUS
    if same_project is False and kind in {"procedural", "facts"}:
        if kind == "facts" and item.get("fact_scope") == "global":
            return 0.0
        return -PROJECT_GLOBAL_MISMATCH_PENALTY
    return 0.0


def _row_score(item: dict, kind: str, prompt_tokens: set[str], *, project_context: dict | None = None) -> float:
    """Composite ranking score: similarity (dominant) + lexical overlap + recency."""
    similarity = float(item.get("similarity", 0.0) or 0.0)
    memory_tokens = _tokenize(_memory_text(item, kind))
    overlap = len(prompt_tokens & memory_tokens) / max(len(prompt_tokens), 1)
    recency = _recency_score(item)
    score = similarity + (0.10 * overlap) + (0.05 * recency)
    if item.get("keyword_hit"):
        score += KEYWORD_HIT_BONUS + (KEYWORD_SCORE_WEIGHT * float(item.get("keyword_score", 0.0) or 0.0))
    if item.get("session_hit"):
        score += SESSION_HIT_BONUS + (SESSION_HIT_SCORE_WEIGHT * float(item.get("session_hit_score", 0.0) or 0.0))
    if project_context:
        score += _project_rank_adjustment(item, kind)
    if kind == "episodic" and _is_substantive_episode(item):
        score += 0.03
    if kind == "episodic" and _looks_like_missing_memory_episode(item):
        score -= 0.25
    return score


def _rank_rows(rows: list[dict], kind: str, prompt: str, *, limit: int, project_context: dict | None = None) -> list[dict]:
    prompt_tokens = _tokenize(prompt)
    filtered = [
        row
        for row in rows
        if not (kind == "episodic" and _looks_like_missing_memory_episode(row))
        and is_row_allowed_for_project_policy(row, kind, project_context)
    ]
    return sorted(
        filtered,
        key=lambda row: (
            float(row.get("rrf_score", 0.0) or 0.0),
            _row_score(row, kind, prompt_tokens, project_context=project_context),
            float(row.get("project_match_score", 0.0) or 0.0),
            float(row.get("session_hit_score", 0.0) or 0.0),
            float(row.get("keyword_score", 0.0) or 0.0),
            float(row.get("similarity", 0.0) or 0.0),
            str(row.get("updated_at", row.get("happened_at", ""))),
        ),
        reverse=True,
    )[:limit]
