from __future__ import annotations

import functools
import math

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

_RESUME_SIMILARITY_THRESHOLD = 0.45

# Budget in characters (tokens * 4) for each intent class
_BUDGET_BY_INTENT = {
    "task": 500 * 4,     # ~500 tokens = 2000 chars
    "resume": 1500 * 4,  # ~1500 tokens = 6000 chars — full handoff context
}


@functools.cache
def _get_resume_exemplar_vecs() -> tuple[list[float], ...]:
    """Embed resume-intent exemplars once and cache for the process lifetime."""
    from memory.inference import embed_text
    return tuple(embed_text(phrase) for phrase in _RESUME_INTENT_EXEMPLARS)


def _cosine_sim(a: list[float], b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _prompt_requests_recent_episode_summary(prompt_vec: list[float]) -> bool:
    """Return True when the prompt semantically asks to resume or review recent work."""
    try:
        exemplar_vecs = _get_resume_exemplar_vecs()
        return any(
            _cosine_sim(prompt_vec, ex_vec) >= _RESUME_SIMILARITY_THRESHOLD
            for ex_vec in exemplar_vecs
        )
    except Exception:
        return False


def _classify_prompt_intent(prompt_vec: list[float]) -> str:
    """Classify the prompt as 'resume' or 'task' for budget selection."""
    try:
        resume_vecs = _get_resume_exemplar_vecs()
        if any(_cosine_sim(prompt_vec, v) >= _RESUME_SIMILARITY_THRESHOLD for v in resume_vecs):
            return "resume"
    except Exception:
        pass
    return "task"
