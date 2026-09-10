from __future__ import annotations

from memory.retrieval._models import CHARS_BUDGET, MemoryRow, WakeUpContext
from memory.retrieval._text import _as_sentence, _naturalize_fact, _normalize_text
from memory.retrieval._intent import _BUDGET_BY_INTENT, _classify_prompt_intent


def _format_episode(item: MemoryRow) -> str:
    title = _normalize_text(item.get("title", ""))
    abstract = _as_sentence(item.get("abstract", ""))
    decisions = [_normalize_text(v) for v in item.get("decisions", []) if _normalize_text(v)]
    outcomes = [_normalize_text(v) for v in item.get("outcomes", []) if _normalize_text(v)]
    follow_ups = [_normalize_text(v) for v in item.get("follow_ups", []) if _normalize_text(v)]
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
    active_tasks = [_normalize_text(v) for v in item.get("active_tasks", []) if _normalize_text(v)]
    constraints = [_normalize_text(v) for v in item.get("constraints", []) if _normalize_text(v)]
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
        steps = [_normalize_text(v) for v in item.get("steps", []) if _normalize_text(v)]
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
        next_steps = [_normalize_text(v) for v in item.get("next_steps", []) if _normalize_text(v)]
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
    """Assemble the wake-up memory block within an adaptive char budget.

    Uses resume-intent detection to expand the budget to 1500 tokens for
    session-handoff prompts; falls back to 500 tokens for regular tasks.
    """
    if context.prompt_vec is not None:
        intent = _classify_prompt_intent(context.prompt_vec)
        char_budget = _BUDGET_BY_INTENT[intent]
    else:
        char_budget = CHARS_BUDGET
    body = _fit_context_to_budget(_context_sections(context), char_budget=char_budget)
    if not body:
        return ""
    return _build_context_envelope(body)
