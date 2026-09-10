from __future__ import annotations

import re

from memory.retrieval._models import RetrievalWarning

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


def _join_clean(values: list[str] | tuple[str, ...] | None) -> str:
    return "; ".join(_normalize_text(value) for value in (values or []) if _normalize_text(value))


def _episodic_semantic_text(item: dict) -> str:
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


def _procedural_semantic_text(item: dict) -> str:
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


def _session_memory_semantic_text(item: dict) -> str:
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


def _memory_text(item: dict, kind: str) -> str:
    if kind == "episodic":
        return _episodic_semantic_text(item)
    if kind == "procedural":
        return _procedural_semantic_text(item)
    if kind == "session_memory":
        return _session_memory_semantic_text(item)
    if kind == "facts":
        return _naturalize_fact(str(item.get("content", "")))
    return _normalize_text(str(item))
