"""Session compaction helpers — summarize raw transcripts via Ollama.

Every session is compacted (short or long) so all five extractors receive a
consistent, model-shaped summary rather than raw turns.  The compacted text is
stored in the DB and reused on subsequent daemon cycles.
"""

from __future__ import annotations

import json

from memory.daemon._core import (
    _COMPACT_INPUT_CHARS,
    _COMPACT_OUTPUT_CHARS,
    _daemon_log,
)
from memory.db import save_session_compaction


def _build_session_text(turns: list[dict], max_chars: int | None = None) -> str:
    """Concatenate turn content into a single readable text block."""
    lines: list[str] = []
    chars = 0
    for turn in turns:
        content = turn.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        role = turn.get("role", "")
        line = f"{role}: {content}"
        if max_chars is not None and chars + len(line) > max_chars:
            break
        lines.append(line)
        chars += len(line)
    return "\n".join(lines)


def _session_text_sample(session: dict) -> str:
    """Return the full raw session text with no size limit and no compaction.

    Used when extractor functions are called directly (e.g. from tests) without
    a pre-built text_sample argument.  The normal daemon path goes through
    _get_or_compact_session_text instead.
    """
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []
    return _build_session_text(turns)


def _compact_session_text(full_text: str) -> str:
    """Summarize a long session transcript via Ollama.

    Passes up to _COMPACT_INPUT_CHARS of the raw text to the model and asks for
    a dense summary under _COMPACT_OUTPUT_CHARS characters.  The summary is saved
    to the DB by the caller so the model is only called once per session.

    Raises InferenceError if Ollama is unavailable — callers decide how to handle.
    """
    from memory.llm.inference import GenerationRequest, generate_text

    input_text = full_text[:_COMPACT_INPUT_CHARS]

    prompt = (
        "You are a session summarizer. Condense the following conversation transcript "
        "into a concise but complete summary.\n"
        "Include:\n"
        "- All key facts (names, settings, preferences, technical details)\n"
        "- Decisions made and their rationale\n"
        "- Steps taken or discussed\n"
        "- Outcomes reached and open questions\n"
        f"Your output must be under {_COMPACT_OUTPUT_CHARS} characters. "
        "Output only the summary — no preamble, no closing remark.\n\n"
        "TRANSCRIPT:\n" + input_text
    )

    result = generate_text(GenerationRequest(prompt=prompt, timeout_seconds=120))
    _daemon_log(f"compacted session: {len(full_text)} → {len(result.text)} chars")
    return result.text


def _get_or_compact_session_text(conn, session: dict) -> str:
    """Return the compacted text for all extractors, creating it if absent.

    - Already compacted: return the stored compacted_text directly.
    - All others: call Ollama, save the result to the DB, return the summary.

    If Ollama fails, the error propagates — the session stays unprocessed and
    will be retried on the next daemon cycle.
    """
    cached = session.get("compacted_text")
    if cached:
        _daemon_log(f"reusing stored compaction for {session['session_id']}")
        return cached

    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []

    full_text = _build_session_text(turns)
    compacted = _compact_session_text(full_text)
    save_session_compaction(conn, session["session_id"], compacted)
    return compacted
