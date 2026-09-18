"""Session text helpers for memory extraction.

Memory extractors now run against the full raw session transcript rather than
an LLM-compacted rewrite. This avoids hallucinated details in downstream facts,
episodes, procedures, working memory, and session memory.

The compaction helper is retained for optional/debug use, but the daemon no
longer uses compacted text as the source for memory extraction.
"""

from __future__ import annotations

import json

from memory.daemon._core import _COMPACT_INPUT_CHARS, _COMPACT_OUTPUT_CHARS, _daemon_log


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
    """Return the full raw session text with no size limit and no compaction."""
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
    """Return the full raw session text for all extractors.

    The daemon intentionally skips compaction here. Durable memory generation
    should operate on source transcript text, not on an LLM-generated rewrite.
    ``conn`` is accepted for backward compatibility with existing callers.
    """
    del conn
    return _session_text_sample(session)
