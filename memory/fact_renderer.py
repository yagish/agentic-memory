"""Natural-language rendering for retrieved facts.

This module keeps presentation logic out of the wake-up hook. Retrieval returns
canonical stored fact strings such as ``user.name = Yagish``. When a fact hit
needs to be shown to the user, we ask the local LLM to render a short answer
using only the retrieved facts.
"""

from __future__ import annotations

import os

from memory.inference import (
    GenerationRequest,
    InferenceError,
    generate_text,
)


_RENDER_TIMEOUT_SECONDS = int(os.environ.get("MEMORY_FACT_RENDER_TIMEOUT_SECONDS", "10"))
_RENDER_MAX_CHARS = int(os.environ.get("MEMORY_FACT_RENDER_MAX_CHARS", "240"))


def _canonical_fallback(fact_contents: list[str]) -> str:
    """Render a deterministic non-LLM fallback from retrieved fact strings."""
    cleaned = [fact.strip() for fact in fact_contents if fact.strip()]
    if not cleaned:
        return "I don't have that fact stored."
    return "\n".join(cleaned)



def build_fact_render_prompt(*, user_prompt: str, fact_contents: list[str]) -> str:
    """Build the prompt used to verbalize retrieved facts conservatively."""
    rendered_facts = "\n".join(f"- {fact.strip()}" for fact in fact_contents if fact.strip())
    return f"""Answer the user's question using only these facts.
Write one short plain-text answer in second person.
Answer only what the user asked.
If the question has multiple parts, answer every part supported by the facts.
Ignore unrelated facts.
Do not guess missing facts.
Do not add caveats or extra commentary.
If the facts do not directly answer the question, return exactly: INSUFFICIENT_FACTS
If the facts are only loosely related to the question, return exactly: INSUFFICIENT_FACTS

Question: {user_prompt}
Facts:
{rendered_facts}
Answer:"""


def _normalize_rendered_answer(text: str) -> str:
    """Clean a renderer response into one short plain-text answer."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()

    cleaned = " ".join(line.strip() for line in cleaned.splitlines() if line.strip())
    return cleaned.strip()


def build_fact_compaction_prompt(*, answer: str, max_chars: int) -> str:
    """Ask the local model to compress an overlong fact answer cleanly."""
    return f"""Rewrite this answer so it stays natural and complete.
Keep all supported facts, remove filler, and keep the same meaning.
Return plain text only.
Stay under {max_chars} characters.

Answer:
{answer}

Compressed answer:"""


def _compact_rendered_answer(answer: str, *, max_chars: int) -> str:
    """Compress an overlong renderer answer instead of cutting it mid-thought."""
    normalized = _normalize_rendered_answer(answer)
    if not normalized or normalized == "INSUFFICIENT_FACTS":
        return normalized
    if len(normalized) <= max_chars:
        return normalized

    try:
        result = generate_text(
            GenerationRequest(
                prompt=build_fact_compaction_prompt(answer=normalized, max_chars=max_chars),
                model=None,
                timeout_seconds=_RENDER_TIMEOUT_SECONDS,
                temperature=0.0,
            )
        )
    except InferenceError:
        return normalized[:max_chars].rstrip()

    compacted = _normalize_rendered_answer(result.text)
    if not compacted:
        return normalized[:max_chars].rstrip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[:max_chars].rstrip()


def render_fact_answer(user_prompt: str, fact_contents: list[str]) -> str:
    """Render a natural-language answer for retrieved facts.

    Returns an empty string when the facts do not directly answer the question
    or when the local renderer fails. This prevents weak fact hits from
    hijacking unrelated prompts.
    """
    canonical = _canonical_fallback(fact_contents)
    if canonical == "I don't have that fact stored.":
        return ""

    try:
        result = generate_text(
            GenerationRequest(
                prompt=build_fact_render_prompt(
                    user_prompt=user_prompt,
                    fact_contents=fact_contents,
                ),
                model=None,
                timeout_seconds=_RENDER_TIMEOUT_SECONDS,
                temperature=0.0,
            )
        )
    except InferenceError:
        return ""

    rendered = _normalize_rendered_answer(result.text)
    if "INSUFFICIENT_FACTS" in rendered:
        rendered = rendered.replace("INSUFFICIENT_FACTS", "").strip(" -:\n\t")
    if not rendered:
        return ""
    return _compact_rendered_answer(rendered, max_chars=_RENDER_MAX_CHARS)
