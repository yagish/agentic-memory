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
_RENDER_MAX_CHARS = 240
_RENDER_MODEL = os.environ.get("MEMORY_FACT_RENDER_MODEL")


def _canonical_fallback(fact_contents: list[str]) -> str:
    """Render a deterministic non-LLM fallback from retrieved fact strings."""
    cleaned = [fact.strip() for fact in fact_contents if fact.strip()]
    if not cleaned:
        return "I don't have that fact stored."
    return "\n".join(cleaned)



def build_fact_render_prompt(*, user_prompt: str, fact_contents: list[str]) -> str:
    """Build the minimal prompt used to verbalize retrieved facts."""
    rendered_facts = "\n".join(f"- {fact.strip()}" for fact in fact_contents if fact.strip())
    return f"""Answer the user's question using only these facts.
Write one short answer in second person.
Answer only what the user asked.
If the question has multiple parts, answer every part supported by the facts.
Ignore unrelated facts.
Do not guess missing facts.
Do not add caveats or extra commentary.

Question: {user_prompt}
Facts:
{rendered_facts}
Answer:"""


def _normalize_rendered_answer(text: str) -> str:
    """Clean a renderer response into one short plain-text sentence."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()

    first_line = cleaned.splitlines()[0].strip()
    if not first_line:
        return ""

    return first_line[:_RENDER_MAX_CHARS].strip()


def render_fact_answer(user_prompt: str, fact_contents: list[str]) -> str:
    """Render a natural-language answer for retrieved facts.

    If the local renderer fails, or returns an obviously wrong perspective, the
    canonical stored facts are returned so the caller still has a deterministic
    fallback.
    """
    canonical = _canonical_fallback(fact_contents)
    if canonical == "I don't have that fact stored.":
        return canonical

    try:
        result = generate_text(
            GenerationRequest(
                prompt=build_fact_render_prompt(
                    user_prompt=user_prompt,
                    fact_contents=fact_contents,
                ),
                model=_RENDER_MODEL,
                timeout_seconds=_RENDER_TIMEOUT_SECONDS,
                temperature=0.0,
            )
        )
    except InferenceError:
        return canonical

    rendered = _normalize_rendered_answer(result.text)
    if not rendered:
        return canonical
    return rendered
