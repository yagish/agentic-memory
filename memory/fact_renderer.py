"""Backward-compatibility shim — content moved to memory/facts/renderer.py."""
from memory.facts.renderer import *  # noqa: F401, F403
from memory.facts.renderer import (  # noqa: F401
    _canonical_fallback,
    _normalize_rendered_answer,
    build_fact_render_prompt,
    render_fact_answer,
)
