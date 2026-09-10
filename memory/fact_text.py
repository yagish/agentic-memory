"""Backward-compatibility shim — content moved to memory/facts/text.py."""
from memory.facts.text import *  # noqa: F401, F403
from memory.facts.text import (  # noqa: F401
    build_canonical_fact_content,
    build_canonical_fact_from_parts,
    build_semantic_fact_text,
    generate_semantic_fact_text,
)
