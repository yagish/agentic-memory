"""Backward-compatibility shim — content moved to memory/facts/repository.py."""
from memory.facts.repository import *  # noqa: F401, F403
from memory.facts.repository import (  # noqa: F401
    build_fact_content,
    build_fact_semantic_content,
    build_fact_tags,
    list_session_facts,
    save_extracted_facts,
)
