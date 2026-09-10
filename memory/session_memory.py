"""Backward-compatibility shim — content moved to memory/session/extractor.py."""
from memory.session.extractor import *  # noqa: F401, F403
from memory.session.extractor import (  # noqa: F401
    build_session_memory_extraction_prompt,
    discover_session_memory_fixture_cases,
    extract_session_memory_from_session_text,
    log_session_memory_event,
    normalized_contains,
    parse_extracted_session_memory,
    session_memory_to_semantic_core,
)
