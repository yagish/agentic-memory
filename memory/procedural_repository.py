"""Backward-compatibility shim — content moved to memory/procedural/repository.py."""
from memory.procedural.repository import *  # noqa: F401, F403
from memory.procedural.repository import (  # noqa: F401
    build_procedural_semantic_text,
    list_session_procedures,
    retrieve_procedural_memories,
    save_extracted_procedure,
)
