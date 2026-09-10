"""Backward-compatibility shim — content moved to memory/episodic/repository.py."""
from memory.episodic.repository import *  # noqa: F401, F403
from memory.episodic.repository import (  # noqa: F401
    build_episodic_semantic_text,
    list_recent_episodic_memories,
    list_session_episodes,
    retrieve_episodic_memories,
    save_extracted_episode,
)
