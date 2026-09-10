from memory.db.schema import bootstrap_db, ensure_schema, init_db, open_db
from memory.db._utils import log_retrieval
from memory.db.sessions import (
    get_latest_session,
    get_session_by_id,
    get_unprocessed_sessions,
    hybrid_search,
    mark_session_processed,
    save_session_compaction,
    search,
    semantic_search,
    upsert_session,
)
from memory.db.facts import (
    delete_fact,
    insert_fact,
    list_facts,
    prune_stale_facts,
    search_facts,
    search_facts_semantic,
    update_fact,
    upsert_fact,
)
from memory.db.episodic import (
    insert_episodic,
    list_recent_episodic,
    prune_stale_episodic,
    search_episodic_semantic,
)
from memory.db.procedural import insert_procedural, search_procedural_semantic
from memory.db.working_memory import get_working_memory, upsert_working_memory
from memory.db.session_memory import search_session_memory_semantic, upsert_session_memory

# scripts/migrate_facts_semantic_text.py imports these from memory.db directly
from memory.vectors import embed, pack_vector

__all__ = [
    "bootstrap_db", "ensure_schema", "init_db", "open_db",
    "log_retrieval",
    "get_latest_session", "get_session_by_id", "get_unprocessed_sessions",
    "hybrid_search", "mark_session_processed", "save_session_compaction",
    "search", "semantic_search", "upsert_session",
    "delete_fact", "insert_fact", "list_facts", "prune_stale_facts",
    "search_facts", "search_facts_semantic", "update_fact", "upsert_fact",
    "insert_episodic", "list_recent_episodic", "prune_stale_episodic", "search_episodic_semantic",
    "insert_procedural", "search_procedural_semantic",
    "get_working_memory", "upsert_working_memory",
    "search_session_memory_semantic", "upsert_session_memory",
    "embed", "pack_vector",
]
