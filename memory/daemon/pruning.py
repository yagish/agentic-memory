"""Stale-memory pruning and embedding warm-up helpers."""

from __future__ import annotations

from memory.daemon._core import (
    _EPISODIC_TTL_DAYS,
    _FACT_TTL_DAYS,
    _SESSION_MEMORY_TTL_DAYS,
    _WORKING_MEMORY_TTL_DAYS,
    _daemon_log,
)
from memory.db import (
    prune_stale_episodic,
    prune_stale_facts,
    prune_stale_session_memory,
    prune_stale_working_memory,
)
from memory.utils.logger import activity_log, error_log
from memory.vectors import embed


def _prune_stale_memories(conn) -> None:
    """Remove old memories whose lifecycle is governed by TTL pruning."""
    try:
        deleted_facts = prune_stale_facts(conn, days=_FACT_TTL_DAYS)
        deleted_episodes = prune_stale_episodic(conn, days=_EPISODIC_TTL_DAYS)
        deleted_working_memory = prune_stale_working_memory(
            conn,
            days=_WORKING_MEMORY_TTL_DAYS,
        )
        deleted_session_memory = prune_stale_session_memory(
            conn,
            days=_SESSION_MEMORY_TTL_DAYS,
        )
        if (
            deleted_facts
            or deleted_episodes
            or deleted_working_memory
            or deleted_session_memory
        ):
            _daemon_log(
                "pruned "
                f"{deleted_facts} stale facts, "
                f"{deleted_episodes} stale episodes, "
                f"{deleted_working_memory} stale working-memory snapshots, "
                f"{deleted_session_memory} stale session summaries"
            )
            activity_log(
                "daemon",
                "prune",
                deleted_facts=deleted_facts,
                deleted_episodes=deleted_episodes,
                deleted_working_memory=deleted_working_memory,
                deleted_session_memory=deleted_session_memory,
            )
    except Exception as exc:
        error_log("daemon", f"pruning failed (non-fatal): {exc}", exc=exc)


def _warm_up_embedding() -> None:
    """Load the sentence-transformer model before the first real embed call."""
    try:
        embed("warmup")
        _daemon_log("embedding model warmed up")
    except Exception as exc:
        _daemon_log(f"embedding warm-up failed (non-fatal): {exc}")
