"""Stale-memory pruning and embedding warm-up helpers."""

from __future__ import annotations

from memory.daemon._core import _EPISODIC_TTL_DAYS, _FACT_TTL_DAYS, _daemon_log
from memory.db import prune_stale_episodic, prune_stale_facts
from memory.logger import activity_log, error_log
from memory.vectors import embed


def _prune_stale_memories(conn) -> None:
    """Remove old facts and episodic memories to keep the DB lean."""
    try:
        deleted_facts = prune_stale_facts(conn, days=_FACT_TTL_DAYS)
        deleted_episodes = prune_stale_episodic(conn, days=_EPISODIC_TTL_DAYS)
        if deleted_facts or deleted_episodes:
            _daemon_log(
                f"pruned {deleted_facts} stale facts, {deleted_episodes} stale episodes"
            )
            activity_log(
                "daemon",
                "prune",
                deleted_facts=deleted_facts,
                deleted_episodes=deleted_episodes,
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
