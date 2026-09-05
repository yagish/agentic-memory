"""Removed consolidation module.

Session summarisation/pruning was part of the old multi-layer memory design and
is no longer supported now that the system retains only sessions, facts, and
episodes.
"""

from __future__ import annotations


def consolidate_old_sessions(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("consolidation was removed from the simplified memory system")


def prune_old_transcripts(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("transcript pruning was removed from the simplified memory system")
