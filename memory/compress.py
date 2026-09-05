"""Removed compression module.

Compression belonged to the old memory architecture and is no longer supported
in the simplified sessions/facts/episodes-only system.
"""

from __future__ import annotations


def compress_memory(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("memory compression was removed from the simplified memory system")
