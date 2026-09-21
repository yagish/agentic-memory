"""Shared constants and logging utilities for the daemon package.

No imports from daemon sibling modules — this module is the leaf that all
others import from, so it must remain import-cycle-free.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# --- Paths and intervals ---

DB_PATH = os.path.expanduser("~/.memory/memory.db")
_DAEMON_LOG_PATH = os.path.expanduser("~/.memory/daemon.log")

POLL_INTERVAL = 5 * 60
LONG_POLL_INTERVAL = 30 * 60
CPU_THRESHOLD = 70

# --- TTL and compaction tunables (env-overridable) ---

_FACT_TTL_DAYS = int(os.environ.get("MEMORY_FACT_TTL_DAYS", "180"))
_EPISODIC_TTL_DAYS = int(os.environ.get("MEMORY_EPISODIC_TTL_DAYS", "90"))
_WORKING_MEMORY_TTL_DAYS = int(os.environ.get("MEMORY_WORKING_MEMORY_TTL_DAYS", "7"))
_SESSION_MEMORY_TTL_DAYS = int(os.environ.get("MEMORY_SESSION_MEMORY_TTL_DAYS", "30"))

# How much raw transcript the Ollama summarizer sees (input token budget).
_COMPACT_INPUT_CHARS = int(os.environ.get("MEMORY_COMPACT_INPUT_CHARS", "40000"))

# Target summary length, passed verbatim in the prompt so the model knows
# how long its output should be.
_COMPACT_OUTPUT_CHARS = int(os.environ.get("MEMORY_COMPACT_OUTPUT_CHARS", "5000"))


# --- Logging ---

def _should_write_log_file() -> bool:
    """Return False in test/CI contexts to avoid writing files to disk."""
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    return True


def _daemon_log(message: str) -> None:
    """Append a timestamped line to the daemon log file.

    Silently no-ops in test contexts and on any I/O error.
    """
    if not _should_write_log_file():
        return
    ts = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(_DAEMON_LOG_PATH), exist_ok=True)
        with open(_DAEMON_LOG_PATH, "a") as f:
            f.write(f"{ts} [daemon] {message}\n")
    except Exception:
        pass
