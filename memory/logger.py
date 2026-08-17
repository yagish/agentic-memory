# logger.py — shared structured logging for the memory system.
#
# Two log files live in ~/.memory/:
#   activity.log — one line per memory operation (searches, saves, injections)
#   error.log    — one line per error from any component, with optional traceback
#
# Both functions are deliberately silent on failure — logging must never
# interfere with the main operation (saving, searching, injecting context).

import os
import traceback as tb_module
from datetime import datetime, timezone


# Both logs live alongside the database in ~/.memory/.
_MEMORY_DIR = os.path.expanduser("~/.memory")
ACTIVITY_LOG = os.path.join(_MEMORY_DIR, "activity.log")
ERROR_LOG     = os.path.join(_MEMORY_DIR, "error.log")


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string (e.g. 2026-08-16T10:00:00Z)."""
    # Replace the +00:00 suffix with the shorter 'Z' convention used in log files.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir() -> None:
    """Create ~/.memory/ if it doesn't already exist."""
    # exist_ok=True means no error is raised if the directory already exists.
    os.makedirs(_MEMORY_DIR, exist_ok=True)


def activity_log(component: str, action: str, **kwargs) -> None:
    """
    Append one structured line to ~/.memory/activity.log.

    Each line has the form:
        2026-08-16T10:00:00Z [component] action  key=value key=value ...

    The component and action columns are padded so lines visually align.

    Args:
        component — which part of the system is logging
                    ("wake_up" | "mcp" | "save_hook" | "daemon")
        action    — what happened (e.g. "semantic_search", "upsert_session")
        **kwargs  — arbitrary key=value pairs appended to the line
                    (e.g. query="auth bug", results=3)
    """
    try:
        _ensure_dir()

        # Build the key=value pairs string.
        # repr() wraps strings in quotes so spaces inside values are obvious.
        kv_parts = []
        for k, v in kwargs.items():
            # Use repr() for strings so spaces and special chars are visible.
            if isinstance(v, str):
                # Truncate long strings to keep lines readable.
                display = repr(v[:80] + "…" if len(v) > 80 else v)
            else:
                display = str(v)
            kv_parts.append(f"{k}={display}")

        kv_str = "  ".join(kv_parts)

        # Left-pad component to 10 chars and action to 22 chars for visual alignment.
        line = f"{_now()}  [{component:<10}]  {action:<22}  {kv_str}\n"

        # 'a' mode appends to the file; creates it if it doesn't exist.
        # On POSIX (macOS/Linux), single write() calls to 'a' mode are atomic
        # for lines under ~4KB — no locking needed for our short log lines.
        with open(ACTIVITY_LOG, "a") as f:
            f.write(line)

    except Exception:
        # Never let logging crash the caller — silently give up.
        pass


def error_log(component: str, message: str, exc: BaseException | None = None) -> None:
    """
    Append one error line (plus optional traceback) to ~/.memory/error.log.

    Format:
        2026-08-16T10:00:00Z [component] ERROR  message text
            Traceback (most recent call last):   ← if exc is provided
              ...

    Args:
        component — which part of the system is logging (same values as activity_log)
        message   — short description of the error (one line)
        exc       — if provided, the current exception; traceback is appended
                    indented with 4 spaces so it's visually grouped with the error line
    """
    try:
        _ensure_dir()

        # Header line — mirrors the activity_log format so both files are easy to grep.
        line = f"{_now()}  [{component:<10}]  ERROR  {message}\n"

        if exc is not None:
            # format_exc() returns the full traceback of the CURRENT exception.
            # We indent each line by 4 spaces so it reads as a continuation.
            traceback_text = tb_module.format_exc()
            indented = "".join("    " + l for l in traceback_text.splitlines(keepends=True))
            line += indented

        with open(ERROR_LOG, "a") as f:
            f.write(line)

    except Exception:
        pass
