# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before every user prompt. Injects memory context from relevant layers,
# within a hard 500-token budget.
#
# Injection layers (priority order):
#   1. Cache hit  — compacted session ≥ 96% similar to this prompt
#   2. Working memory — rolling task context (first message of session only)
#   3. Enrichment   — compacted sessions 70–95% similar
#   4. Facts         — semantic search over facts table (identity surfaces here)
#   5. Procedural    — how-to patterns (only when prompt has how-to markers)
#
# Gates that skip ALL injection:
#   - DB not found
#   - Embed fails (returns error to avoid blocking the user)
#
# Short/trivial prompts (yes, ok, continue) are NOT gated — they simply find
# nothing in the DB and produce no injection, so zero extra LLM tokens are spent.
#
# Output protocol (JSON to stdout):
#   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "userPrompt": "..."}}
#   {}  → do nothing

import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import init_db, log_retrieval
from memory.logger import activity_log
from memory.debug import enable_debug
from memory.retrieval import (
    build_fact_query as _build_fact_query,
    build_wake_up_injection as _build_injection,
    retrieve_wake_up_context,
)

DB_PATH = os.path.expanduser("~/.memory/memory.db")
LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")


def _log_error(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{ts} ERROR {msg}\n")
    except Exception:
        pass


def _log_info(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{ts} INFO {msg}\n")
    except Exception:
        pass


def _allow() -> None:
    print(json.dumps({}))
    sys.exit(0)


def _first_message_flag(session_id: str) -> str:
    return f"/tmp/memory_first_msg_{session_id}"


def _is_first_message(session_id: str) -> bool:
    flag = _first_message_flag(session_id)
    if not os.path.exists(flag):
        try:
            with open(flag, "w") as f:
                f.write("1")
        except Exception:
            pass
        return True
    return False


def main() -> None:
    enable_debug("wake_up")
    _log_info("=" * 60)
    _log_info("HOOK INVOKED BY CLAUDE CODE")
    try:
        payload = json.load(sys.stdin)
        raw_sid = payload.get("session_id", "unknown")
        # Sanitize session_id before use in file paths to prevent path traversal.
        session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_sid)
        prompt = payload.get("prompt", "").strip()
    except Exception as exc:
        _log_error(f"failed to parse stdin: {exc}")
        _allow()

    _log_info(f"hook invoked, prompt={prompt!r}")

    # Gate 1: no prompt, no injection.
    if not prompt:
        _allow()

    if not os.path.exists(DB_PATH):
        _allow()

    try:
        conn = init_db(DB_PATH)
    except Exception:
        _log_error(f"failed to open DB: {traceback.format_exc()}")
        _allow()

    _log_info(f"running semantic search for prompt={prompt!r}")
    try:
        context = retrieve_wake_up_context(
            conn,
            prompt,
            include_working_memory=_is_first_message(session_id),
        )
    except Exception as exc:
        _log_error(f"embed failed: {exc}")
        conn.close()
        _allow()

    for warning in context.warnings:
        _log_error(f"{warning.stage} failed: {warning.message}")

    # Build the injection block.
    injection = _build_injection(context)

    if not injection:
        _log_info("semantic search complete, no injection produced")
        conn.close()
        _allow()

    updated_prompt = f"{injection}\nUser: {prompt}"
    _log_info(f"semantic search complete, updated prompt={updated_prompt!r}")

    # Log the retrieval for the audit trail.
    try:
        est_tokens = len(injection) // 4
        log_retrieval(conn, "wake_up_injection", prompt, est_tokens)
        activity_log(
            "wake_up",
            "injected",
            session=session_id,
            has_cache_hit=bool(context.cache_hit),
            has_working_mem=bool(context.working_mem),
            enrichment_count=len(context.enrichment),
            facts_count=len(context.facts),
            procedural_count=len(context.procedural),
            est_tokens=est_tokens,
        )
    except Exception:
        pass

    conn.close()

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "userPrompt": updated_prompt,
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
