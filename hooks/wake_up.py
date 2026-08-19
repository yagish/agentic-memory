# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before every user prompt. Injects memory context from relevant layers,
# within a hard 500-token budget.
#
# Injection layers (priority order):
#   1. Cache hit  — compacted session ≥ 96% similar to this prompt
#   2. Working memory — rolling task context (first message of session only)
#   3. Enrichment   — compacted sessions 70–95% similar
#   4. Facts         — FTS5 search over facts table (identity surfaces here)
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
#   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "userPromptSuffix": "..."}}
#   {}  → do nothing

import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    init_db,
    embed,
    search_compacted_sessions,
    increment_compacted_hit,
    get_active_working_memory,
    search_facts,
    search_procedural,
    log_retrieval,
)
from memory.logger import activity_log, error_log

DB_PATH  = os.path.expanduser("~/.memory/memory.db")
LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")

# Semantic thresholds for compacted-session retrieval.
SEMANTIC_CACHE_THRESHOLD = 0.96   # full cache-hit injection
ENRICHMENT_THRESHOLD     = 0.70   # context enrichment injection
WORKING_MEM_THRESHOLD    = 0.50   # loose match for working memory lookup

# How-to markers — enable procedural memory injection when present.
HOW_TO_MARKERS = {
    "how", "steps", "step by", "best way", "should i",
    "approach", "workflow", "procedure", "guide", "tutorial",
}

# 500-token hard budget (estimated as chars / 4).
TOKEN_BUDGET = 500
CHARS_BUDGET = TOKEN_BUDGET * 4


def _log_error(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(LOG_PATH, "a") as f:
            f.write(f"{ts} ERROR {msg}\n")
    except Exception:
        pass


def _allow() -> None:
    print(json.dumps({}))
    sys.exit(0)


def _is_how_to(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(marker in lowered for marker in HOW_TO_MARKERS)


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


def _build_injection(
    cache_hit:   dict | None,
    working_mem: dict | None,
    enrichment:  list[dict],
    facts:       list[dict],
    procedural:  list[dict],
) -> str:
    """Assemble the memory block, enforcing the char budget."""
    sections: list[str] = []
    chars_used = 0

    def _add(block: str) -> bool:
        nonlocal chars_used
        if chars_used + len(block) > CHARS_BUDGET:
            return False
        sections.append(block)
        chars_used += len(block)
        return True

    # 1. Cache hit — highest priority.
    if cache_hit:
        sim = cache_hit.get("similarity", 0)
        content = cache_hit.get("content", "")
        block = (
            f"[Cached Session — {sim:.0%} match]\n"
            f"{content}\n\n"
            "[System] A cached session summary closely matches this prompt. "
            "Respond from this memory, prefix your answer with [From Memory].\n"
        )
        _add(block)

    # 2. Working memory — task context for this session start.
    if working_mem:
        block = (
            "[Working Memory — current task context]\n"
            f"{working_mem['summary']}\n"
        )
        _add(block)

    # 3. Enrichment context (70–95% similarity).
    if enrichment:
        lines = ["[Relevant Past Work]"]
        for cs in enrichment:
            sim = cs.get("similarity", 0)
            lines.append(f"({sim:.0%} match)")
            lines.append(cs.get("content", ""))
        block = "\n".join(lines) + "\n"
        _add(block)

    # 4. Relevant facts (identity surfaces here when the prompt asks about the user).
    if facts:
        lines = ["[Relevant Facts]"]
        for f in facts:
            lines.append(f"• {f.get('content', '')}")
        block = "\n".join(lines) + "\n"
        _add(block)

    # 5. Procedural memory.
    if procedural:
        lines = ["[How-To Patterns]"]
        for p in procedural:
            lines.append(f"**{p.get('title', '')}**")
            lines.append(p.get("steps", ""))
        block = "\n".join(lines) + "\n"
        _add(block)

    if not sections:
        return ""

    return "=== MEMORY ===\n\n" + "\n".join(sections) + "\n=== END MEMORY ==="


def main() -> None:
    try:
        payload    = json.load(sys.stdin)
        raw_sid    = payload.get("session_id", "unknown")
        # Sanitize session_id before use in file paths to prevent path traversal.
        session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_sid)
        prompt     = payload.get("prompt", "").strip()
    except Exception as exc:
        _log_error(f"failed to parse stdin: {exc}")
        _allow()

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

    # Embed the prompt for semantic retrieval.
    try:
        prompt_vec = embed(prompt)
    except Exception as exc:
        _log_error(f"embed failed: {exc}")
        conn.close()
        _allow()

    # --- Semantic retrieval from compacted_sessions ---
    cache_hit:  dict | None = None
    enrichment: list[dict]  = []
    cluster_id_for_wm: str | None = None

    try:
        results = search_compacted_sessions(conn, prompt_vec, limit=5)
        for cs in results:
            sim = cs.get("similarity", 0)
            # Record the cluster_id of the best match for working memory lookup.
            if cluster_id_for_wm is None and sim >= WORKING_MEM_THRESHOLD:
                cluster_id_for_wm = cs.get("cluster_id")
            if sim >= SEMANTIC_CACHE_THRESHOLD:
                cache_hit = cs
                try:
                    increment_compacted_hit(conn, cs["id"])
                except Exception:
                    pass
                break  # one cache hit is enough
            elif ENRICHMENT_THRESHOLD <= sim < SEMANTIC_CACHE_THRESHOLD:
                enrichment.append(cs)
    except Exception as exc:
        _log_error(f"compacted search failed: {exc}")

    # --- Working memory (first message of session only) ---
    working_mem: dict | None = None
    if _is_first_message(session_id) and cluster_id_for_wm:
        try:
            working_mem = get_active_working_memory(conn, cluster_id_for_wm)
        except Exception as exc:
            _log_error(f"working memory lookup failed: {exc}")

    # --- Facts (identity surfaces here when relevant) ---
    facts: list[dict] = []
    try:
        tokens  = [t for t in prompt.split() if len(t) > 2]
        or_query = " OR ".join(tokens) if tokens else prompt
        facts   = search_facts(conn, or_query, limit=5)
    except Exception as exc:
        _log_error(f"facts search failed: {exc}")

    # --- Procedural memory (how-to prompts only) ---
    procedural: list[dict] = []
    if _is_how_to(prompt):
        try:
            procedural = search_procedural(conn, prompt, min_confidence=0.6, limit=3)
        except Exception as exc:
            _log_error(f"procedural search failed: {exc}")

    # Build the injection block.
    injection = _build_injection(cache_hit, working_mem, enrichment, facts, procedural)

    if not injection:
        conn.close()
        _allow()

    # Log the retrieval for the audit trail.
    try:
        est_tokens = len(injection) // 4
        log_retrieval(conn, "wake_up_injection", prompt, est_tokens)
        activity_log(
            "wake_up", "injected",
            session=session_id,
            has_cache_hit=bool(cache_hit),
            has_working_mem=bool(working_mem),
            enrichment_count=len(enrichment),
            facts_count=len(facts),
            procedural_count=len(procedural),
            est_tokens=est_tokens,
        )
    except Exception:
        pass

    conn.close()

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "userPromptSuffix": "\n\n" + injection,
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
