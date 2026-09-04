# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before every user prompt. Injects memory context from relevant layers,
# within a hard 500-token budget.
#
# Retrieval layers (priority order):
#   1. Cache hit      — compacted session ≥ 96% similar to this prompt
#   2. Working memory — rolling task context (first message of session only)
#   3. Enrichment     — compacted sessions 70–95% similar
#   4. Episodic       — semantically related session summaries
#   5. Facts          — semantic search over facts table (identity surfaces here)
#   6. Procedural     — how-to patterns (only when prompt has how-to markers)
#
# Gates that skip ALL injection:
#   - DB not found
#   - Embed fails (returns error to avoid blocking the user)
#
# Short/trivial prompts (yes, ok, continue) are NOT gated — they simply find
# nothing in the DB and produce no injection, so zero extra LLM tokens are spent.
#
# Output protocol:
#   {"decision": "block", "reason": "...", "suppressOriginalPrompt": true}
#       → deterministically answer direct fact questions through a documented
#          UserPromptSubmit block after rendering the stored facts with the local LLM
#   {}  → do nothing

from dataclasses import dataclass
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import open_db, log_retrieval
from memory.logger import activity_log
from memory.debug import enable_debug
from memory.fact_renderer import render_fact_answer
from memory.retrieval import (
    build_fact_query as _build_fact_query,
    build_wake_up_injection as _build_injection,
    retrieve_wake_up_context,
)

DB_PATH = os.path.expanduser("~/.memory/memory.db")
LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")


@dataclass(frozen=True)
class HookRequest:
    session_id: str
    prompt: str


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


def _respond_with_blocked_prompt(reason: str) -> None:
    """Return the documented UserPromptSubmit block response shape.

    Claude Code's current hooks contract does not allow UserPromptSubmit to
    replace the user prompt. The supported deterministic path is to block the
    prompt with a reason shown to the user.
    """
    print(json.dumps({
        "decision": "block",
        "reason": reason.strip(),
        "suppressOriginalPrompt": True,
    }))
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


def _sanitize_session_id(raw_session_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw_session_id)


def _strip_xml_tags(prompt: str) -> str:
    """Remove all XML-like tags and their contents (Claude Code context tags).
    
    Claude Code injects context tags like <ide_opened_file>...</ide_opened_file>
    into the prompt field. Strip all such tags generically to avoid contaminating
    semantic search embeddings with unrelated context.
    """
    # Match any <tag>...</tag> pattern and remove it, including contents and whitespace.
    cleaned = re.sub(r'<[a-z_]+[^>]*>.*?</[a-z_]+>\s*', '', prompt, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _parse_request() -> HookRequest:
    payload = json.load(sys.stdin)
    raw_sid = payload.get("session_id", "unknown")
    prompt = payload.get("prompt", "").strip()
    # Strip any XML tags injected by Claude Code before semantic processing.
    prompt = _strip_xml_tags(prompt)
    return HookRequest(
        session_id=_sanitize_session_id(raw_sid),
        prompt=prompt,
    )


def _open_connection():
    _log_info(f"opening memory db at {DB_PATH}")
    return open_db(DB_PATH)


def _retrieve_context(conn, request: HookRequest):
    # Inject working memory once per session; other memory layers run on every prompt.
    include_working_memory = _is_first_message(request.session_id)
    _log_info(
        f"running semantic search for prompt={request.prompt!r}, "
        f"include_working_memory={include_working_memory}"
    )
    return retrieve_wake_up_context(
        conn,
        request.prompt,
        include_working_memory=include_working_memory,
    )


def _log_context_warnings(context) -> None:
    for warning in context.warnings:
        _log_error(f"{warning.stage} failed: {warning.message}")


def _log_fact_lookup_details(request: HookRequest, context) -> None:
    """Log the fact-search inputs and outputs for live retrieval debugging."""
    query_terms = _build_fact_query(request.prompt)
    _log_info(f"fact lookup query_terms={query_terms!r}")

    results = []
    for fact in context.facts:
        results.append({
            "content": str(fact.get("content", "")).strip(),
            "similarity": fact.get("similarity"),
            "id": fact.get("id"),
        })
    _log_info(f"fact lookup results={json.dumps(results, ensure_ascii=False)}")



def _log_context_details(context) -> None:
    """Log the other retrieved memory layers for wake-up debugging."""
    if context.cache_hit:
        _log_info(
            "cache hit=" + json.dumps(
                {
                    "id": context.cache_hit.get("id"),
                    "similarity": context.cache_hit.get("similarity"),
                    "content": str(context.cache_hit.get("content", "")).strip(),
                },
                ensure_ascii=False,
            )
        )

    if context.working_mem:
        _log_info(
            "working memory=" + json.dumps(
                {
                    "id": context.working_mem.get("id"),
                    "similarity": context.working_mem.get("similarity"),
                    "summary": str(context.working_mem.get("summary", "")).strip(),
                },
                ensure_ascii=False,
            )
        )

    if context.enrichment:
        _log_info("enrichment=" + json.dumps(context.enrichment, ensure_ascii=False))

    if context.episodic:
        _log_info("episodic=" + json.dumps(context.episodic, ensure_ascii=False))

    if context.procedural:
        _log_info("procedural=" + json.dumps(context.procedural, ensure_ascii=False))



def _should_block_with_fact_answer(request: HookRequest, context) -> bool:
    """Only block when wake-up found facts and no richer context layers compete.

    Claude Code's UserPromptSubmit hook cannot inject context into the prompt, so
    direct blocking is reserved for fact-only recall. When other memory layers are
    relevant we log them and allow the prompt through unchanged.
    """
    return bool(context.facts) and not any([
        context.cache_hit,
        context.working_mem,
        context.enrichment,
        context.episodic,
        context.procedural,
    ])



def _log_retrieval_metrics(
    conn,
    *,
    tool_name: str,
    action: str,
    request: HookRequest,
    context,
    payload_text: str,
) -> None:
    try:
        est_tokens = len(payload_text) // 4
        log_retrieval(conn, tool_name, request.prompt, est_tokens)
        activity_log(
            "wake_up",
            action,
            session=request.session_id,
            has_cache_hit=bool(context.cache_hit),
            has_working_mem=bool(context.working_mem),
            enrichment_count=len(context.enrichment),
            episodic_count=len(context.episodic),
            facts_count=len(context.facts),
            procedural_count=len(context.procedural),
            est_tokens=est_tokens,
        )
    except Exception:
        pass


def main() -> None:
    enable_debug("wake_up")
    _log_info("=" * 60)
    _log_info("HOOK INVOKED BY CLAUDE CODE")

    try:
        request = _parse_request()
    except Exception as exc:
        _log_error(f"failed to parse stdin: {exc}")
        _allow()

    _log_info(f"hook invoked, prompt={request.prompt!r}")

    if not request.prompt:
        _log_info("empty prompt, skipping wake_up injection")
        _allow()

    if not os.path.exists(DB_PATH):
        _log_info(f"memory db not found at {DB_PATH}, skipping wake_up injection")
        _allow()

    try:
        conn = _open_connection()
    except Exception:
        _log_error(f"failed to open DB: {traceback.format_exc()}")
        _allow()

    try:
        try:
            context = _retrieve_context(conn, request)
        except Exception as exc:
            _log_error(f"embed failed: {exc}")
            _allow()

        _log_context_warnings(context)
        _log_fact_lookup_details(request, context)
        _log_context_details(context)

        if _should_block_with_fact_answer(request, context):
            fact_contents = [str(fact.get("content", "")) for fact in context.facts]
            answer = render_fact_answer(request.prompt, fact_contents)
            canonical_facts = [fact.strip() for fact in fact_contents if fact.strip()]
            _log_info(f"fact renderer input={json.dumps(canonical_facts, ensure_ascii=False)}")
            _log_info(f"fact renderer output={answer!r}")
            if answer.strip() in canonical_facts or answer.strip() == "\n".join(canonical_facts):
                _log_info("fact lookup hit; renderer fell back to canonical facts")
            else:
                _log_info("fact lookup hit; rendered answer from retrieved facts with local llm")
            _log_retrieval_metrics(
                conn,
                tool_name="wake_up_fact_hit",
                action="fact_hit_blocked",
                request=request,
                context=context,
                payload_text=answer,
            )
            _respond_with_blocked_prompt(answer)

        injection = _build_injection(context)
        if injection:
            _log_info(f"wake-up context found but prompt injection is unavailable for this hook: {injection}")
            _log_retrieval_metrics(
                conn,
                tool_name="wake_up_context_found",
                action="context_found_allow",
                request=request,
                context=context,
                payload_text=injection,
            )
        else:
            _log_info("no wake-up memory found; allowing prompt through")
            _log_retrieval_metrics(
                conn,
                tool_name="wake_up_noop",
                action="context_miss_allow",
                request=request,
                context=context,
                payload_text="",
            )
        _allow()
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
