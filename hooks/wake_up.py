# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before every user prompt. Injects memory context from relevant layers,
# within a hard 500-token budget.
#
# Retrieval layers (priority order):
#   1. Cache hit      — compacted session ≥ 96% similar to this prompt
#   2. Working memory — rolling task context (first message of session only)
#   3. Enrichment     — compacted sessions 70–95% similar
#   4. Facts          — semantic search over facts table (identity surfaces here)
#   5. Procedural     — how-to patterns (only when prompt has how-to markers)
#
# Gates that skip ALL injection:
#   - DB not found
#   - Embed fails (returns error to avoid blocking the user)
#
# Short/trivial prompts (yes, ok, continue) are NOT gated — they simply find
# nothing in the DB and produce no injection, so zero extra LLM tokens are spent.
#
# Output protocol:
#   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "userPrompt": "..."}}
#   stderr + exit code 2  → short-circuit with saved response from memory
#   {}                    → do nothing

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


def _respond_with_prompt(updated_prompt: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "userPrompt": updated_prompt,
        }
    }))
    sys.exit(0)


def _respond_with_saved_response(answer: str) -> None:
    response = answer.strip()
    if response:
        # Wrap the message in the mandatory JSON error envelope
        error_payload = {
            "error": {
                "message": response
            }
        }
        # 3. Write the JSON payload directly to stderr
        sys.stderr.write(json.dumps(error_payload))
        sys.stderr.write("\n")
        
    # 4. Signal Claude Code to abort and display your message
    sys.exit(2)

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


def _get_saved_response(context) -> str:
    if not context.cache_hit:
        return ""
    return context.cache_hit.get("response", "").strip()


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
            facts_count=len(context.facts),
            procedural_count=len(context.procedural),
            est_tokens=est_tokens,
        )
    except Exception:
        pass


def _build_updated_prompt(prompt: str, injection: str) -> str:
    return f"{injection}\nUser: {prompt}"


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

        saved_response = _get_saved_response(context)
        if saved_response:
            similarity = context.cache_hit.get("similarity", 0) if context.cache_hit else 0
            _log_info(
                f"high-similarity saved response found ({similarity:.0%} match); "
                "returning cached answer and skipping LLM"
            )
            _log_retrieval_metrics(
                conn,
                tool_name="wake_up_cache_hit",
                action="cache_hit_short_circuit",
                request=request,
                context=context,
                payload_text=saved_response,
            )
            _respond_with_saved_response(saved_response)

        injection = _build_injection(context)
        if not injection:
            _log_info("semantic search complete, no injection produced")
            _allow()

        updated_prompt = _build_updated_prompt(request.prompt, injection)
        _log_info(f"semantic search complete, updated prompt={updated_prompt!r}")
        _log_retrieval_metrics(
            conn,
            tool_name="wake_up_injection",
            action="injected",
            request=request,
            context=context,
            payload_text=injection,
        )
        _respond_with_prompt(updated_prompt)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
