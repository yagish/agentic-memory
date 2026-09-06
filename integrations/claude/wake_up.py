# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before every user prompt. Searches memory and either:
# - deterministically answers from retrieved facts, or
# - allows the prompt through (Claude's hook contract cannot inject text here)
#
# Usage:
#   python3 /path/to/integrations/claude/wake_up.py

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.common import (
    DEFAULT_DB_PATH,
    decide_prompt_memory_action,
    open_existing_memory_db,
    retrieve_prompt_memory,
)
from memory.db import log_retrieval
from memory.logger import activity_log
from memory.debug import enable_debug
from memory.retrieval import build_wake_up_injection as _build_injection

DB_PATH = DEFAULT_DB_PATH
WAKE_UP_LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")


@dataclass(frozen=True)
class HookRequest:
    session_id: str
    prompt: str


def _append_log_line(level: str, msg: str) -> None:
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if "pytest" in sys.modules:
        return
    try:
        os.makedirs(os.path.dirname(WAKE_UP_LOG_PATH), exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open(WAKE_UP_LOG_PATH, "a") as handle:
            handle.write(f"{timestamp} {level.upper()} {msg}\n")
    except Exception:
        pass



def _log_error(msg: str) -> None:
    _append_log_line("error", msg)



def _log_info(msg: str) -> None:
    _append_log_line("info", msg)


def _allow() -> None:
    print(json.dumps({}))
    sys.exit(0)


def _respond_with_blocked_prompt(reason: str) -> None:
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
    cleaned = re.sub(r'<[a-z_]+[^>]*>.*?</[a-z_]+>\s*', '', prompt, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _parse_request() -> HookRequest:
    payload = json.load(sys.stdin)
    raw_sid = payload.get("session_id", "unknown")
    prompt = _strip_xml_tags(payload.get("prompt", "").strip())
    return HookRequest(
        session_id=_sanitize_session_id(raw_sid),
        prompt=prompt,
    )


def _open_connection():
    _log_info(f"opening memory db at {DB_PATH}")
    return open_existing_memory_db(DB_PATH)


def _retrieve_context(conn, request: HookRequest):
    include_working_memory = _is_first_message(request.session_id)
    _log_info(
        f"running semantic search for prompt={request.prompt!r}, "
        f"include_working_memory={include_working_memory}"
    )
    return retrieve_prompt_memory(
        conn,
        request.prompt,
        include_working_memory=include_working_memory,
    )


def _log_context_warnings(context) -> None:
    for warning in context.warnings:
        _log_error(f"{warning.stage} failed: {warning.message}")


def _log_fact_lookup_details(_request: HookRequest, context) -> None:
    results = []
    for fact in context.facts:
        results.append({
            "content": str(fact.get("content", "")).strip(),
            "similarity": fact.get("similarity"),
            "id": fact.get("id"),
        })
    _log_info(f"fact lookup results={json.dumps(results, ensure_ascii=False)}")


def _log_context_details(context) -> None:
    if context.episodic:
        _log_info("episodic=" + json.dumps(context.episodic, ensure_ascii=False))
    if context.procedural:
        _log_info("procedural=" + json.dumps(context.procedural, ensure_ascii=False))


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

    conn = None
    try:
        conn = _open_connection()
    except Exception:
        _log_error(f"failed to open DB: {traceback.format_exc()}")
        _allow()

    if conn is None:
        _log_info(f"memory db not found at {DB_PATH}, skipping wake_up injection")
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

        outcome = decide_prompt_memory_action(request.prompt, context)
        if outcome.action == "answer":
            fact_contents = [str(fact.get("content", "")) for fact in context.facts]
            canonical_facts = [fact.strip() for fact in fact_contents if fact.strip()]
            _log_info(f"fact renderer input={json.dumps(canonical_facts, ensure_ascii=False)}")
            _log_info(f"fact renderer output={outcome.answer!r}")
            if outcome.answer.strip() in canonical_facts or outcome.answer.strip() == "\n".join(canonical_facts):
                _log_info("fact lookup hit; renderer fell back to canonical facts")
            else:
                _log_info("fact lookup hit; rendered answer from retrieved facts with local llm")
            _log_retrieval_metrics(
                conn,
                tool_name="wake_up_fact_hit",
                action="fact_hit_blocked",
                request=request,
                context=context,
                payload_text=outcome.answer,
            )
            _respond_with_blocked_prompt(outcome.answer)

        if outcome.injection:
            _log_info(f"wake-up context found but prompt injection is unavailable for this hook: {outcome.injection}")
            _log_retrieval_metrics(
                conn,
                tool_name="wake_up_context_found",
                action="context_found_allow",
                request=request,
                context=context,
                payload_text=outcome.injection,
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
