# wake_up.py — Claude Code UserPromptSubmit hook.
#
# Runs before every user prompt. Searches memory and either:
# - deterministically answers from retrieved facts, or
# - enriches the prompt with additional context when related memory is found
# - otherwise allows the prompt through unchanged
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
from memory.client import MemoryClient
from memory.db import log_retrieval
from memory.logger import activity_log
from memory.debug import enable_debug
from memory.retrieval import build_wake_up_injection as _build_injection

DB_PATH = DEFAULT_DB_PATH
WAKE_UP_LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")
AGENT_NAME = "claude"


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


def _emit_hook_response(payload: dict) -> None:
    _log_info(f"agent={AGENT_NAME} returning hook response=" + json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload))
    sys.exit(0)


def _allow() -> None:
    _emit_hook_response({})


def _respond_with_blocked_prompt(reason: str) -> None:
    _emit_hook_response({
        "decision": "block",
        "reason": reason.strip(),
        "suppressOriginalPrompt": True,
    })


def _respond_with_additional_context(additional_context: str) -> None:
    _emit_hook_response({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": additional_context,
        }
    })


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
    _log_info(f"agent={AGENT_NAME} opening memory db at {DB_PATH}")
    return open_existing_memory_db(DB_PATH)


def _retrieve_context(conn, request: HookRequest, *, include_working_memory: bool):
    _log_info(
        f"agent={AGENT_NAME} running semantic search for prompt={request.prompt!r}, "
        f"include_working_memory={include_working_memory}"
    )
    return retrieve_prompt_memory(
        conn,
        request.prompt,
        include_working_memory=include_working_memory,
    )


def _recall_via_server(request: HookRequest, *, include_working_memory: bool) -> dict | None:
    try:
        _log_info(
            f"agent={AGENT_NAME} calling recall server for prompt={request.prompt!r}, "
            f"include_working_memory={include_working_memory}, session_id={request.session_id}"
        )
        return MemoryClient(port=int(os.environ.get("MEMORY_INGEST_PORT", "7747"))).recall(
            request.prompt,
            include_working_memory=include_working_memory,
            session_id=request.session_id,
            agent="claude",
        )
    except (ConnectionError, RuntimeError) as exc:
        _log_info(f"agent={AGENT_NAME} recall server unavailable ({exc})")
        return None


def _log_context_warnings(context) -> None:
    for warning in context.warnings:
        _log_error(f"agent={AGENT_NAME} {warning.stage} failed: {warning.message}")


def _log_response_warnings(response: dict) -> None:
    for warning in response.get("warnings", []):
        _log_error(f"agent={AGENT_NAME} {warning.get('stage')} failed: {warning.get('message')}")


def _log_fact_lookup_details(_request: HookRequest, context) -> None:
    results = []
    for fact in context.facts:
        results.append({
            "content": str(fact.get("content", "")).strip(),
            "similarity": fact.get("similarity"),
            "id": fact.get("id"),
        })
    _log_info(f"agent={AGENT_NAME} fact lookup results={json.dumps(results, ensure_ascii=False)}")


def _log_response_fact_lookup_details(response: dict) -> None:
    results = []
    for fact in response.get("context", {}).get("facts", []):
        results.append({
            "content": str(fact.get("content", "")).strip(),
            "similarity": fact.get("similarity"),
            "id": fact.get("id"),
        })
    _log_info(f"agent={AGENT_NAME} fact lookup results={json.dumps(results, ensure_ascii=False)}")


def _log_context_details(context) -> None:
    if getattr(context, "working_mem", None):
        _log_info(f"agent={AGENT_NAME} working_mem=" + json.dumps(context.working_mem, ensure_ascii=False))
    if context.episodic:
        _log_info(f"agent={AGENT_NAME} episodic=" + json.dumps(context.episodic, ensure_ascii=False))
    if context.procedural:
        _log_info(f"agent={AGENT_NAME} procedural=" + json.dumps(context.procedural, ensure_ascii=False))
    if getattr(context, "session_memory", None):
        _log_info(f"agent={AGENT_NAME} session_memory=" + json.dumps(context.session_memory, ensure_ascii=False))


def _log_response_context_details(response: dict) -> None:
    context = response.get("context", {})
    if context.get("working_mem"):
        _log_info(f"agent={AGENT_NAME} working_mem=" + json.dumps(context["working_mem"], ensure_ascii=False))
    if context.get("episodic"):
        _log_info(f"agent={AGENT_NAME} episodic=" + json.dumps(context["episodic"], ensure_ascii=False))
    if context.get("procedural"):
        _log_info(f"agent={AGENT_NAME} procedural=" + json.dumps(context["procedural"], ensure_ascii=False))
    if context.get("session_memory"):
        _log_info(f"agent={AGENT_NAME} session_memory=" + json.dumps(context["session_memory"], ensure_ascii=False))


def _compose_enriched_prompt(prompt: str, injection: str) -> str:
    prompt = prompt.strip()
    injection = injection.strip()
    if not injection:
        return prompt
    if not prompt:
        return injection
    return f"{injection}\n\n[User prompt]\n{prompt}"


def _log_prompt_enrichment(request: HookRequest, injection: str) -> None:
    if not injection:
        return
    _log_info(f"agent={AGENT_NAME} prompt enrichment context for session={request.session_id}: {injection}")
    enriched_prompt = _compose_enriched_prompt(request.prompt, injection)
    _log_info(f"agent={AGENT_NAME} effective prompt to Claude=" + repr(enriched_prompt))


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
            session_memory_count=len(getattr(context, "session_memory", []) or []),
            working_memory_count=1 if getattr(context, "working_mem", None) else 0,
            est_tokens=est_tokens,
        )
    except Exception:
        pass


def main() -> None:
    enable_debug("wake_up")
    _log_info("=" * 60)
    _log_info(f"HOOK INVOKED BY CLAUDE CODE agent={AGENT_NAME}")

    try:
        request = _parse_request()
    except Exception as exc:
        _log_error(f"failed to parse stdin: {exc}")
        _allow()

    _log_info(f"agent={AGENT_NAME} hook invoked, prompt={request.prompt!r}, session_id={request.session_id}")

    if not request.prompt:
        _log_info(f"agent={AGENT_NAME} empty prompt, skipping wake_up injection")
        _allow()

    include_working_memory = _is_first_message(request.session_id)

    server_response = _recall_via_server(request, include_working_memory=include_working_memory)
    if server_response is None:
        _log_info(f"agent={AGENT_NAME} recall server unavailable; allowing prompt through")
        _allow()

    conn = None
    try:
        conn = _open_connection()
    except Exception:
        conn = None

    try:
        _log_response_warnings(server_response)
        _log_response_fact_lookup_details(server_response)
        _log_response_context_details(server_response)

        if server_response.get("action") == "answer":
            answer = str(server_response.get("answer", ""))
            canonical_facts = [
                str(fact.get("content", "")).strip()
                for fact in server_response.get("context", {}).get("facts", [])
                if str(fact.get("content", "")).strip()
            ]
            _log_info(f"agent={AGENT_NAME} fact renderer input={json.dumps(canonical_facts, ensure_ascii=False)}")
            _log_info(f"agent={AGENT_NAME} fact renderer output={answer!r}")
            _log_info(f"agent={AGENT_NAME} fact lookup hit; rendered answer deterministically from stored facts")
            if conn is not None:
                _log_retrieval_metrics(
                    conn,
                    tool_name="wake_up_fact_hit",
                    action="fact_hit_blocked",
                    request=request,
                    context=type("Ctx", (), server_response.get("context", {}))(),
                    payload_text=answer,
                )
            _respond_with_blocked_prompt(answer)

        injection = str(server_response.get("injection", ""))
        if injection:
            _log_prompt_enrichment(request, injection)
            if conn is not None:
                _log_retrieval_metrics(
                    conn,
                    tool_name="wake_up_context_found",
                    action="context_found_inject",
                    request=request,
                    context=type("Ctx", (), server_response.get("context", {}))(),
                    payload_text=injection,
                )
            _respond_with_additional_context(injection)
        else:
            _log_info(f"agent={AGENT_NAME} no wake-up memory found; allowing prompt through")
            if conn is not None:
                _log_retrieval_metrics(
                    conn,
                    tool_name="wake_up_noop",
                    action="context_miss_allow",
                    request=request,
                    context=type("Ctx", (), server_response.get("context", {}))(),
                    payload_text="",
                )
        _allow()
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
