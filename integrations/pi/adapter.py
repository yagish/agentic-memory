"""JSON-over-stdin adapter used by the pi extension.

Commands:
  python3 integrations/pi/adapter.py recall
  python3 integrations/pi/adapter.py save
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.common import (
    DEFAULT_DB_PATH,
    open_memory_db_for_ingest,
    save_session_to_memory,
)
from memory.client import MemoryClient


DB_PATH = DEFAULT_DB_PATH
WAKE_UP_LOG_PATH = os.path.expanduser("~/.memory/wake_up.log")
SAVE_HOOK_LOG_PATH = os.path.expanduser("~/.memory/save_hook.log")
AGENT_NAME = "pi"


def _read_payload() -> dict:
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def _emit(data: dict) -> None:
    print(json.dumps(data))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _should_write_log_file() -> bool:
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    return True


def _append_log_line(path: str, level: str, msg: str) -> None:
    if not _should_write_log_file():
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        with open(path, "a") as handle:
            handle.write(f"{timestamp} {level.upper()} {msg}\n")
    except Exception:
        pass


def _save_log_info(msg: str) -> None:
    _append_log_line(SAVE_HOOK_LOG_PATH, "info", msg)


def _save_log_error(msg: str) -> None:
    _append_log_line(SAVE_HOOK_LOG_PATH, "error", msg)


def _wake_log_info(msg: str) -> None:
    _append_log_line(WAKE_UP_LOG_PATH, "info", msg)


def _wake_log_error(msg: str) -> None:
    _append_log_line(WAKE_UP_LOG_PATH, "error", msg)


def handle_save(payload: dict, *, db_path: str = DB_PATH) -> dict:
    turns = payload.get("turns") or []
    if not turns:
        _save_log_error(f"agent={AGENT_NAME} save rejected: turns must not be empty")
        return {"ok": False, "error": "turns must not be empty"}

    session_id = payload.get("session_id")
    if not session_id:
        _save_log_error(f"agent={AGENT_NAME} save rejected: session_id is required")
        return {"ok": False, "error": "session_id is required"}

    _save_log_info(f"agent={AGENT_NAME} save invoked, session_id={session_id}, turns={len(turns)}")
    conn = open_memory_db_for_ingest(db_path)
    try:
        outcome = save_session_to_memory(
            conn,
            session_id=session_id,
            agent=payload.get("agent", "pi"),
            turns=turns,
            started_at=payload.get("started_at") or _utc_now(),
            updated_at=payload.get("updated_at") or _utc_now(),
            metadata=payload.get("metadata"),
        )
        for warning in outcome.warnings:
            _save_log_error(f"agent={AGENT_NAME} {warning.stage} failed for session {session_id}: {warning.message}")
        _save_log_info(f"agent={AGENT_NAME} saved session {session_id} ({outcome.turn_count} turns) via pi adapter")
        return {
            "ok": True,
            "session_id": outcome.session_id,
            "turns_stored": outcome.turn_count,
            "warnings": [warning.__dict__ for warning in outcome.warnings],
        }
    finally:
        conn.close()


def handle_recall(payload: dict, *, db_path: str = DB_PATH) -> dict:
    del db_path

    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return {"action": "noop"}

    session_id = payload.get("session_id")
    include_working_memory = bool(payload.get("include_working_memory", False))
    _wake_log_info(
        f"agent={AGENT_NAME} recall invoked, prompt={prompt!r}, include_working_memory={include_working_memory}, session_id={session_id}"
    )

    try:
        response = MemoryClient(port=int(os.environ.get("MEMORY_INGEST_PORT", "7747"))).recall(
            prompt,
            include_working_memory=include_working_memory,
            session_id=session_id,
            agent="pi",
        )
        for warning in response.get("warnings", []):
            _wake_log_error(f"agent={AGENT_NAME} recall {warning.get('stage')} failed: {warning.get('message')}")
        _wake_log_info(
            f"agent={AGENT_NAME} recall response=" + json.dumps(
                {
                    "action": response.get("action"),
                    "facts_count": response.get("facts_count", 0),
                    "episodic_count": response.get("episodic_count", 0),
                    "procedural_count": response.get("procedural_count", 0),
                    "session_memory_count": response.get("session_memory_count", 0),
                    "working_memory_count": response.get("working_memory_count", 0),
                },
                ensure_ascii=False,
            )
        )
        return response
    except (ConnectionError, RuntimeError) as exc:
        _wake_log_error(f"agent={AGENT_NAME} recall server unavailable ({exc})")
        return {
            "action": "noop",
            "error": str(exc),
            "server_required": True,
        }


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if not argv or argv[0] not in {"save", "recall"}:
        print("usage: python3 integrations/pi/adapter.py [save|recall]", file=sys.stderr)
        return 2

    try:
        payload = _read_payload()
        if argv[0] == "save":
            _emit(handle_save(payload))
        else:
            _emit(handle_recall(payload))
        return 0
    except Exception as exc:
        if argv[0] == "save":
            _save_log_error(f"agent={AGENT_NAME} save adapter failed: {exc}")
        else:
            _wake_log_error(f"agent={AGENT_NAME} recall adapter failed: {exc}")
        _emit({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
