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


def _read_payload() -> dict:
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def _emit(data: dict) -> None:
    print(json.dumps(data))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def handle_save(payload: dict, *, db_path: str = DB_PATH) -> dict:
    turns = payload.get("turns") or []
    if not turns:
        return {"ok": False, "error": "turns must not be empty"}

    session_id = payload.get("session_id")
    if not session_id:
        return {"ok": False, "error": "session_id is required"}

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

    try:
        return MemoryClient(port=int(os.environ.get("MEMORY_INGEST_PORT", "7747"))).recall(
            prompt,
            include_working_memory=bool(payload.get("include_working_memory", False)),
            session_id=payload.get("session_id"),
        )
    except (ConnectionError, RuntimeError) as exc:
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
        _emit({"ok": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
