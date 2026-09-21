# save_hook.py — Claude Code Stop hook.
#
# Claude Code calls this script after every assistant response.
# It receives a JSON payload on stdin, reads the conversation transcript
# from disk, and saves it to the memory database.
#
# Usage (automatic via .claude/settings.json):
#   python3 /path/to/integrations/claude/save_hook.py
#
# Manual dry-run (prints what would be saved, writes nothing):
#   echo '{"session_id":"test","transcript_path":"/path/to/file.jsonl","stop_hook_active":false}' \
#     | python3 integrations/claude/save_hook.py --dry-run

import json
import logging
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integrations.common import DEFAULT_DB_PATH, normalize_project_context, open_memory_db_for_ingest, save_session_to_memory
from memory.utils.logger import activity_log, error_log
from memory.utils.debug import enable_debug


DB_PATH = DEFAULT_DB_PATH
SAVE_HOOK_LOG_PATH = os.path.expanduser("~/.memory/save_hook.log")
AGENT_NAME = "claude"


def _setup_logging() -> None:
    """Configure save-hook logging.

    Keep stderr output for hook debugging and also write a retained file so the
    dashboard can surface save-hook activity again.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if (
        os.environ.get("MEMORY_DISABLE_FILE_LOGS") != "1"
        and not os.environ.get("PYTEST_CURRENT_TEST")
        and "pytest" not in sys.modules
    ):
        os.makedirs(os.path.dirname(SAVE_HOOK_LOG_PATH), exist_ok=True)
        handlers.append(logging.FileHandler(SAVE_HOOK_LOG_PATH))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _extract_text_content(content) -> str:
    """Return visible text from either a string or Claude block-list content."""
    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    text_parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in text_parts if part).strip()


def parse_transcript(jsonl_path: str) -> tuple[list[dict], str, str]:
    """
    Read the JSONL transcript file and extract conversation turns.

    Claude Code transcripts mix human prompts, assistant progress/tool-use
    events, tool results, and hook attachments. We retain only visible user
    text plus the assistant's final visible reply for each turn.

    Returns:
        turns       — list of {"role": "user"/"assistant", "content": "..."}
        started_at  — ISO timestamp of the first retained turn
        session_id  — the session ID (taken from the first matching line)
    """
    turns = []
    started_at = None
    session_id = None

    with open(jsonl_path, "r") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            event_type = obj.get("type")

            if session_id is None and obj.get("sessionId"):
                session_id = obj["sessionId"]

            if event_type == "user":
                if obj.get("isMeta"):
                    continue

                content = _extract_text_content(obj.get("message", {}).get("content", ""))
                if not content:
                    continue

                timestamp = obj.get("timestamp")
                if started_at is None and timestamp:
                    started_at = timestamp

                turns.append({"role": "user", "content": content})

            elif event_type == "assistant":
                message = obj.get("message", {})
                stop_reason = message.get("stop_reason")

                if stop_reason == "tool_use":
                    continue

                combined = _extract_text_content(message.get("content", []))
                if not combined:
                    continue

                timestamp = obj.get("timestamp")
                if started_at is None and timestamp:
                    started_at = timestamp

                turns.append({"role": "assistant", "content": combined})

    return turns, started_at, session_id


def _git_output(args: list[str], *, cwd: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    output = completed.stdout.strip()
    return output or None


def _build_project_context(payload: dict, transcript_path: str) -> dict[str, str]:
    workspace = payload.get("workspace") if isinstance(payload.get("workspace"), dict) else {}
    candidates = [
        payload.get("repo_root"),
        payload.get("cwd"),
        workspace.get("repo_root"),
        workspace.get("cwd"),
        os.path.dirname(os.path.abspath(transcript_path)),
        os.getcwd(),
    ]

    repo_root = None
    cwd = None
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized_candidate = os.path.abspath(os.path.expanduser(candidate.strip()))
        cwd = cwd or normalized_candidate
        repo_root = _git_output(["rev-parse", "--show-toplevel"], cwd=normalized_candidate)
        if repo_root:
            break

    git_base = repo_root or cwd
    git_remote = _git_output(["config", "--get", "remote.origin.url"], cwd=git_base) if git_base else None
    git_branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=git_base) if git_base else None

    return normalize_project_context(
        {
            "project_id": payload.get("project_id"),
            "repo_root": repo_root or payload.get("repo_root") or workspace.get("repo_root"),
            "cwd": payload.get("cwd") or workspace.get("cwd") or cwd,
            "git_remote": payload.get("git_remote") or workspace.get("git_remote") or git_remote,
            "git_branch": payload.get("git_branch") or workspace.get("git_branch") or payload.get("branch") or git_branch,
        }
    )


def save_session(payload: dict, dry_run: bool = False) -> None:
    """
    Core logic: given the Stop hook payload, parse the transcript and save it.

    Extracted as a separate function so tests can call it directly without
    needing to mock stdin or subprocesses.
    """
    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        raise ValueError("payload missing 'transcript_path'")

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"transcript not found: {transcript_path}")

    turns, started_at, session_id = parse_transcript(transcript_path)

    if not session_id:
        session_id = payload.get("session_id")

    if not session_id:
        raise ValueError("could not determine session_id")

    if not turns:
        return

    updated_at = datetime.now(timezone.utc).isoformat()
    project_context = _build_project_context(payload, transcript_path)

    if dry_run:
        print(f"[dry-run] session_id={session_id}")
        print(f"[dry-run] started_at={started_at}, updated_at={updated_at}")
        print(f"[dry-run] turns={len(turns)}")
        for i, turn in enumerate(turns[:4]):
            preview = turn["content"][:80].replace("\n", " ")
            print(f"[dry-run]   turn {i}: {turn['role']}: {preview}")
        if len(turns) > 4:
            print(f"[dry-run]   ... {len(turns) - 4} more turns")
        return

    conn = open_memory_db_for_ingest(DB_PATH)
    agent_name = os.environ.get("MEMORY_AGENT_NAME", AGENT_NAME)
    outcome = save_session_to_memory(
        conn,
        session_id=session_id,
        agent=agent_name,
        turns=turns,
        started_at=started_at or updated_at,
        updated_at=updated_at,
        metadata={
            "integration": "claude",
            "transcript_path": transcript_path,
            "project_context": project_context,
        },
        project_context=project_context,
    )
    activity_log("save_hook", "upsert_session", session=session_id, agent=agent_name, turns=outcome.turn_count)

    warning_by_stage = {warning.stage: warning for warning in outcome.warnings}
    for warning in warning_by_stage.values():
        logging.warning("%s skipped: %s", warning.stage, warning.message)
        error_log("save_hook", f"agent={agent_name} {warning.stage} failed for session {session_id}: {warning.message}")

    conn.close()
    logging.info("agent=%s saved session %s (%d turns)", agent_name, session_id, outcome.turn_count)


def main() -> None:
    """Entry point when the script is run by Claude Code's Stop hook."""
    dry_run = "--dry-run" in sys.argv

    _setup_logging()
    enable_debug("save_hook")

    try:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}

        if payload.get("stop_hook_active"):
            print("{}")
            sys.exit(0)

        save_session(payload, dry_run=dry_run)

    except Exception:
        logging.error(traceback.format_exc())

    if not dry_run:
        print("{}")


if __name__ == "__main__":
    main()
