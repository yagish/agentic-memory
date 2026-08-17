# save_hook.py — Claude Code Stop hook.
#
# Claude Code calls this script after every assistant response.
# It receives a JSON payload on stdin, reads the conversation transcript
# from disk, and saves it to the memory database.
#
# Usage (automatic via .claude/settings.json):
#   python3 /path/to/hooks/save_hook.py
#
# Manual dry-run (prints what would be saved, writes nothing):
#   echo '{"session_id":"test","transcript_path":"/path/to/file.jsonl","stop_hook_active":false}' \
#     | python3 hooks/save_hook.py --dry-run

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone

# Add the project root to Python's module search path so we can import memory.db.
# __file__ is this script's path; we go up one level to reach the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    init_db,
    upsert_session,
    embed,
    store_embedding,
    chunk_transcript,
    store_chunk,
    delete_chunks_for_session,
)
from memory.logger import activity_log, error_log


# Where the memory database lives on disk.
DB_PATH = os.path.expanduser("~/.memory/memory.db")

# Log file — INFO on success, ERROR on failure.
LOG_PATH = os.path.expanduser("~/.memory/save_hook.log")


def _setup_logging() -> None:
    """
    Configure the logging module to write to LOG_PATH.

    Creates the ~/.memory directory if it doesn't exist yet.
    Each log line looks like: 2026-08-16T10:00:00 INFO saved session abc (12 turns)
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        # %(asctime)s = timestamp, %(levelname)s = INFO/ERROR, %(message)s = our text
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def parse_transcript(jsonl_path: str) -> tuple[list[dict], str, str]:
    """
    Read the JSONL transcript file and extract conversation turns.

    Each line in the file is a JSON object describing one event in the session.
    We care only about 'user' and 'assistant' type lines that are real messages
    (not meta/system events).

    Returns:
        turns       — list of {"role": "user"/"assistant", "content": "..."}
        started_at  — ISO timestamp of the first turn
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
                continue  # skip malformed lines

            event_type = obj.get("type")

            # Grab the session_id from the first line that has it.
            if session_id is None and obj.get("sessionId"):
                session_id = obj["sessionId"]

            # --- User turns ---
            if event_type == "user":
                # isMeta=True lines are internal system messages — skip them.
                if obj.get("isMeta"):
                    continue

                content = obj.get("message", {}).get("content", "")

                # content is a plain string for user messages.
                if not isinstance(content, str) or not content.strip():
                    continue

                timestamp = obj.get("timestamp")
                if started_at is None and timestamp:
                    started_at = timestamp

                turns.append({"role": "user", "content": content})

            # --- Assistant turns ---
            elif event_type == "assistant":
                content_blocks = obj.get("message", {}).get("content", [])

                # content is a list of typed blocks: text, thinking, tool_use, etc.
                # We only want the text blocks — that's the visible response.
                text_parts = [
                    block.get("text", "")
                    for block in content_blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ]

                combined = "\n".join(text_parts).strip()
                if not combined:
                    continue

                turns.append({"role": "assistant", "content": combined})

    return turns, started_at, session_id


def save_session(payload: dict, dry_run: bool = False) -> None:
    """
    Core logic: given the Stop hook payload, parse the transcript and save it.

    Extracted as a separate function so tests can call it directly without
    needing to mock stdin or subprocesses.

    Args:
        payload  — the parsed JSON from stdin (session_id, transcript_path, etc.)
        dry_run  — if True, print what would be saved but don't write to DB
    """
    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        raise ValueError("payload missing 'transcript_path'")

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"transcript not found: {transcript_path}")

    turns, started_at, session_id = parse_transcript(transcript_path)

    # Fall back to the session_id from the payload if the file didn't have one.
    if not session_id:
        session_id = payload.get("session_id")

    if not session_id:
        raise ValueError("could not determine session_id")

    if not turns:
        # Nothing to save — this can happen for very short or meta-only sessions.
        return

    updated_at = datetime.now(timezone.utc).isoformat()

    if dry_run:
        print(f"[dry-run] session_id={session_id}")
        print(f"[dry-run] started_at={started_at}, updated_at={updated_at}")
        print(f"[dry-run] turns={len(turns)}")
        for i, t in enumerate(turns[:4]):  # show first 4 turns
            preview = t["content"][:80].replace("\n", " ")
            print(f"[dry-run]   turn {i}: {t['role']}: {preview}")
        if len(turns) > 4:
            print(f"[dry-run]   ... {len(turns) - 4} more turns")
        return

    # Ensure the ~/.memory directory exists before opening the DB.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = init_db(DB_PATH)
    # Agent name is read from MEMORY_AGENT_NAME env var so any tool using this
    # hook can identify itself without code changes (defaults to "assistant").
    agent_name = os.environ.get("MEMORY_AGENT_NAME", "assistant")
    upsert_session(
        conn,
        session_id=session_id,
        agent=agent_name,
        transcript=turns,
        started_at=started_at or updated_at,
        updated_at=updated_at,
    )
    # Log the session save so the activity log shows when each session was stored.
    activity_log("save_hook", "upsert_session", session=session_id, turns=len(turns))

    # --- Phase 5: store a semantic embedding alongside the transcript ---
    # We catch all errors here so that a missing model or import never
    # prevents the hook from exiting 0 and unblocking Claude.
    try:
        # Concatenate all turn text into one string for embedding.
        # Joining everything gives the model the full conversation context.
        full_text = " ".join(
            turn["content"]
            for turn in turns
            if isinstance(turn.get("content"), str)
        )
        if full_text.strip():
            vector = embed(full_text)
            store_embedding(conn, session_id, vector)
            logging.info("stored embedding for session %s", session_id)
            activity_log("save_hook", "embedding", session=session_id, status="ok")
        else:
            activity_log("save_hook", "embedding", session=session_id, status="skipped_empty")
    except Exception as e:
        # Log the problem but never let it block the hook.
        logging.warning("embedding skipped: %s", traceback.format_exc())
        activity_log("save_hook", "embedding", session=session_id, status="skipped_error")
        error_log("save_hook", f"embedding failed for session {session_id}", exc=e)

    # --- Phase 7: store sub-session chunks for finer-grained semantic search ---
    # Chunking splits the transcript into overlapping windows and embeds each
    # window separately. This lets semantic search find the right session even
    # when the relevant content is a small part of a long conversation.
    # We wrap the whole block in try/except — chunking failure must never
    # prevent a successful session save.
    try:
        # Split the transcript into overlapping text windows.
        chunk_texts = chunk_transcript(turns)

        # Remove any old chunks for this session so we don't accumulate stale
        # rows if the transcript grew since the last save.
        delete_chunks_for_session(conn, session_id)

        # Embed and store each chunk. We index from 0 so chunk_index is stable
        # even if the transcript grows — the first window is always chunk 0.
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_vector = embed(chunk_text) if chunk_text.strip() else None
            store_chunk(conn, session_id, chunk_index, chunk_text, chunk_vector)

        logging.info("saved %d chunks for session %s", len(chunk_texts), session_id)
        activity_log("save_hook", "chunking", session=session_id, chunks=len(chunk_texts), status="ok")
    except Exception as e:
        # Log the failure but allow the hook to complete normally.
        logging.warning("chunking skipped: %s", traceback.format_exc())
        activity_log("save_hook", "chunking", session=session_id, status="skipped_error")
        error_log("save_hook", f"chunking failed for session {session_id}", exc=e)

    conn.close()

    logging.info("saved session %s (%d turns)", session_id, len(turns))


def main() -> None:
    """
    Entry point when the script is run by Claude Code's Stop hook.

    Claude Code passes a JSON object on stdin. We must:
    - Always exit 0 (non-zero would block the assistant from responding).
    - Always print at least '{}' on stdout (Claude Code reads this).
    - Never raise unhandled exceptions (swallow everything to the log).
    """
    dry_run = "--dry-run" in sys.argv

    _setup_logging()

    try:
        # Read the hook payload from stdin.
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}

        # stop_hook_active=True means *we* triggered this stop (e.g. from a previous
        # hook run). Exit immediately to prevent an infinite save loop.
        if payload.get("stop_hook_active"):
            print("{}")
            sys.exit(0)

        save_session(payload, dry_run=dry_run)

    except Exception:
        # Log the full traceback at ERROR level but never let it bubble up and block Claude.
        logging.error(traceback.format_exc())

    # Claude Code expects at least '{}' on stdout to know the hook finished.
    if not dry_run:
        print("{}")


if __name__ == "__main__":
    main()
