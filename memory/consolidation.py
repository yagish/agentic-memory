# consolidation.py — summarise old sessions and prune raw transcripts.
#
# This module runs periodically (via CLI commands) to keep the memory database
# manageable. Old sessions are first summarised with a local ollama LLM, then
# their raw transcripts can be pruned to free up space.
#
# IMPORTANT: no Claude API or anthropic package is used here.
# All LLM calls go to the local ollama server via plain HTTP (urllib.request).

import json              # for encoding request body and decoding ollama response
import os               # for reading MEMORY_OLLAMA_MODEL environment variable
import urllib.error      # for catching connection errors when ollama is unavailable
import urllib.request   # Python's built-in HTTP client — no external libs needed

from memory.db import (
    insert_summary,              # write a generated summary to the summaries table
    prune_transcript,            # set a session's transcript to '[]'
    sessions_needing_prune,      # find sessions with summaries ready for pruning
    sessions_needing_summary,    # find old sessions that have no summary yet
)
from memory.logger import activity_log, error_log


# URL of the local ollama HTTP API endpoint for text generation.
_OLLAMA_URL = "http://localhost:11434/api/generate"

# Default model name — can be overridden via the MEMORY_OLLAMA_MODEL env var.
_DEFAULT_MODEL = "qwen2.5:3b"


def _call_ollama(prompt_text: str) -> str:
    """
    Send a prompt to the local ollama API and return the generated text.

    Uses Python's built-in urllib.request so there are no external HTTP
    dependencies. The request times out after 30 seconds.

    No Claude API or anthropic package is used — only the local ollama server.

    Args:
        prompt_text — the full prompt string to send to the model

    Returns:
        The generated text string from ollama.

    Raises:
        RuntimeError if ollama is unreachable (not running) or returns an
        unexpected response (missing "response" key).
    """
    # Read the model name from the environment, falling back to the default.
    # This lets users switch models without changing code.
    model = os.environ.get("MEMORY_OLLAMA_MODEL", _DEFAULT_MODEL)

    # Build the JSON request body as a Python dict, then encode it to bytes.
    # stream=False tells ollama to return the full response in one HTTP reply.
    body = json.dumps({
        "model":  model,
        "prompt": prompt_text,
        "stream": False,
    }).encode("utf-8")

    # urllib.request.Request lets us set the HTTP method and headers explicitly.
    req = urllib.request.Request(
        _OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # urlopen() sends the HTTP POST and waits up to 30 seconds for a response.
        # If ollama is not running, this raises urllib.error.URLError immediately.
        with urllib.request.urlopen(req, timeout=30) as resp:
            # Read the response body (bytes) and decode to a UTF-8 string.
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        # URLError covers both "connection refused" (ollama not running) and timeout.
        # We re-raise as RuntimeError so callers have a single exception type to catch.
        raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    # Parse the JSON response. Ollama returns something like:
    # {"model": "...", "response": "the generated text", "done": true, ...}
    data = json.loads(raw)

    # Check that the expected "response" key is present — if not, something went wrong.
    if "response" not in data:
        raise RuntimeError(f"Unexpected ollama response (missing 'response' key): {raw[:200]}")

    return data["response"]


def _build_prompt(transcript_json: str) -> str:
    """
    Build a summarisation prompt from a stored transcript JSON string.

    The transcript is a JSON array of {"role": ..., "content": ...} dicts.
    We join them into a readable "role: content" format and truncate to 4000
    characters so the prompt fits comfortably within the model's context window.

    Args:
        transcript_json — the raw JSON string from the sessions.transcript column

    Returns:
        A formatted prompt string ready to send to ollama.
    """
    # Parse the JSON string back into a Python list of turn dicts.
    # If the JSON is malformed, we fall back to an empty list.
    try:
        turns = json.loads(transcript_json or "[]")
    except json.JSONDecodeError:
        turns = []

    # Build a flat text representation of the conversation.
    lines = []
    for turn in turns:
        role    = turn.get("role", "unknown")
        content = turn.get("content", "")
        # Skip turns whose content is not a plain string (e.g. tool-call dicts).
        # Tool-call turns contain nested lists/dicts that don't make sense as text.
        if isinstance(content, str):
            lines.append(f"{role}: {content}")

    # Join all lines, then take at most 4000 characters.
    # This keeps the prompt short enough for small local models like llama3.2:3b.
    transcript_text = "\n".join(lines)[:4000]

    # The prompt template instructs the model to be concise and focus on what matters.
    return (
        "Summarise this conversation in 2-3 sentences. "
        "Focus on the main topics discussed, key decisions made, "
        "and anything worth remembering. Be concise.\n\n"
        f"Conversation:\n{transcript_text}\n\nSummary:"
    )


def consolidate_old_sessions(
    conn,
    days_threshold: int = 30,
    dry_run: bool = False,
) -> dict:
    """
    Summarise sessions older than days_threshold days that have no summary yet.

    For each qualifying session:
      1. Build a summarisation prompt from the stored transcript.
      2. Call the local ollama API to generate a 2-3 sentence summary.
      3. Store the summary in the summaries table.
      4. Log the outcome via activity_log.

    If ollama is unavailable for a session, that session is skipped and an error
    is logged — the whole run does not abort. This means some sessions may need
    to be retried on the next consolidation run.

    Args:
        conn           — open SQLite connection from init_db()
        days_threshold — sessions updated more than this many days ago are candidates
        dry_run        — if True, print what would be done without writing or calling ollama

    Returns:
        A dict with keys "summarised" (count written) and "skipped" (count skipped).
    """
    # Fetch all sessions that need a summary according to the database.
    sessions = sessions_needing_summary(conn, days_threshold)

    # Read the model name once so it's consistent across all inserts in this run.
    model = os.environ.get("MEMORY_OLLAMA_MODEL", _DEFAULT_MODEL)

    # Counters reported back to the caller (and printed by the CLI command).
    summarised = 0
    skipped    = 0

    for session in sessions:
        session_id = session["session_id"]
        transcript = session.get("transcript") or "[]"

        # Extra defensive check: skip sessions with no meaningful transcript.
        # sessions_needing_summary() already filters these via SQL, but being
        # explicit here makes the logic easier to follow and test.
        if not transcript or transcript == "[]":
            activity_log(
                "consolidation", "summarise",
                session=session_id, status="skipped_empty",
            )
            skipped += 1
            continue

        # In dry-run mode, just describe what would happen and move on.
        if dry_run:
            print(
                f"[dry-run] Would summarise session {session_id} "
                f"(updated {session['updated_at']})"
            )
            skipped += 1  # count as skipped since nothing is written
            continue

        # Build the prompt from the transcript text.
        prompt = _build_prompt(transcript)

        # Call ollama. If it fails, log the error and skip this session.
        try:
            summary_text = _call_ollama(prompt)
        except RuntimeError as exc:
            # Log the error to error.log — never let it crash the whole run.
            error_log(
                "consolidation",
                f"ollama call failed for session {session_id}: {exc}",
                exc=exc,
            )
            activity_log(
                "consolidation", "summarise",
                session=session_id, status="error",
            )
            skipped += 1
            continue

        # Store the summary. strip() removes any leading/trailing whitespace
        # that the model might prepend (e.g. a blank line before the summary text).
        insert_summary(conn, session_id, summary_text.strip(), model)

        # Log success so the activity log shows what was processed.
        activity_log(
            "consolidation", "summarise",
            session=session_id, status="ok",
        )
        summarised += 1

    return {"summarised": summarised, "skipped": skipped}


def prune_old_transcripts(
    conn,
    days_threshold: int = 90,
    dry_run: bool = False,
) -> dict:
    """
    Null out transcripts for sessions older than days_threshold that have summaries.

    A session's raw transcript is only pruned if a summary already exists in the
    summaries table. This guarantees no conversation is silently lost — the summary
    must be created first (via consolidate_old_sessions) before pruning is allowed.

    After pruning, the session row still exists (metadata like updated_at and
    turn_count are preserved), but the transcript column is set to '[]'.

    Args:
        conn           — open SQLite connection from init_db()
        days_threshold — sessions older than this many days are candidates
        dry_run        — if True, print what would be pruned without writing

    Returns:
        A dict with key "pruned" (count of transcripts nulled out).
    """
    # Fetch sessions that are old enough and have a summary already.
    sessions = sessions_needing_prune(conn, days_threshold)

    # Counter reported back to the caller.
    pruned = 0

    for session in sessions:
        session_id = session["session_id"]

        if dry_run:
            # Print a description of what would happen without touching the database.
            print(
                f"[dry-run] Would prune transcript for session {session_id} "
                f"(updated {session['updated_at']}, {session['turn_count']} turns)"
            )
            continue

        # Null out the transcript and get the old turn count for the log entry.
        turns_pruned = prune_transcript(conn, session_id)

        # Log the pruning event so the activity log shows what was removed.
        activity_log(
            "consolidation", "prune",
            session=session_id, turns_pruned=turns_pruned,
        )
        pruned += 1

    return {"pruned": pruned}
