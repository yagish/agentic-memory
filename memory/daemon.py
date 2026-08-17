# daemon.py — background relearning daemon.
#
# Runs as a long-lived process. Every POLL_INTERVAL seconds:
#   1. Picks up sessions not yet processed (daemon_processed_at IS NULL)
#   2. For each: extracts facts via ollama, assigns to topic cluster
#   3. Every N=10 new sessions: generates cross-session insights
#   4. Backs off to LONG_POLL_INTERVAL when nothing new is found
#   5. Skips inference cycles when CPU load > 70%
#
# Run:
#   python3 memory/daemon.py            # runs forever
#   python3 memory/daemon.py --once     # one pass then exit (for tests)
#
# Stop: SIGTERM — the daemon exits cleanly after the current session.
#
# IMPORTANT: no Claude API or anthropic package is used here.
# All LLM calls go to the local ollama server via plain HTTP (urllib.request).

import json              # for encoding request body and decoding JSON responses
import os               # for environment variable access and path expansion
import signal           # for handling SIGTERM / SIGINT gracefully
import struct           # for packing float arrays into blobs (used indirectly via db)
import sys              # for sys.path manipulation at the bottom
import time             # for sleep between polling cycles
import urllib.error      # for catching connection errors to ollama
import urllib.request   # Python's built-in HTTP client — no external libs needed
from datetime import datetime, timezone   # for timestamps in daemon log messages

# Add the project root to the module search path so we can import memory.*
# when this file is run as a script (python3 memory/daemon.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    init_db,                  # open (or create) the SQLite database
    get_unprocessed_sessions, # find sessions the daemon hasn't processed yet
    mark_session_processed,   # stamp daemon_processed_at on a finished session
    insert_fact,              # write a discovered fact to the facts table
    list_facts,               # retrieve stored facts for insight generation
    upsert_insight,           # write a cross-session insight to the insights table
    list_insights,            # retrieve stored insights (not directly used here but exported)
    assign_to_cluster,        # assign a session embedding to a topic cluster
)
from memory.logger import activity_log, error_log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# URL of the local ollama HTTP API endpoint for text generation.
# Must match the same constant in consolidation.py.
_OLLAMA_URL = "http://localhost:11434/api/generate"

# Default model name — overridden by the MEMORY_OLLAMA_MODEL env var.
_DEFAULT_MODEL = "llama3.2:3b"

# Where the main memory database lives.
DB_PATH = os.path.expanduser("~/.memory/memory.db")

# Where the daemon writes its own simple log file (in addition to activity_log).
_DAEMON_LOG_PATH = os.path.expanduser("~/.memory/daemon.log")

# How long to wait between processing cycles when sessions are available.
POLL_INTERVAL = 5 * 60       # 5 minutes

# How long to wait when there are no new sessions to process.
LONG_POLL_INTERVAL = 30 * 60  # 30 minutes

# Run cross-session insight generation every N newly-processed sessions.
INSIGHT_EVERY_N = 10

# Skip an inference cycle if the CPU % is above this threshold.
CPU_THRESHOLD = 70

# Global flag set by signal handlers so the main loop can exit cleanly.
_shutdown = False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_sigterm(signum, frame):
    """
    Handle SIGTERM and SIGINT by setting the global shutdown flag.

    The main loop checks this flag after each session so it can exit cleanly
    without interrupting a processing step mid-way.

    Args:
        signum — the signal number (SIGTERM=15, SIGINT=2)
        frame  — the current stack frame (not used here)
    """
    global _shutdown
    _shutdown = True


# Register the handler for both SIGTERM (sent by kill) and SIGINT (Ctrl-C).
signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _daemon_log(message: str) -> None:
    """
    Append a timestamped line to the daemon log file at ~/.memory/daemon.log.

    This is a simple flat-file log separate from the structured activity_log.
    It gives operators a quick way to tail -f the daemon's progress.

    Args:
        message — the message text to append
    """
    # Build a UTC timestamp prefix, e.g. "2026-08-17T10:00:00+00:00".
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        # Ensure the ~/.memory directory exists before writing.
        os.makedirs(os.path.dirname(_DAEMON_LOG_PATH), exist_ok=True)
        with open(_DAEMON_LOG_PATH, "a") as f:
            f.write(f"{timestamp} [daemon] {message}\n")
    except Exception:
        # Logging must never crash the daemon — silently ignore errors here.
        pass


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

def _call_ollama(prompt_text: str) -> str:
    """
    Send a prompt to the local ollama API and return the generated text.

    This is a copy of the same function in consolidation.py — both modules
    call ollama the same way so there is no cross-dependency.

    Uses Python's built-in urllib.request so there are no external HTTP
    dependencies. The request times out after 30 seconds.

    No Claude API or anthropic package is used — only the local ollama server.

    Args:
        prompt_text — the full prompt string to send to the model

    Returns:
        The generated text string from ollama.

    Raises:
        RuntimeError if ollama is unreachable or returns an unexpected response.
    """
    # Read the model name from the environment, falling back to the default.
    model = os.environ.get("MEMORY_OLLAMA_MODEL", _DEFAULT_MODEL)

    # Build the JSON request body as bytes.
    # stream=False tells ollama to return the full response in one HTTP reply.
    body = json.dumps({
        "model":  model,
        "prompt": prompt_text,
        "stream": False,
    }).encode("utf-8")

    # Build the HTTP POST request with the correct Content-Type header.
    req = urllib.request.Request(
        _OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # urlopen() sends the HTTP POST and blocks for up to 30 seconds.
        # If ollama is not running, this raises urllib.error.URLError immediately.
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        # Re-raise as RuntimeError so callers have a single exception type.
        raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    # Parse the JSON response. Ollama returns {"response": "...", "done": true, ...}
    data = json.loads(raw)

    # Check that the expected "response" key is present.
    if "response" not in data:
        raise RuntimeError(
            f"Unexpected ollama response (missing 'response' key): {raw[:200]}"
        )

    return data["response"]


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def re_extract_facts(conn, session: dict) -> None:
    """
    Extract factual statements from a session's transcript using ollama.

    The prompt asks ollama for a JSON array of facts.  Each returned fact is
    inserted into the facts table with source="daemon".  Duplicate content is
    detected by comparing against all existing fact content strings — only new
    content is inserted.

    Args:
        conn    — open SQLite connection from init_db()
        session — dict with at least "session_id" and "transcript" keys
                  (as returned by get_unprocessed_sessions)
    """
    session_id = session["session_id"]

    # Parse the stored transcript JSON into a list of turn dicts.
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        # If the transcript is malformed, skip this session gracefully.
        turns = []

    # Build a flat text representation of the conversation (first 3000 chars).
    # Tool-call turns have non-string content — skip those.
    lines = []
    for turn in turns:
        role    = turn.get("role", "unknown")
        content = turn.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")

    transcript_text = "\n".join(lines)[:3000]

    # The prompt instructs the model to return ONLY a JSON array of facts.
    prompt = (
        "Extract factual statements worth remembering from this conversation. "
        "Focus on: user preferences, technical decisions, recurring patterns, "
        "and domain knowledge.\n\n"
        "Return ONLY a JSON array with no other text. "
        'Each item: {"content": "...", "tags": ["tag1", "tag2"]}. Maximum 5 facts.\n\n'
        f"Conversation:\n{transcript_text}"
    )

    # Call ollama to generate the fact list.
    # If ollama fails, we log the error and return without inserting anything.
    try:
        raw_response = _call_ollama(prompt)
    except RuntimeError as exc:
        error_log("daemon", f"ollama call failed for session {session_id}: {exc}", exc=exc)
        return

    # Parse the JSON array from the response.
    # The model may include extra text before/after the JSON — we try to
    # extract the first [...] block if a plain parse fails.
    try:
        facts_data = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response.
        start = raw_response.find("[")
        end   = raw_response.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                facts_data = json.loads(raw_response[start:end])
            except json.JSONDecodeError:
                facts_data = []
        else:
            facts_data = []

    # Load all existing fact content strings so we can skip duplicates.
    # list_facts returns all facts; we only care about content strings.
    existing_facts = list_facts(conn, limit=10000)
    existing_contents = {f["content"] for f in existing_facts}

    # Insert each new fact, skipping any whose content already exists.
    inserted = 0
    for item in facts_data:
        # Each item must be a dict with at least a "content" key.
        if not isinstance(item, dict):
            continue
        content = item.get("content", "").strip()
        tags    = item.get("tags", [])
        if not content:
            continue
        if content in existing_contents:
            # Duplicate — skip to avoid re-inserting the same fact.
            continue
        # Insert the fact with source="daemon" to distinguish from agent-saved facts.
        insert_fact(conn, content, tags=tags, source="daemon", session_id=session_id)
        existing_contents.add(content)
        inserted += 1

    # Log the outcome so the activity log shows what was found.
    activity_log("daemon", "extract_facts", session=session_id, facts_found=inserted)
    _daemon_log(f"extracted {inserted} facts from session {session_id}")


def update_topic_clusters(conn, session: dict) -> None:
    """
    Embed the session transcript and assign it to the nearest topic cluster.

    If sentence-transformers is not installed, this function skips silently
    (wrapped in try/except ImportError) so the daemon can still run on
    machines without the embedding library.

    Args:
        conn    — open SQLite connection from init_db()
        session — dict with at least "session_id" and "transcript" keys
    """
    session_id = session["session_id"]

    # Import embed here so we can catch ImportError if sentence-transformers
    # is not installed on this machine.
    try:
        from memory.db import embed
    except ImportError:
        # sentence-transformers not installed — skip clustering silently.
        return

    # Parse the transcript to build the text we'll embed.
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []

    # Build a short text representation for the cluster embedding.
    # We use the first 500 chars of the joined transcript as the "session summary".
    lines = []
    for turn in turns:
        content = turn.get("content", "")
        if isinstance(content, str):
            lines.append(content)
    session_text = "\n".join(lines)[:500]

    # Derive a short label for a potentially new cluster (first 40 chars of text).
    label = session_text[:40].replace("\n", " ").strip()

    try:
        # embed() may raise ImportError (if sentence-transformers is gone) or
        # any other exception — we catch all to avoid crashing the daemon.
        embedding = embed(session_text)
    except ImportError:
        # sentence-transformers not installed — skip.
        return
    except Exception as exc:
        error_log("daemon", f"embedding failed for session {session_id}: {exc}", exc=exc)
        return

    # Assign the session to the nearest cluster (or create a new one).
    cluster_id = assign_to_cluster(conn, session_id, embedding, label=label)

    activity_log("daemon", "cluster", session=session_id, cluster_id=cluster_id)
    _daemon_log(f"assigned session {session_id} to cluster {cluster_id}")


# ---------------------------------------------------------------------------
# Cross-session insight generation
# ---------------------------------------------------------------------------

def generate_cross_session_insights(conn) -> None:
    """
    Gather recent facts and session summaries, then call ollama to identify
    cross-session patterns about the user's preferences, recurring topics, or skills.

    Each returned insight is stored in the insights table via upsert_insight.

    Args:
        conn — open SQLite connection from init_db()
    """
    # Gather up to 20 recent facts.
    facts = list_facts(conn, limit=20)
    if not facts:
        # Nothing to analyse — return without calling ollama.
        _daemon_log("skipped cross-session insights: no facts stored yet")
        return

    # Gather up to 10 recent session summaries from the summaries table.
    summary_rows = conn.execute(
        """
        SELECT session_id, summary
        FROM summaries
        ORDER BY created_at DESC
        LIMIT 10
        """
    ).fetchall()

    # Format the facts and summaries into text blocks for the prompt.
    facts_text = "\n".join(f"- {f['content']}" for f in facts)
    summaries_text = "\n".join(
        f"- {row['summary']}" for row in summary_rows
    ) or "(none)"

    # The prompt instructs the model to return ONLY a JSON array of insights.
    prompt = (
        "Based on these facts and conversation summaries, identify 2-3 cross-session "
        "patterns about this user's preferences, recurring topics, or skills.\n\n"
        "Return ONLY a JSON array. "
        'Each item: {"insight_type": "pattern"|"preference"|"skill", "content": "...", '
        '"confidence": 0.0-1.0}\n\n'
        f"Facts:\n{facts_text}\n\n"
        f"Recent summaries:\n{summaries_text}"
    )

    try:
        raw_response = _call_ollama(prompt)
    except RuntimeError as exc:
        error_log("daemon", f"ollama call failed for cross-session insights: {exc}", exc=exc)
        raise  # re-raise so the caller's except block can log it

    # Parse the JSON array from the response.
    try:
        insights_data = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to find a JSON array in the response.
        start = raw_response.find("[")
        end   = raw_response.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                insights_data = json.loads(raw_response[start:end])
            except json.JSONDecodeError:
                insights_data = []
        else:
            insights_data = []

    # Insert each insight into the database.
    count = 0
    for item in insights_data:
        if not isinstance(item, dict):
            continue
        insight_type = item.get("insight_type", "pattern")
        content      = item.get("content", "").strip()
        confidence   = float(item.get("confidence", 0.5))
        if not content:
            continue
        upsert_insight(conn, insight_type, content, evidence=[], confidence=confidence)
        count += 1

    activity_log("daemon", "insights", count=count)
    _daemon_log(f"generated {count} cross-session insights")


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def run(once: bool = False) -> None:
    """
    Run the daemon's main polling loop.

    Each iteration:
      1. Checks CPU load — skips the cycle if load > CPU_THRESHOLD.
      2. Opens the database and fetches unprocessed sessions.
      3. For each session: extracts facts, assigns a cluster, marks processed.
      4. Every INSIGHT_EVERY_N sessions: generates cross-session insights.
      5. Closes the DB and sleeps for POLL_INTERVAL (or LONG_POLL_INTERVAL if idle).

    Args:
        once — if True, run exactly one pass then return (useful for testing and CLI).
    """
    global _shutdown

    # Reset shutdown flag at the start of each run() call so tests can call
    # run(once=True) multiple times without the flag being sticky.
    _shutdown = False

    # Counter that triggers cross-session insight generation every INSIGHT_EVERY_N sessions.
    sessions_since_insight = 0

    _daemon_log("daemon started")

    while not _shutdown:
        # ------------------------------------------------------------------ #
        # CPU check: skip heavy inference if the machine is already busy.     #
        # ------------------------------------------------------------------ #
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
        except ImportError:
            # psutil not installed — assume CPU is fine and continue.
            cpu = 0

        if cpu > CPU_THRESHOLD:
            activity_log("daemon", "skip_cycle", reason="high_cpu", cpu=cpu)
            _daemon_log(f"skipping cycle: CPU at {cpu:.1f}%")
            if once:
                break
            time.sleep(60)
            continue

        # ------------------------------------------------------------------ #
        # Open DB and fetch unprocessed sessions.                             #
        # ------------------------------------------------------------------ #
        conn = init_db(DB_PATH)
        sessions = get_unprocessed_sessions(conn, limit=10)

        if sessions:
            # Process each unprocessed session in turn.
            for session in sessions:
                if _shutdown:
                    # SIGTERM received — stop cleanly between sessions.
                    break

                session_id = session["session_id"]
                _daemon_log(f"processing session {session_id}")

                try:
                    re_extract_facts(conn, session)
                    update_topic_clusters(conn, session)
                    mark_session_processed(conn, session_id)
                    sessions_since_insight += 1
                except Exception as exc:
                    error_log(
                        "daemon",
                        f"processing failed for {session_id}: {exc}",
                        exc=exc,
                    )
                    _daemon_log(f"error processing session {session_id}: {exc}")
                    # Skip this session — it will remain unprocessed and can be
                    # retried on the next run.

            # Every INSIGHT_EVERY_N sessions, generate cross-session insights.
            if sessions_since_insight >= INSIGHT_EVERY_N:
                try:
                    generate_cross_session_insights(conn)
                    sessions_since_insight = 0
                except Exception as exc:
                    error_log("daemon", "cross-session insight generation failed", exc=exc)
                    _daemon_log(f"insight generation error: {exc}")

            # Sleep for the normal interval since there was work to do.
            poll = POLL_INTERVAL
        else:
            # Nothing to process — back off to the long idle interval.
            _daemon_log("no new sessions; backing off")
            poll = LONG_POLL_INTERVAL

        conn.close()

        # In once mode, exit after the first processing pass regardless of result.
        if once:
            break

        # Sleep until the next polling cycle (or until SIGTERM wakes us).
        time.sleep(poll)

    _daemon_log("daemon stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # Parse command-line arguments.
    # --once makes the daemon process all pending sessions and exit immediately.
    # Without --once it runs forever until SIGTERM.
    parser = argparse.ArgumentParser(
        description="Background relearning daemon for the agentic memory system."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one polling pass then exit (for testing or manual runs).",
    )
    args = parser.parse_args()

    run(once=args.once)
