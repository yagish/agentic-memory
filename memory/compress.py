# compress.py — compress all sessions into a single structured memory document.
#
# Reads every session (using per-session summaries where available, raw
# transcripts otherwise), calls ollama in map-reduce fashion to produce
# intermediate summaries, then synthesises everything into a final structured
# memory document. The sessions that were compressed are deleted afterward.
#
# Triggers:
#   CLI:    python3 memory/compress.py [--dry-run] [--model MODEL]
#   HTTP:   POST /compress on query_server (7748) or ingest_server (7747)
#   Daemon: automatically every COMPRESS_EVERY_N sessions
#
# Output:
#   DB:     compressed_memory table (each run appends a row; latest = current)
#   File:   ~/.memory/compressed_memory.md  (human-readable copy)

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

# Add project root to path so this file can be run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    init_db,
    list_insights,
    get_all_sessions_for_compression,
    save_compressed_memory,
    get_compressed_memory,
    delete_sessions,
)
from memory.logger import activity_log, error_log


# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH     = os.path.expanduser("~/.memory/memory.db")
OUTPUT_PATH = os.path.expanduser("~/.memory/compressed_memory.md")

_OLLAMA_URL    = "http://localhost:11434/api/generate"
_DEFAULT_MODEL = os.environ.get("MEMORY_OLLAMA_MODEL", "llama3.2:3b")

# Number of sessions to fold into each intermediate summary call.
_BATCH_SIZE = 10

# Max characters taken from each session (summary or transcript) per batch.
_SNIPPET_LEN = 800

# Ollama request timeout — larger than the daemon's 30 s because compression
# prompts are longer and each intermediate call may take 1-2 minutes.
_TIMEOUT = 120


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class CompressResult:
    """Returned by compress_memory() regardless of dry_run mode."""
    content: str              # the full structured markdown document
    sessions_compressed: int  # how many sessions were folded in
    model: str                # ollama model used
    created_at: str           # ISO UTC timestamp of this run


# ── Ollama lifecycle ──────────────────────────────────────────────────────────

def _ensure_ollama() -> None:
    """
    Confirm ollama is reachable. If not, start it via `open -a Ollama` (macOS)
    and wait up to 30 s. Raises RuntimeError if it never comes up.
    """
    def _is_up() -> bool:
        # Check if ollama API is responding.
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2):
                return True
        except Exception:
            return False

    if _is_up():
        return

    # Ollama is not running — try to launch it on macOS.
    try:
        subprocess.Popen(["open", "-a", "Ollama"])
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not running and could not be started: {exc}"
        ) from exc

    # Poll until it is ready (up to 30 s).
    for _ in range(30):
        time.sleep(1)
        if _is_up():
            return

    raise RuntimeError(
        "Ollama was launched but did not become ready within 30 seconds."
    )


# ── Ollama client ─────────────────────────────────────────────────────────────

def _call_ollama(prompt: str, model: str) -> str:
    """
    Send a prompt to the local ollama API and return the generated text.

    Uses Python's built-in urllib so there are no extra dependencies.
    Times out after _TIMEOUT seconds to avoid blocking indefinitely.

    Raises:
        RuntimeError if ollama is unreachable or returns an unexpected shape.
    """
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        _OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unavailable: {exc}") from exc
    if "response" not in data:
        raise RuntimeError(f"Unexpected ollama response: {str(data)[:200]}")
    return data["response"]


# ── Text helpers ──────────────────────────────────────────────────────────────

def _session_text(session: dict) -> str:
    """
    Return the best available text for a session.

    Prefers the per-session summary (already compressed by the daemon) over
    the raw transcript. If neither is usable, returns an empty string.
    """
    if session.get("summary"):
        return session["summary"].strip()[:_SNIPPET_LEN]

    # Fall back to extracting text turns from the raw transcript JSON.
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except Exception:
        return ""

    lines = []
    for t in turns:
        content = t.get("content", "")
        if isinstance(content, str) and content.strip():
            role = t.get("role", "?")
            lines.append(f"{role}: {content}")
    return "\n".join(lines)[:_SNIPPET_LEN]


# ── Compression stages ────────────────────────────────────────────────────────

def _intermediate_summary(session_texts: list[str], model: str) -> str:
    """
    Ask ollama to distil a batch of session texts into key bullet points.

    This is the "map" step of the map-reduce compression. Each batch of up to
    _BATCH_SIZE sessions is condensed into a short list of things worth keeping.
    """
    numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(session_texts))
    prompt = (
        "You are compressing conversation history into a permanent memory. "
        "From these session summaries, extract ONLY what is worth remembering long-term:\n"
        "- User facts (name, role, expertise, preferences)\n"
        "- Technical decisions and project context\n"
        "- Recurring patterns and working style\n\n"
        "Write concise bullet points only. No preamble, no section headers.\n\n"
        f"Sessions:\n{numbered}"
    )
    return _call_ollama(prompt, model)


def _final_compress(
    intermediates: list[str],
    insights: list[str],
    previous: str | None,
    session_count: int,
    model: str,
) -> str:
    """
    Synthesise intermediate summaries + insights + prior memory into one document.

    This is the "reduce" step. The output is a structured markdown document
    that can be handed to a new agent as a complete memory bootstrap.

    The five sections are fixed so downstream tools can reliably parse them.
    """
    # Assemble source material.
    parts = []
    if previous:
        parts.append(f"=== PRIOR COMPRESSED MEMORY ===\n{previous.strip()}")
    parts.append("=== NEW SESSION SUMMARIES ===\n" + "\n\n---\n\n".join(intermediates))
    if insights:
        parts.append("=== CROSS-SESSION INSIGHTS ===\n" + "\n".join(f"- {i}" for i in insights))
    context = "\n\n".join(parts)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prior_note = ", building on prior compressed memory" if previous else ""

    prompt = (
        f"Create a permanent memory document synthesising {session_count} conversation sessions.\n\n"
        "Output EXACTLY this markdown structure and nothing else:\n\n"
        f"# Memory — Compressed on {date_str}\n"
        f"> Synthesized from {session_count} sessions{prior_note}\n\n"
        "## About the User\n"
        "[role, expertise, background, personal facts worth remembering]\n\n"
        "## Working Style & Preferences\n"
        "[how they like to work, communication style, tooling choices, code preferences]\n\n"
        "## Active Projects & Context\n"
        "[current work, goals, domain context, tech stack]\n\n"
        "## Key Decisions & Facts\n"
        "[important technical choices, domain facts, constraints worth remembering]\n\n"
        "## Recurring Patterns\n"
        "[common problems, repeated questions, habitual approaches, themes]\n\n"
        "Fill each section with concise bullets drawn from the source material. "
        "Write '(none noted)' for any section with nothing to say. "
        "Do not include any text outside the above structure.\n\n"
        f"Source material:\n\n{context}"
    )
    return _call_ollama(prompt, model)


# ── Main function ─────────────────────────────────────────────────────────────

def compress_memory(
    conn,
    model: str | None = None,
    dry_run: bool = False,
) -> CompressResult:
    """
    Compress all sessions into a structured memory document, then delete them.

    Steps:
      1. Load all sessions (with summaries where available).
      2. Load previous compressed memory (for incremental runs).
      3. Load cross-session insights for extra context.
      4. Map: call ollama on batches of _BATCH_SIZE sessions → intermediate summaries.
      5. Reduce: synthesise intermediates + insights + previous → final document.
      6. Save to DB and to ~/.memory/compressed_memory.md.
      7. Delete the sessions that were folded in.

    Args:
        conn    — open DB connection from init_db()
        model   — ollama model to use (falls back to MEMORY_OLLAMA_MODEL env var)
        dry_run — if True, return the result without writing to DB or deleting sessions

    Returns:
        CompressResult with the full markdown content and metadata.

    Raises:
        RuntimeError if there are no sessions to compress or ollama is unreachable.
    """
    m = model or _DEFAULT_MODEL

    # Ensure ollama is running before making any LLM calls.
    _ensure_ollama()

    # Step 1: Load sessions.
    sessions = get_all_sessions_for_compression(conn)
    if not sessions:
        raise RuntimeError("No sessions to compress.")

    # Step 2: Load prior compressed memory for incremental runs.
    previous_row = get_compressed_memory(conn)
    previous_content = previous_row["content"] if previous_row else None

    # Step 3: Load cross-session insights as additional context.
    insight_rows = list_insights(conn, limit=30)
    insights = [r["content"] for r in insight_rows]

    # Step 4 (Map): process sessions in batches → intermediate summaries.
    intermediates: list[str] = []
    for i in range(0, len(sessions), _BATCH_SIZE):
        batch = sessions[i : i + _BATCH_SIZE]
        texts = [t for t in (_session_text(s) for s in batch) if t.strip()]
        if not texts:
            continue
        summary = _intermediate_summary(texts, m)
        intermediates.append(summary)

    if not intermediates:
        raise RuntimeError("No usable session content found — all transcripts are empty.")

    # Step 5 (Reduce): synthesise into one structured document.
    final_content = _final_compress(intermediates, insights, previous_content, len(sessions), m)

    now = datetime.now(timezone.utc).isoformat()
    result = CompressResult(
        content=final_content,
        sessions_compressed=len(sessions),
        model=m,
        created_at=now,
    )

    if dry_run:
        return result

    # Step 6: Persist to DB and write the human-readable file.
    save_compressed_memory(conn, final_content, len(sessions), m)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(final_content)

    # Step 7: Delete all sessions that were folded into the compressed document.
    session_ids = [s["session_id"] for s in sessions]
    deleted = delete_sessions(conn, session_ids)

    activity_log(
        "compress", "compress_memory",
        sessions=len(sessions), deleted=deleted, model=m,
    )

    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compress all sessions into a structured memory document."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the output without writing to the DB or deleting sessions.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Ollama model to use (default: llama3.2:3b or MEMORY_OLLAMA_MODEL env var).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = init_db(DB_PATH)
    try:
        result = compress_memory(conn, model=args.model, dry_run=args.dry_run)
        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"\n{prefix}Compressed {result.sessions_compressed} sessions using {result.model}.\n")
        print(result.content)
        if not args.dry_run:
            print(f"\nSaved to {OUTPUT_PATH}")
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()
