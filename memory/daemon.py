# daemon.py — background memory compaction daemon.
#
# Runs as a long-lived process. Every POLL_INTERVAL seconds:
#   Pass 1 — per new session:
#     1. Creates an episodic memory entry (title + abstract)
#     2. Assigns session to a topic cluster
#     3. Updates working memory for that cluster
#     4. Compacts the cluster into a vector-indexed summary (when ≥ COMPACT_MIN_SESSIONS ready)
#     5. Extracts reusable procedural patterns
#     6. Extracts durable cross-session insights
#     7. Extracts durable facts about the user (identity, preferences, decisions)
#   Pass 2 — periodic maintenance (every PERIODIC_EVERY_N sessions):
#     8. Closes working memory entries inactive > 14 days
#     9. Merges near-duplicate compacted session entries (≥ 92% similarity)
#
# Run:
#   python3 memory/daemon.py            # runs forever
#   python3 memory/daemon.py --once     # one pass then exit (for tests)
#
# Stop: SIGTERM — the daemon exits cleanly after the current session.
#
# IMPORTANT: no Claude API or anthropic package is used here.
# All LLM calls go to the local ollama server via plain HTTP (urllib.request).

import json
import os
import psutil
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    bootstrap_db,
    open_db,
    get_unprocessed_sessions,
    mark_session_processed,
    assign_to_cluster,
    get_cluster_sessions,
    insert_episodic,
    upsert_working_memory,
    get_stale_working_memory,
    close_working_memory,
    upsert_compacted_session,
    get_near_duplicate_compacted,
    merge_compacted_sessions,
    upsert_procedural,
    upsert_insight,
    insert_fact,
    prune_transcript,
    delete_sessions,
    embed,
)
from memory.logger import activity_log, error_log
from memory.debug import enable_debug
from memory.ollama import (
    call_ollama as _call_ollama,
    start_ollama_if_needed as _start_ollama_if_needed,
    stop_ollama as _stop_ollama,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH           = os.path.expanduser("~/.memory/memory.db")
_DAEMON_LOG_PATH  = os.path.expanduser("~/.memory/daemon.log")

POLL_INTERVAL      = 5 * 60    # seconds between cycles when work exists
LONG_POLL_INTERVAL = 30 * 60   # seconds between cycles when idle
CPU_THRESHOLD      = 70         # skip LLM work above this CPU %
RETAIN_PROCESSED_DAYS = 30      # delete processed sessions older than this
# Compact as soon as a cluster has an uncompacted session — most real sessions land in
# singleton clusters, so waiting for 3+ meant compaction (and wake-up cache hits) never ran.
COMPACT_MIN_SESSIONS  = 1
INSIGHT_MIN_CONFIDENCE = 0.6    # discard low-confidence insight extractions
FACT_MIN_CONFIDENCE   = 0.6    # discard low-confidence fact extractions
PERIODIC_EVERY_N      = 20      # run Pass 2 maintenance every N processed sessions
WORKING_MEMORY_TTL    = 14      # days before working memory is considered stale

_shutdown = False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_sigterm(signum, frame):
    # Ask the polling loop to stop at its next safe checkpoint.
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _daemon_log(message: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(_DAEMON_LOG_PATH), exist_ok=True)
        with open(_DAEMON_LOG_PATH, "a") as f:
            f.write(f"{ts} [daemon] {message}\n")
    except Exception:
        pass


def _parse_json_from(raw: str) -> list | dict:
    """Extract the first JSON array or object from an ollama response."""
    # Accept both strict JSON responses and responses wrapped in explanatory text.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = raw.find(start_char)
            end   = raw.rfind(end_char) + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass
    return []


def _call_ollama_json(prompt: str, expected_type: type, retries: int = 1):
    """
    Call ollama and parse a JSON value of `expected_type` from the response.

    qwen2.5:3b frequently wraps JSON in prose or emits malformed JSON. On a
    type mismatch we retry once with a stricter instruction appended before
    giving up, instead of silently discarding the (possibly valid) output.

    Returns the parsed value, or None if no valid response was obtained.
    """
    attempt_prompt = prompt
    for attempt in range(retries + 1):
        raw = _call_ollama(attempt_prompt, log_fn=_daemon_log)
        data = _parse_json_from(raw)
        if isinstance(data, expected_type):
            return data
        attempt_prompt = (
            f"{prompt}\n\nReturn ONLY valid JSON matching the format above. "
            "No prose, no markdown fences, no explanation."
        )
    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_session_text(turns: list, max_chars: int | None = None) -> str:
    """Concatenate turn content into a readable text sample."""
    # Preserve full conversation context with role labels for the local model.
    lines = []
    chars = 0
    for turn in turns:
        content = turn.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        role = turn.get("role", "")
        line = f"{role}: {content}"
        if max_chars is not None and chars + len(line) > max_chars:
            break
        lines.append(line)
        chars += len(line)
    return "\n".join(lines)


def _summary_target_chars(turn_count: int) -> int | None:
    """Return target char length for a compacted summary, or None to skip."""
    if turn_count < 5:
        return None
    if turn_count <= 15:
        return 600    # ~150 tokens
    if turn_count <= 40:
        return 1600   # ~400 tokens
    return 3200       # ~800 tokens


def _session_text_sample(session: dict, max_chars: int | None = None) -> str:
    """Prepare full transcript for per-session LLM prompts."""
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []
    return _build_session_text(turns, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Pass 1 — per-session processing
# ---------------------------------------------------------------------------

def _create_episodic_entry(
    conn,
    session: dict,
    text_sample: str | None = None,
) -> tuple[str, str]:
    """
    Generate a title and 2-sentence abstract for the session and store it
    in episodic_memory. Returns (title, abstract) or ("", "") on failure.
    """
    # Turn the raw transcript into a searchable title and short event summary.
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return "", ""

    prompt = (
        "Summarize this conversation:\n"
        "1. TITLE: One line, max 10 words, describing what was done.\n"
        "2. ABSTRACT: Two sentences — what happened and what was resolved.\n\n"
        'Return ONLY a JSON object: {"title": "...", "abstract": "..."}\n\n'
        f"Conversation sample:\n{text_sample}"
    )

    try:
        data = _call_ollama_json(prompt, dict)
        if data is None:
            return "", ""
        title    = data.get("title", "").strip()[:200]
        abstract = data.get("abstract", "").strip()[:500]
        if title and abstract:
            happened_at = session.get("updated_at") or datetime.now(timezone.utc).isoformat()
            insert_episodic(conn, session_id, title, abstract, happened_at)
            _daemon_log(f"episodic entry created for {session_id}: {title}")
            return title, abstract
    except Exception as exc:
        error_log("daemon", f"episodic creation failed for {session_id}: {exc}", exc=exc)

    return "", ""


def _assign_cluster(conn, session: dict) -> str | None:
    """Embed the session and assign it to the nearest topic cluster."""
    # Use a short transcript sample so clustering stays cheap and consistent.
    session_id = session["session_id"]
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []

    lines = [
        t.get("content", "") for t in turns
        if isinstance(t.get("content"), str)
    ]
    session_text = "\n".join(lines)[:500]
    label = session_text[:40].replace("\n", " ").strip()

    try:
        embedding = embed(session_text)
        cluster_id = assign_to_cluster(conn, session_id, embedding, label=label)
        activity_log("daemon", "cluster", session=session_id, cluster_id=cluster_id)
        _daemon_log(f"assigned {session_id} to cluster {cluster_id}")
        return cluster_id
    except Exception as exc:
        error_log("daemon", f"cluster assignment failed for {session_id}: {exc}", exc=exc)
        return None


def _compact_cluster_if_ready(conn, cluster_id: str) -> None:
    """
    Compact all sessions in a cluster that still have transcripts into one
    structured summary stored in compacted_sessions with a vector.

    Triggers only when ≥ COMPACT_MIN_SESSIONS sessions have uncompacted transcripts.
    Source session transcripts are nulled out after successful compaction.
    """
    # Compact only complete groups of related sessions, leaving small groups intact.
    session_ids = get_cluster_sessions(conn, cluster_id)
    if not session_ids:
        return

    # Find sessions in this cluster that still have transcript content.
    placeholders = ",".join("?" * len(session_ids))
    rows = conn.execute(
        f"""
        SELECT session_id, transcript, turn_count
        FROM sessions
        WHERE session_id IN ({placeholders})
          AND transcript IS NOT NULL AND transcript != '[]'
        """,
        session_ids,
    ).fetchall()

    if len(rows) < COMPACT_MIN_SESSIONS:
        return

    # Gather all turns from uncompacted sessions.
    all_turns = []
    source_ids = []
    for row in rows:
        try:
            turns = json.loads(row["transcript"] or "[]")
            all_turns.extend(turns)
            source_ids.append(row["session_id"])
        except json.JSONDecodeError:
            continue

    total_turns = len(all_turns)
    target_chars = _summary_target_chars(total_turns)
    if target_chars is None:
        return

    text_sample = _build_session_text(all_turns, max_chars=min(target_chars * 2, 6000))

    prompt = (
        f"Summarize these {len(source_ids)} related work sessions into a structured note.\n"
        f"Target length: approximately {target_chars // 4} words.\n\n"
        "Use EXACTLY this format:\n"
        "Task: [one line — what the work was about]\n"
        "Context: [repo, language, key components]\n"
        "What was tried: [bullet points]\n"
        "Outcome: [what worked / current state]\n"
        "Left off at: [where to pick up next time]\n\n"
        f"Sessions:\n{text_sample}"
    )

    try:
        summary = _call_ollama(prompt, log_fn=_daemon_log)
        summary = summary.strip()
        if not summary:
            return

        try:
            summary_vec = embed(summary)
        except Exception:
            summary_vec = None

        upsert_compacted_session(conn, cluster_id, summary, summary_vec, source_ids)

        # Null out the source session transcripts — they are now compacted.
        for sid in source_ids:
            prune_transcript(conn, sid)

        activity_log(
            "daemon", "compact",
            cluster_id=cluster_id,
            sessions=len(source_ids),
            turns=total_turns,
        )
        _daemon_log(
            f"compacted {len(source_ids)} sessions ({total_turns} turns) for cluster {cluster_id}"
        )
    except Exception as exc:
        error_log("daemon", f"compaction failed for cluster {cluster_id}: {exc}", exc=exc)


def _extract_procedural_patterns(
    conn,
    session: dict,
    text_sample: str | None = None,
) -> None:
    """
    Identify reusable how-to patterns from a session and store them in
    procedural_memory. Only generalizable procedures are stored.
    """
    # Ask the local model to retain only workflows that generalize beyond this session.
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return

    prompt = (
        "Identify any reusable how-to patterns or workflows in this conversation.\n"
        "Only include patterns useful for FUTURE sessions. Skip one-off actions.\n\n"
        'Return ONLY a JSON array: [{"title": "Short name", "steps": "1. Step\\n2. Step"}]\n'
        "Return [] if no reusable patterns exist.\n\n"
        f"Conversation:\n{text_sample}"
    )

    try:
        data = _call_ollama_json(prompt, list)
        if not data:
            return

        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            title = item.get("title", "").strip()[:200]
            steps = item.get("steps", "").strip()
            if title and steps:
                upsert_procedural(conn, title, steps)
                count += 1

        if count:
            activity_log("daemon", "procedural", session=session_id, patterns=count)
            _daemon_log(f"extracted {count} procedural patterns from {session_id}")
    except Exception as exc:
        error_log("daemon", f"procedural extraction failed for {session_id}: {exc}", exc=exc)


def _extract_insight_patterns(
    conn,
    session: dict,
    text_sample: str | None = None,
) -> None:
    """
    Identify a durable cross-session insight (recurring mistake, strong
    preference, or skill gap) from a session and store it in insights.
    Only high-confidence, generalizable observations are kept.
    """
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return

    prompt = (
        "Identify ONE durable cross-session insight about the user or their work from "
        "this conversation \u2014 e.g. a recurring mistake, a strong preference, or a skill gap. "
        "Skip anything only relevant to this single session.\n\n"
        'Return ONLY a JSON object: {"insight_type": "preference|pattern|skill_gap", '
        '"content": "...", "confidence": 0.0-1.0}\n'
        "Return {} if no durable insight exists.\n\n"
        f"Conversation:\n{text_sample}"
    )

    try:
        data = _call_ollama_json(prompt, dict)
        if not data:
            return

        insight_type = data.get("insight_type", "").strip()[:50]
        content = data.get("content", "").strip()[:500]
        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = None

        if insight_type and content and (confidence is None or confidence >= INSIGHT_MIN_CONFIDENCE):
            upsert_insight(conn, insight_type, content, [session_id], confidence)
            activity_log("daemon", "insight", session=session_id, insight_type=insight_type)
            _daemon_log(f"extracted insight ({insight_type}) from {session_id}")
    except Exception as exc:
        error_log("daemon", f"insight extraction failed for {session_id}: {exc}", exc=exc)


def _extract_facts(
    conn,
    session: dict,
    text_sample: str | None = None,
) -> None:
    """
    Identify durable facts about the user (identity, preferences, decisions) from
    a session and store them in facts. Only clearly-stated, generalizable facts
    are kept — skips anything only relevant to this single session.
    """
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return

    prompt = (
        "Identify any durable facts about the user worth remembering permanently — "
        "their name, role, background, preferences, or key decisions they made. "
        "Skip anything only relevant to this single session.\n\n"
        'Return ONLY a JSON array: [{"content": "...", "tags": ["tag1"], "confidence": 0.0-1.0}]\n'
        "Return [] if no durable facts exist.\n\n"
        f"Conversation:\n{text_sample}"
    )

    try:
        data = _call_ollama_json(prompt, list)
        if not data:
            return

        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "").strip()[:500]
            tags = item.get("tags") or []
            if not isinstance(tags, list):
                tags = []
            confidence = item.get("confidence")
            if not isinstance(confidence, (int, float)):
                confidence = None
            if content and (confidence is None or confidence >= FACT_MIN_CONFIDENCE):
                insert_fact(conn, content, tags=tags, source="daemon", session_id=session_id)
                count += 1

        if count:
            activity_log("daemon", "fact", session=session_id, facts=count)
            _daemon_log(f"extracted {count} facts from {session_id}")
    except Exception as exc:
        error_log("daemon", f"fact extraction failed for {session_id}: {exc}", exc=exc)


# ---------------------------------------------------------------------------
# Pass 2 — periodic maintenance
# ---------------------------------------------------------------------------

def _close_stale_working_memory(conn) -> None:
    """
    Close working memory entries that have had no activity for WORKING_MEMORY_TTL days.
    Stamps closed_at on each entry after deciding if it warrants a final episodic note.
    """
    # Preserve significant completed work as an episode before closing stale context.
    stale = get_stale_working_memory(conn, days=WORKING_MEMORY_TTL)
    if not stale:
        return

    for wm in stale:
        try:
            prompt = (
                "This task context has been inactive for 14+ days:\n\n"
                f"{wm['summary'][:800]}\n\n"
                "In one sentence: was anything significant accomplished that should be remembered? "
                "Start with YES or NO."
            )
            raw = _call_ollama(prompt, log_fn=_daemon_log)
            if raw.strip().upper().startswith("YES"):
                title_prompt = (
                    f"Give a 5-8 word title for this completed task:\n{wm['summary'][:400]}"
                )
                title = _call_ollama(title_prompt, log_fn=_daemon_log).strip()[:150]
                insert_episodic(
                    conn, wm["cluster_id"],
                    title or "Completed task (working memory closed)",
                    "Working memory closed after 14 days of inactivity.",
                    None,
                )
            close_working_memory(conn, wm["id"])
            _daemon_log(f"closed stale working memory {wm['id']}")
        except Exception as exc:
            error_log("daemon", f"closing working memory {wm['id']} failed: {exc}", exc=exc)

    activity_log("daemon", "close_working_memory", closed=len(stale))


def _merge_near_duplicate_compacted(conn) -> None:
    """
    Find compacted session pairs with vector similarity ≥ 92% and merge them
    into a single entry via LLM. Limits to 3 merges per maintenance cycle.
    """
    # Collapse highly similar summaries so retrieval does not return duplicates.
    try:
        pairs = get_near_duplicate_compacted(conn, similarity_threshold=0.92)
    except Exception:
        return

    merged = 0
    for keep_id, drop_id in pairs[:3]:
        try:
            keep_row = conn.execute(
                "SELECT content FROM compacted_sessions WHERE id = ?", (keep_id,)
            ).fetchone()
            drop_row = conn.execute(
                "SELECT content FROM compacted_sessions WHERE id = ?", (drop_id,)
            ).fetchone()
            if not keep_row or not drop_row:
                continue

            prompt = (
                "Merge these two overlapping task summaries into one coherent note. "
                "Use the same structured format (Task / Context / What was tried / Outcome / Left off at).\n\n"
                f"--- Summary A ---\n{keep_row['content']}\n\n"
                f"--- Summary B ---\n{drop_row['content']}"
            )
            merged_content = _call_ollama(prompt, log_fn=_daemon_log).strip()
            if not merged_content:
                continue

            try:
                merged_vec = embed(merged_content)
            except Exception:
                merged_vec = None

            merge_compacted_sessions(conn, keep_id, drop_id, merged_content, merged_vec)
            merged += 1
            _daemon_log(f"merged compacted entries {drop_id} into {keep_id}")
        except Exception as exc:
            error_log("daemon", f"merge failed for ({keep_id}, {drop_id}): {exc}", exc=exc)

    if merged:
        activity_log("daemon", "merge_compacted", merged=merged)


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------

def prune_old_sessions(conn) -> None:
    """
    Delete processed sessions older than RETAIN_PROCESSED_DAYS days.
    Episodic entries, compacted summaries, and procedural patterns are kept.
    """
    # Remove raw processed sessions while retaining their derived memory records.
    rows = conn.execute(
        """
        SELECT session_id FROM sessions
        WHERE daemon_processed_at IS NOT NULL
          AND daemon_processed_at < DATETIME('now', ? || ' days')
        """,
        (f"-{RETAIN_PROCESSED_DAYS}",),
    ).fetchall()

    if not rows:
        return

    ids = [r["session_id"] for r in rows]
    deleted = delete_sessions(conn, ids)
    activity_log("daemon", "prune_sessions", deleted=deleted, retain_days=RETAIN_PROCESSED_DAYS)
    _daemon_log(f"pruned {deleted} sessions older than {RETAIN_PROCESSED_DAYS} days")


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def run(once: bool = False) -> None:
    """
    Run the daemon's main polling loop.

    Each iteration runs Pass 1 for all new sessions, then Pass 2 maintenance
    every PERIODIC_EVERY_N sessions. Backs off to LONG_POLL_INTERVAL when idle.

    Args:
        once — if True, run exactly one pass then return (for testing and CLI).
    """
    # Reset the signal-controlled flag whenever a new daemon run begins.
    global _shutdown
    _shutdown = False

    sessions_since_periodic = 0

    _daemon_log("daemon started")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    bootstrap_conn = bootstrap_db(DB_PATH)
    bootstrap_conn.close()

    while not _shutdown:
        # Each cycle handles new sessions first, then performs occasional maintenance.
        # CPU check: skip heavy LLM work if the machine is busy.
        cpu = psutil.cpu_percent(interval=1)

        if cpu > CPU_THRESHOLD:
            activity_log("daemon", "skip_cycle", reason="high_cpu", cpu=cpu)
            _daemon_log(f"skipping cycle: CPU at {cpu:.1f}%")
            if once:
                break
            time.sleep(60)
            continue

        conn = open_db(DB_PATH)
        # Fetch a batch of unprocessed sessions from the database.
        sessions = get_unprocessed_sessions(conn, limit=10)

        if sessions:
            _ollama_proc = _start_ollama_if_needed(log_fn=_daemon_log)

            for session in sessions:
                if _shutdown:
                    break

                session_id = session["session_id"]
                _daemon_log(f"processing session {session_id}")

                try:
                    # Reuse one bounded transcript sample across per-session LLM prompts.
                    text_sample = _session_text_sample(session)

                    # Step 1: Episodic entry (title + abstract).
                    title, abstract = _create_episodic_entry(
                        conn, session, text_sample=text_sample
                    )

                    # Step 2: Assign to topic cluster.
                    cluster_id = _assign_cluster(conn, session)

                    # Step 3: Update working memory for this cluster.
                    if cluster_id and title:
                        summary_addition = f"**{title}**\n{abstract}"
                        upsert_working_memory(conn, cluster_id, session_id, summary_addition)

                    # Step 4: Compact the cluster if enough sessions are ready.
                    if cluster_id:
                        _compact_cluster_if_ready(conn, cluster_id)

                    # Step 5: Extract procedural patterns.
                    _extract_procedural_patterns(
                        conn, session, text_sample=text_sample
                    )

                    # Step 6: Extract durable cross-session insights.
                    _extract_insight_patterns(
                        conn, session, text_sample=text_sample
                    )

                    # Step 7: Extract durable facts about the user.
                    _extract_facts(
                        conn, session, text_sample=text_sample
                    )

                    # Step 8: Only mark the session processed after all required
                    # per-session steps completed without bubbling an exception.
                    mark_session_processed(conn, session_id)

                    sessions_since_periodic += 1

                except Exception as exc:
                    error_log(
                        "daemon",
                        f"processing failed for {session_id}: {exc}",
                        exc=exc,
                    )
                    _daemon_log(f"error processing session {session_id}: {exc}")

            # Pass 2: periodic maintenance (close stale WM, merge near-duplicates).
            if sessions_since_periodic >= PERIODIC_EVERY_N:
                try:
                    _close_stale_working_memory(conn)
                    _merge_near_duplicate_compacted(conn)
                    sessions_since_periodic = 0
                except Exception as exc:
                    error_log("daemon", "periodic maintenance failed", exc=exc)
                    _daemon_log(f"maintenance error: {exc}")

            # Remove old processed sessions.
            try:
                prune_old_sessions(conn)
            except Exception as exc:
                error_log("daemon", "session pruning failed", exc=exc)

            if _ollama_proc is not None:
                _stop_ollama(_ollama_proc, log_fn=_daemon_log)

            poll = POLL_INTERVAL
        else:
            _daemon_log("no new sessions; backing off")
            poll = LONG_POLL_INTERVAL

        conn.close()

        if once:
            break

        time.sleep(poll)

    _daemon_log("daemon stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle("AgenticMemoryDaemon")
    enable_debug("daemon")
    import argparse

    parser = argparse.ArgumentParser(
        description="Background memory compaction daemon for the agentic memory system."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one polling pass then exit.",
    )
    args = parser.parse_args()

    run(once=args.once)
