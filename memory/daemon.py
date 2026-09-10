# daemon.py — background daemon for structured memory extraction.
#
# Runs as a long-lived process. Every poll cycle:
#   1. finds unprocessed sessions
#   2. extracts durable facts from the session
#   3. extracts one episodic memory from the session
#   4. extracts one procedural memory from the session
#   5. extracts one working-memory snapshot from the session
#   6. extracts one compacted session memory from the session
#   7. marks the session processed
#
# Run:
#   python3 memory/daemon.py        # runs forever
#   python3 memory/daemon.py --once # force one extraction pass now
#
# When --once is used, the daemon skips the CPU gate and immediately processes
# the current unprocessed batch.
#
# Stop: SIGTERM — the daemon exits cleanly after the current session.

from __future__ import annotations

import concurrent.futures
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    bootstrap_db,
    get_unprocessed_sessions,
    mark_session_processed,
    open_db,
    prune_stale_episodic,
    prune_stale_facts,
    save_session_compaction,
)
from memory.debug import enable_debug
from memory.episodic import extract_episode_from_session_text
from memory.episodic_repository import save_extracted_episode
from memory.fact_repository import build_fact_content, save_extracted_facts
from memory.facts import extract_facts_from_session_text, normalize_extracted_facts
from memory.logger import activity_log, error_log
from memory.procedural import extract_procedure_from_session_text
from memory.procedural_repository import save_extracted_procedure
from memory.session_memory import extract_session_memory_from_session_text
from memory.session_memory_repository import save_extracted_session_memory
from memory.working_memory import extract_working_memory_from_session_text
from memory.working_memory_repository import save_extracted_working_memory
from memory.ollama import (
    start_ollama_if_needed as _start_ollama_if_needed,
    stop_ollama as _stop_ollama,
)
from memory.vectors import embed


DB_PATH = os.path.expanduser("~/.memory/memory.db")
_DAEMON_LOG_PATH = os.path.expanduser("~/.memory/daemon.log")

POLL_INTERVAL = 5 * 60
LONG_POLL_INTERVAL = 30 * 60
CPU_THRESHOLD = 70
_MAX_SESSION_CHARS = int(os.environ.get("MEMORY_MAX_SESSION_CHARS", "60000"))
_FACT_TTL_DAYS = int(os.environ.get("MEMORY_FACT_TTL_DAYS", "180"))
_EPISODIC_TTL_DAYS = int(os.environ.get("MEMORY_EPISODIC_TTL_DAYS", "90"))
# Compaction: long sessions are summarized via Ollama before extraction.
# MEMORY_COMPACT_INPUT_CHARS caps how much raw transcript the summarizer sees
# (Ollama context limit). The summary is saved to the DB and reused on re-runs.
_COMPACT_INPUT_CHARS = int(os.environ.get("MEMORY_COMPACT_INPUT_CHARS", "40000"))
_COMPACT_TARGET_WORDS = int(os.environ.get("MEMORY_COMPACT_TARGET_WORDS", "800"))

_shutdown = False


def _handle_sigterm(signum, frame):
    del signum, frame
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def _should_write_log_file() -> bool:
    if os.environ.get("MEMORY_DISABLE_FILE_LOGS") == "1":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "pytest" in sys.modules:
        return False
    return True


def _daemon_log(message: str) -> None:
    if not _should_write_log_file():
        return
    ts = datetime.now(timezone.utc).isoformat()
    try:
        os.makedirs(os.path.dirname(_DAEMON_LOG_PATH), exist_ok=True)
        with open(_DAEMON_LOG_PATH, "a") as f:
            f.write(f"{ts} [daemon] {message}\n")
    except Exception:
        pass


def _build_session_text(turns: list[dict], max_chars: int | None = None) -> str:
    """Concatenate turn content into a readable text sample."""
    lines: list[str] = []
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


def _session_text_sample(session: dict) -> str:
    """Return the full raw session text with no size limit and no compaction.

    Used as a fallback when individual extractor functions are called directly
    (e.g. from tests or manual scripts) without a pre-built text_sample argument.
    The daemon's normal code path goes through _get_or_compact_session_text instead.
    """
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []
    return _build_session_text(turns)


def _compact_session_text(full_text: str) -> str:
    """Summarize a long session transcript via Ollama.

    Passes up to _COMPACT_INPUT_CHARS of the raw text to the model and asks for
    a dense summary. The summary is saved to the DB and reused on subsequent
    daemon cycles so the model is only called once per session.

    Raises InferenceError if Ollama is unavailable — callers decide how to handle.
    """
    from memory.inference import GenerationRequest, generate_text

    input_text = full_text[:_COMPACT_INPUT_CHARS]

    prompt = (
        "You are a session summarizer. Condense the following conversation transcript "
        "into a concise but complete summary.\n"
        "Include:\n"
        "- All key facts (names, settings, preferences, technical details)\n"
        "- Decisions made and their rationale\n"
        "- Steps taken or discussed\n"
        "- Outcomes reached and open questions\n"
        f"Target: under {_COMPACT_TARGET_WORDS} words. "
        "Output only the summary — no preamble, no closing remark.\n\n"
        "TRANSCRIPT:\n" + input_text
    )

    result = generate_text(GenerationRequest(prompt=prompt, timeout_seconds=120))
    _daemon_log(f"compacted session: {len(full_text)} → {len(result.text)} chars")
    return result.text


def _get_or_compact_session_text(conn, session: dict) -> str:
    """Return the text to feed to all extractors for this session.

    - Sessions already compacted: return the stored compacted_text directly.
    - Short sessions (under _MAX_SESSION_CHARS): return the full transcript text.
    - Long sessions not yet compacted: call Ollama to summarize, save the result
      to the DB for reuse, then return the summary.

    If Ollama fails during compaction the error propagates — the session stays
    unprocessed and will be retried on the next daemon cycle.
    """
    # Reuse a compaction saved from a previous daemon cycle.
    cached = session.get("compacted_text")
    if cached:
        _daemon_log(f"reusing stored compaction for {session['session_id']}")
        return cached

    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []

    full_text = _build_session_text(turns)

    if len(full_text) <= _MAX_SESSION_CHARS:
        return full_text

    compacted = _compact_session_text(full_text)
    save_session_compaction(conn, session["session_id"], compacted)
    return compacted


def _log_episode_details(session_id: str, episode) -> None:
    _daemon_log(
        f"episode extracted for {session_id}: title={episode.title} | abstract={episode.abstract}"
    )
    if getattr(episode, "decisions", None):
        _daemon_log(f"episode decisions for {session_id}: {'; '.join(episode.decisions)}")
    if getattr(episode, "outcomes", None):
        _daemon_log(f"episode outcomes for {session_id}: {'; '.join(episode.outcomes)}")
    if getattr(episode, "follow_ups", None):
        _daemon_log(f"episode follow-ups for {session_id}: {'; '.join(episode.follow_ups)}")


def _create_episodic_entry(conn, session: dict, text_sample: str | None = None):
    """Generate and persist one structured episodic memory for a session."""
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return None

    try:
        episode = extract_episode_from_session_text(
            text_sample,
            source="daemon",
            session_id=session_id,
        )
        happened_at = session.get("updated_at") or datetime.now(timezone.utc).isoformat()
        save_extracted_episode(
            conn,
            episode,
            session_id=session_id,
            happened_at=happened_at,
            source="daemon",
            embed_fn=embed,
        )
        activity_log("daemon", "episode", session=session_id, title=episode.title)
        _log_episode_details(session_id, episode)
        return episode
    except Exception as exc:
        error_log("daemon", f"episodic creation failed for {session_id}: {exc}", exc=exc)
        return None


def _extract_facts(conn, session: dict, text_sample: str | None = None) -> list[str]:
    """Extract structured facts from a session and persist them."""
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return []

    try:
        facts = normalize_extracted_facts(extract_facts_from_session_text(text_sample))
        if not facts:
            return []

        saved_ids = save_extracted_facts(
            conn,
            facts,
            session_id=session_id,
            source="daemon_fact_extractor",
        )
        if not saved_ids:
            return []

        fact_contents = [build_fact_content(fact) for fact in facts]
        activity_log("daemon", "fact", session=session_id, facts=len(saved_ids))
        for content in fact_contents:
            _daemon_log(f"fact extracted for {session_id}: {content}")
        _daemon_log(f"extracted {len(saved_ids)} structured facts from {session_id}")
        return fact_contents
    except Exception as exc:
        error_log("daemon", f"fact extraction failed for {session_id}: {exc}", exc=exc)
        return []



def _log_procedure_details(session_id: str, procedure) -> None:
    _daemon_log(
        f"procedure extracted for {session_id}: title={procedure.title} | summary={procedure.summary}"
    )
    if getattr(procedure, "steps", None):
        _daemon_log(f"procedure steps for {session_id}: {'; '.join(procedure.steps)}")
    if getattr(procedure, "trigger_phrases", None):
        _daemon_log(f"procedure triggers for {session_id}: {'; '.join(procedure.trigger_phrases)}")



def _create_procedural_entry(conn, session: dict, text_sample: str | None = None):
    """Generate and persist one structured procedural memory for a session."""
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return None

    try:
        procedure = extract_procedure_from_session_text(
            text_sample,
            source="daemon",
            session_id=session_id,
        )
        if procedure is None:
            _daemon_log(f"no procedural memory extracted for {session_id}")
            return None
        updated_at = session.get("updated_at") or datetime.now(timezone.utc).isoformat()
        save_extracted_procedure(
            conn,
            procedure,
            session_id=session_id,
            updated_at=updated_at,
            source="daemon",
            embed_fn=embed,
        )
        activity_log("daemon", "procedure", session=session_id, title=procedure.title)
        _log_procedure_details(session_id, procedure)
        return procedure
    except Exception as exc:
        error_log("daemon", f"procedural creation failed for {session_id}: {exc}", exc=exc)
        return None



def _log_working_memory_details(session_id: str, working_memory) -> None:
    _daemon_log(
        f"working memory extracted for {session_id}: goal={working_memory.current_goal} | next_step={working_memory.next_step} | status={working_memory.status}"
    )
    if getattr(working_memory, "active_tasks", None):
        _daemon_log(f"working memory tasks for {session_id}: {'; '.join(working_memory.active_tasks)}")
    if getattr(working_memory, "constraints", None):
        _daemon_log(f"working memory constraints for {session_id}: {'; '.join(working_memory.constraints)}")



def _create_working_memory_entry(conn, session: dict, text_sample: str | None = None):
    """Generate and persist one working-memory snapshot for a session."""
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return None

    try:
        working_memory = extract_working_memory_from_session_text(
            text_sample,
            source="daemon",
            session_id=session_id,
        )
        if working_memory is None:
            _daemon_log(f"no working memory extracted for {session_id}")
            return None
        updated_at = session.get("updated_at") or datetime.now(timezone.utc).isoformat()
        save_extracted_working_memory(
            conn,
            working_memory,
            session_id=session_id,
            updated_at=updated_at,
            source="daemon",
            embed_fn=embed,
        )
        activity_log("daemon", "working_memory", session=session_id, goal=working_memory.current_goal)
        _log_working_memory_details(session_id, working_memory)
        return working_memory
    except Exception as exc:
        error_log("daemon", f"working-memory creation failed for {session_id}: {exc}", exc=exc)
        return None



def _log_session_memory_details(session_id: str, session_memory) -> None:
    _daemon_log(
        f"session memory extracted for {session_id}: title={session_memory.title} | left_off_at={session_memory.left_off_at}"
    )
    if getattr(session_memory, "outcomes", None):
        _daemon_log(f"session memory outcomes for {session_id}: {'; '.join(session_memory.outcomes)}")
    if getattr(session_memory, "next_steps", None):
        _daemon_log(f"session memory next steps for {session_id}: {'; '.join(session_memory.next_steps)}")



def _create_session_memory_entry(conn, session: dict, text_sample: str | None = None):
    """Generate and persist one compacted session memory for a session."""
    session_id = session["session_id"]
    if text_sample is None:
        text_sample = _session_text_sample(session)
    if not text_sample:
        return None

    try:
        session_memory = extract_session_memory_from_session_text(
            text_sample,
            source="daemon",
            session_id=session_id,
        )
        if session_memory is None:
            _daemon_log(f"no session memory extracted for {session_id}")
            return None
        updated_at = session.get("updated_at") or datetime.now(timezone.utc).isoformat()
        save_extracted_session_memory(
            conn,
            session_memory,
            session_id=session_id,
            updated_at=updated_at,
            source="daemon",
            embed_fn=embed,
        )
        activity_log("daemon", "session_memory", session=session_id, title=session_memory.title)
        _log_session_memory_details(session_id, session_memory)
        return session_memory
    except Exception as exc:
        error_log("daemon", f"session-memory creation failed for {session_id}: {exc}", exc=exc)
        return None



def _run_extractor_in_thread(extractor_fn, session: dict, text_sample: str) -> None:
    """Open a fresh DB connection and run one extractor function.

    Each extractor runs in its own thread, so it must open its own SQLite
    connection. SQLite connections are NOT safe to share across threads — each
    thread needs its own handle to avoid data corruption or "database is locked"
    errors. SQLite's WAL (write-ahead log) mode lets multiple connections write
    concurrently without serializing on the GIL.

    Args:
        extractor_fn  — one of the five _extract_* / _create_*_entry functions
        session       — the raw session dict from the DB
        text_sample   — the pre-built conversation text (computed once, read-only)
    """
    conn = open_db(DB_PATH)
    try:
        extractor_fn(conn, session, text_sample=text_sample)
    finally:
        conn.close()


def _process_session(conn, session: dict) -> bool:
    """Process one session through the structured memory pipeline in parallel.

    Previously the five extractors ran sequentially: facts → episodic →
    procedural → working memory → session memory. Since each extractor spends
    most of its time waiting on an Ollama HTTP response (15–60 s per call),
    Python's GIL releases during that I/O wait and all five can run truly
    concurrently with ThreadPoolExecutor.

    Total time per session drops from ~5× per-extractor latency to ~1× (the
    slowest single extractor). On a fast machine this is a 4–5× speedup.

    The caller's `conn` is kept for mark_session_processed only — it runs in
    the main thread after all futures complete, so there is no cross-thread
    sharing of that connection.

    Args:
        conn    — main-thread DB connection (used only for mark_session_processed)
        session — the raw session dict from the DB
    """
    session_id = session["session_id"]
    _daemon_log(f"processing session {session_id} (parallel extractors)")

    # Build or retrieve the compacted text once in the main thread.
    # All five parallel extractors share this single pre-processed input.
    text_sample = _get_or_compact_session_text(conn, session)

    # Each tuple is (extractor_function, human_readable_label_for_logs).
    # The label is used only for error logging if a future raises unexpectedly.
    extractors = [
        (_extract_facts,                "facts"),
        (_create_episodic_entry,        "episodic"),
        (_create_procedural_entry,      "procedural"),
        (_create_working_memory_entry,  "working_memory"),
        (_create_session_memory_entry,  "session_memory"),
    ]

    # max_workers=5: one thread per extractor — they all block on Ollama I/O,
    # so there is no CPU contention and no benefit to fewer workers.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        # Submit all five extractors at once. Each opens its own DB connection
        # via _run_extractor_in_thread.
        future_to_label = {
            pool.submit(_run_extractor_in_thread, fn, session, text_sample): label
            for fn, label in extractors
        }
        # Wait for all futures to finish (as_completed yields in completion order).
        # Collect unexpected exceptions (normal extractor errors are caught inside
        # each extractor function and never reach this level).
        unexpected_errors: list[tuple[str, BaseException]] = []
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            exc = future.exception()
            if exc:
                # This fires only if _run_extractor_in_thread itself raised —
                # e.g. the extractor was patched to raise in a test, or the DB
                # connection could not be opened.
                _daemon_log(f"unexpected thread-level error in {label} for {session_id}: {exc}")
                unexpected_errors.append((label, exc))

    if unexpected_errors:
        # Re-raise so the caller (_run_unprocessed_batch) catches it and skips
        # mark_session_processed. This preserves the original behavior: a session
        # with a failed extractor stays in the unprocessed queue and will be
        # retried on the next daemon cycle.
        label, exc = unexpected_errors[0]
        raise RuntimeError(f"extractor '{label}' failed for {session_id}: {exc}") from exc

    # Mark the session processed in the main thread using the caller's connection,
    # only after all extractors have finished successfully. This prevents partial
    # processing from being silently skipped on the next daemon cycle.
    mark_session_processed(conn, session_id)
    activity_log("daemon", "processed", session=session_id)
    return True


def _prune_stale_memories(conn) -> None:
    """Remove old facts and episodic memories to keep the DB lean."""
    try:
        deleted_facts = prune_stale_facts(conn, days=_FACT_TTL_DAYS)
        deleted_episodes = prune_stale_episodic(conn, days=_EPISODIC_TTL_DAYS)
        if deleted_facts or deleted_episodes:
            _daemon_log(f"pruned {deleted_facts} stale facts, {deleted_episodes} stale episodes")
            activity_log("daemon", "prune", deleted_facts=deleted_facts, deleted_episodes=deleted_episodes)
    except Exception as exc:
        error_log("daemon", f"pruning failed (non-fatal): {exc}", exc=exc)


def _warm_up_embedding() -> None:
    """Load the sentence-transformer model before the first real embed call."""
    try:
        embed("warmup")
        _daemon_log("embedding model warmed up")
    except Exception as exc:
        _daemon_log(f"embedding warm-up failed (non-fatal): {exc}")


def _run_unprocessed_batch(conn) -> int:
    _prune_stale_memories(conn)
    sessions = get_unprocessed_sessions(conn, limit=10)
    if not sessions:
        _daemon_log("no new sessions; backing off")
        return 0

    ollama_proc = _start_ollama_if_needed(log_fn=_daemon_log)
    try:
        for session in sessions:
            if _shutdown:
                break

            session_id = session["session_id"]
            try:
                _process_session(conn, session)
            except Exception as exc:
                error_log(
                    "daemon",
                    f"processing failed for {session_id}: {exc}",
                    exc=exc,
                )
                _daemon_log(f"error processing session {session_id}: {exc}")
    finally:
        if ollama_proc is not None:
            _stop_ollama(ollama_proc, log_fn=_daemon_log)

    return len(sessions)


def process_one_unprocessed_session() -> dict:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = bootstrap_db(DB_PATH) if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0 else open_db(DB_PATH)
    try:
        sessions = get_unprocessed_sessions(conn, limit=1)
        if not sessions:
            _daemon_log("manual process-one requested but no unprocessed sessions found")
            return {"ok": True, "processed": False, "reason": "no_unprocessed_sessions"}

        session = sessions[0]
        session_id = session["session_id"]
        ollama_proc = _start_ollama_if_needed(log_fn=_daemon_log)
        try:
            _process_session(conn, session)
        except Exception as exc:
            error_log("daemon", f"manual processing failed for {session_id}: {exc}", exc=exc)
            _daemon_log(f"manual process-one failed for {session_id}: {exc}")
            return {"ok": False, "processed": False, "session_id": session_id, "error": str(exc)}
        finally:
            if ollama_proc is not None:
                _stop_ollama(ollama_proc, log_fn=_daemon_log)

        return {"ok": True, "processed": True, "session_id": session_id}
    finally:
        conn.close()


def run(once: bool = False) -> None:
    """Run the daemon main loop.

    Default mode respects the CPU gate before each polling cycle.
    ``once=True`` acts as a force flag and immediately runs one extraction pass.
    """
    global _shutdown
    _shutdown = False

    _daemon_log("daemon started")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    bootstrap_conn = bootstrap_db(DB_PATH)
    bootstrap_conn.close()
    _warm_up_embedding()

    while not _shutdown:
        if not once:
            cpu = psutil.cpu_percent(interval=1)
            if cpu > CPU_THRESHOLD:
                activity_log("daemon", "skip_cycle", reason="high_cpu", cpu=cpu)
                _daemon_log(f"skipping cycle: CPU at {cpu:.1f}%")
                time.sleep(60)
                continue

        conn = open_db(DB_PATH)
        try:
            processed = _run_unprocessed_batch(conn)
            if once:
                break
            time.sleep(POLL_INTERVAL if processed else LONG_POLL_INTERVAL)
        finally:
            conn.close()

    _daemon_log("daemon stopped")


if __name__ == "__main__":
    import argparse
    import setproctitle

    setproctitle.setproctitle("AgenticMemoryDaemon")
    enable_debug("daemon")

    parser = argparse.ArgumentParser(
        description="Background memory daemon for fact and episodic extraction."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Force one extraction pass immediately, then exit.",
    )
    args = parser.parse_args()

    run(once=args.once)
