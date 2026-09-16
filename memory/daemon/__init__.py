"""Background daemon for structured memory extraction.

Runs as a long-lived process. Every poll cycle:
  1. finds unprocessed sessions
  2. compact each session via Ollama (cached in the DB)
  3. runs five extractors in parallel: facts, episodic, procedural,
     working-memory, session-memory
  4. marks each session processed

Run:
  python3 -m memory.daemon        # runs forever
  python3 -m memory.daemon --once # one extraction pass, then exit

Stop: SIGTERM — exits cleanly after the current session.
"""

from __future__ import annotations

import concurrent.futures
import os
import signal
import time
from datetime import datetime, timezone

import psutil

# --- Core constants and logger (re-exported for backward compat and patchability) ---
from memory.daemon._core import (  # noqa: F401
    CPU_THRESHOLD,
    DB_PATH,
    LONG_POLL_INTERVAL,
    POLL_INTERVAL,
    _COMPACT_INPUT_CHARS,
    _COMPACT_OUTPUT_CHARS,
    _DAEMON_LOG_PATH,
    _EPISODIC_TTL_DAYS,
    _FACT_TTL_DAYS,
    _daemon_log,
    _should_write_log_file,
)

# --- Compaction helpers (re-exported for patchability) ---
from memory.daemon.compaction import (  # noqa: F401
    _build_session_text,
    _compact_session_text,
    _get_or_compact_session_text,
    _session_text_sample,
)

# --- Pruning helpers (re-exported for patchability) ---
from memory.daemon.pruning import _prune_stale_memories, _warm_up_embedding  # noqa: F401

# --- DB layer ---
from memory.db import (
    bootstrap_db,
    get_unprocessed_sessions,
    mark_session_processed,
    open_db,
)

# --- Memory-type extractors and repositories ---
# These names are imported into the memory.daemon namespace so that
# patch.object(daemon_module, "extract_episode_from_session_text", ...) works.
from memory.episodic import extract_episode_from_session_text
from memory.episodic.repository import save_extracted_episode
from memory.facts.repository import build_fact_content, save_extracted_facts
from memory.facts import extract_facts_from_session_text, normalize_extracted_facts
from memory.utils.logger import activity_log, error_log
from memory.llm.ollama import (
    start_ollama_if_needed as _start_ollama_if_needed,
    stop_ollama as _stop_ollama,
)
from memory.procedural import extract_procedure_from_session_text
from memory.procedural.repository import save_extracted_procedure
from memory.session import extract_session_memory_from_session_text
from memory.session.repository import save_extracted_session_memory
from memory.vectors import embed
from memory.working_memory import extract_working_memory_from_session_text
from memory.working_memory.repository import save_extracted_working_memory

# --- Shutdown flag and signal handling ---

_shutdown = False


def _handle_sigterm(signum, frame):
    del signum, frame
    global _shutdown
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


# ---------------------------------------------------------------------------
# Extractor functions
#
# These are defined in this module (not in a submodule) so that tests using
# patch.object(daemon_module, "extract_episode_from_session_text", ...) and
# patch.object(daemon_module, "_daemon_log", ...) intercept the correct
# name-lookups that happen inside each function's body.
# ---------------------------------------------------------------------------


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
        f"working memory extracted for {session_id}: "
        f"goal={working_memory.current_goal} | next_step={working_memory.next_step} | "
        f"status={working_memory.status}"
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
        f"session memory extracted for {session_id}: "
        f"title={session_memory.title} | left_off_at={session_memory.left_off_at}"
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


# ---------------------------------------------------------------------------
# Parallel extraction pipeline
# ---------------------------------------------------------------------------


def _run_extractor_in_thread(extractor_fn, session: dict, text_sample: str) -> None:
    """Open a fresh DB connection and run one extractor function.

    Each extractor runs in its own thread, so it must open its own SQLite
    connection — connections are NOT safe to share across threads.  SQLite's WAL
    mode lets multiple connections write concurrently.
    """
    conn = open_db(DB_PATH)
    try:
        extractor_fn(conn, session, text_sample=text_sample)
    finally:
        conn.close()


def _process_session(conn, session: dict) -> bool:
    """Process one session through the structured memory pipeline in parallel.

    All five extractors run concurrently in a ThreadPoolExecutor. Each spends
    most of its time waiting on an Ollama HTTP response, so Python's GIL
    releases during I/O and true concurrency is achieved. Total time drops from
    ~5× per-extractor latency to ~1× (slowest single extractor).

    The caller's `conn` is used only for mark_session_processed after all
    futures complete — never shared across threads.
    """
    session_id = session["session_id"]
    _daemon_log(f"processing session {session_id} (parallel extractors)")

    # Compact once in the main thread; all five extractors share this text.
    text_sample = _get_or_compact_session_text(conn, session)

    extractors = [
        (_extract_facts,                "facts"),
        (_create_episodic_entry,        "episodic"),
        (_create_procedural_entry,      "procedural"),
        (_create_working_memory_entry,  "working_memory"),
        (_create_session_memory_entry,  "session_memory"),
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        future_to_label = {
            pool.submit(_run_extractor_in_thread, fn, session, text_sample): label
            for fn, label in extractors
        }
        unexpected_errors: list[tuple[str, BaseException]] = []
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            exc = future.exception()
            if exc:
                _daemon_log(f"unexpected thread-level error in {label} for {session_id}: {exc}")
                unexpected_errors.append((label, exc))

    if unexpected_errors:
        label, exc = unexpected_errors[0]
        raise RuntimeError(f"extractor '{label}' failed for {session_id}: {exc}") from exc

    mark_session_processed(conn, session_id)
    activity_log("daemon", "processed", session=session_id)
    return True


# ---------------------------------------------------------------------------
# Daemon main loop
# ---------------------------------------------------------------------------


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
    """Process a single unprocessed session.  Used by the dashboard / API."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = (
        bootstrap_db(DB_PATH)
        if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0
        else open_db(DB_PATH)
    )
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
