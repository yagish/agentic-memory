# daemon.py — background daemon for structured memory extraction.
#
# Runs as a long-lived process. Every poll cycle:
#   1. finds unprocessed sessions
#   2. extracts durable facts from the session
#   3. extracts one episodic memory from the session
#   4. extracts one procedural memory from the session
#   5. marks the session processed
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
)
from memory.debug import enable_debug
from memory.episodic import extract_episode_from_session_text
from memory.episodic_repository import save_extracted_episode
from memory.fact_repository import build_fact_content, save_extracted_facts
from memory.facts import extract_facts_from_session_text, normalize_extracted_facts
from memory.logger import activity_log, error_log
from memory.procedural import extract_procedure_from_session_text
from memory.procedural_repository import save_extracted_procedure
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


def _session_text_sample(session: dict, max_chars: int | None = None) -> str:
    try:
        turns = json.loads(session.get("transcript") or "[]")
    except json.JSONDecodeError:
        turns = []
    return _build_session_text(turns, max_chars=max_chars)


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


def _process_session(conn, session: dict) -> bool:
    """Process one session through the facts/episodic/procedural pipeline."""
    session_id = session["session_id"]
    _daemon_log(f"processing session {session_id}")

    text_sample = _session_text_sample(session)
    _extract_facts(conn, session, text_sample=text_sample)
    _create_episodic_entry(conn, session, text_sample=text_sample)
    _create_procedural_entry(conn, session, text_sample=text_sample)

    mark_session_processed(conn, session_id)
    activity_log("daemon", "processed", session=session_id)
    return True


def _run_unprocessed_batch(conn) -> int:
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
