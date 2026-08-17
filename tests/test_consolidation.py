# test_consolidation.py — tests for Phase 12 memory consolidation and decay.
#
# Run with: python3 -m pytest tests/test_consolidation.py -v
#
# Each test creates a fresh in-memory SQLite database so there are no side
# effects between tests. Ollama is never actually called — we use unittest.mock
# to replace _call_ollama with a controlled return value.

import json
import unittest
import unittest.mock as mock
from datetime import datetime, timezone, timedelta

from memory.db import (
    get_summary,
    init_db,
    insert_summary,
    prune_transcript,
    sessions_needing_prune,
    sessions_needing_summary,
    upsert_session,
)
from memory.consolidation import consolidate_old_sessions, prune_old_transcripts


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A short transcript used across multiple tests.
# Two turns so the session has real content that passes the SQL filter.
_TRANSCRIPT_TWO_TURNS = [
    {"role": "user",      "content": "Tell me about Python generators."},
    {"role": "assistant", "content": "Generators use yield to produce values lazily."},
]

# An ISO timestamp far in the past — always older than any reasonable threshold.
_VERY_OLD_DATE = "2020-01-01T00:00:00+00:00"


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    # Used for 'updated_at' values that should look "too new" to be included.
    return datetime.now(timezone.utc).isoformat()


def _days_ago_iso(n: int) -> str:
    """Return an ISO 8601 timestamp for exactly n days ago."""
    # timedelta subtracts n days from the current time.
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _insert_session(conn, session_id: str, updated_at: str, transcript=None):
    """
    Insert a test session into the database.

    Defaults to a two-turn transcript so the session is eligible for
    sessions_needing_summary (which requires transcript != '[]').
    """
    # Use the provided transcript or fall back to the default two-turn example.
    turns = transcript if transcript is not None else _TRANSCRIPT_TWO_TURNS
    upsert_session(conn, session_id, "claude", turns, updated_at, updated_at)


# ---------------------------------------------------------------------------
# Test group: sessions_needing_summary (db helper)
# ---------------------------------------------------------------------------

class TestSessionsNeedingSummary(unittest.TestCase):

    def setUp(self):
        # Create a fresh in-memory database before each test.
        self.conn = init_db(":memory:")

    def test_sessions_needing_summary_respects_threshold(self):
        # A session older than the threshold should be returned.
        # A session newer than the threshold should be excluded.
        #
        # We insert an "old" session (2020 = definitely >30 days ago)
        # and a "new" session (right now = definitely <30 days ago).
        _insert_session(self.conn, "old-sess", _VERY_OLD_DATE)
        _insert_session(self.conn, "new-sess", _now_iso())

        # With threshold=30 days, only old-sess should appear.
        results = sessions_needing_summary(self.conn, days_threshold=30)
        session_ids = [r["session_id"] for r in results]

        # The old session must be included.
        self.assertIn("old-sess", session_ids)
        # The new session must be excluded (it was updated too recently).
        self.assertNotIn("new-sess", session_ids)

    def test_sessions_needing_summary_excludes_already_summarised(self):
        # A session that already has a row in the summaries table must not be
        # returned — we don't want to re-summarise sessions we already processed.

        # Insert an old session and then give it a summary.
        _insert_session(self.conn, "already-done", _VERY_OLD_DATE)
        insert_summary(self.conn, "already-done", "Existing summary.", "llama3.2:3b")

        # Insert another old session with no summary — this one should be returned.
        _insert_session(self.conn, "needs-summary", _VERY_OLD_DATE)

        results = sessions_needing_summary(self.conn, days_threshold=1)
        session_ids = [r["session_id"] for r in results]

        # Only the session without a summary should appear.
        self.assertNotIn("already-done", session_ids)
        self.assertIn("needs-summary", session_ids)


# ---------------------------------------------------------------------------
# Test group: insert_summary and get_summary (db helpers)
# ---------------------------------------------------------------------------

class TestInsertAndGetSummary(unittest.TestCase):

    def setUp(self):
        # Fresh database with one session we can attach a summary to.
        self.conn = init_db(":memory:")
        _insert_session(self.conn, "sess-a", _VERY_OLD_DATE)

    def test_insert_and_get_summary(self):
        # After inserting a summary, get_summary must return the exact same text.

        # Insert a summary for sess-a.
        insert_summary(self.conn, "sess-a", "This was a Python session.", "llama3.2:3b")

        # Retrieve it and check the round-trip is exact.
        retrieved = get_summary(self.conn, "sess-a")

        # The text must come back unchanged.
        self.assertEqual(retrieved, "This was a Python session.")

    def test_get_summary_returns_none_for_missing_session(self):
        # get_summary should return None when no summary exists yet.
        # This is the initial state for every new session.
        result = get_summary(self.conn, "no-summary-yet")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test group: prune_transcript (db helper)
# ---------------------------------------------------------------------------

class TestPruneTranscript(unittest.TestCase):

    def setUp(self):
        # Fresh database with a session that has a non-empty transcript.
        self.conn = init_db(":memory:")
        _insert_session(self.conn, "prune-sess", _VERY_OLD_DATE)

    def test_prune_transcript_nulls_out_data(self):
        # After calling prune_transcript, the transcript column must be '[]'.

        prune_transcript(self.conn, "prune-sess")

        # Read the transcript back directly from the database.
        row = self.conn.execute(
            "SELECT transcript FROM sessions WHERE session_id = ?", ("prune-sess",)
        ).fetchone()

        # The transcript must be '[]' — not None and not the original content.
        self.assertIsNotNone(row)
        self.assertEqual(row["transcript"], "[]")


# ---------------------------------------------------------------------------
# Test group: consolidate_old_sessions (main consolidation function)
# ---------------------------------------------------------------------------

class TestConsolidateOldSessions(unittest.TestCase):

    def setUp(self):
        # Fresh database before each test.
        self.conn = init_db(":memory:")

    def test_consolidate_calls_ollama(self):
        # consolidate_old_sessions should call _call_ollama with a prompt that
        # contains text from the session's transcript, then store the result.

        # Insert an old session (1-day threshold, session is from 2020 — definitely old).
        _insert_session(self.conn, "old-sess", _VERY_OLD_DATE)

        # Patch _call_ollama so we never actually contact ollama in tests.
        # The patch target must be the name as it appears in the consolidation module.
        with mock.patch(
            "memory.consolidation._call_ollama",
            return_value="This session covered Python generators.",
        ) as mock_ollama:
            consolidate_old_sessions(self.conn, days_threshold=1, dry_run=False)

        # _call_ollama must have been called exactly once (one session to summarise).
        mock_ollama.assert_called_once()

        # The prompt argument must contain text from the transcript.
        # _call_ollama receives one positional argument: the prompt string.
        prompt_arg = mock_ollama.call_args[0][0]
        self.assertIn("Python generators", prompt_arg)

        # The returned summary must be stored in the summaries table.
        stored = get_summary(self.conn, "old-sess")
        self.assertIsNotNone(stored)
        self.assertEqual(stored, "This session covered Python generators.")

    def test_consolidate_dry_run_does_not_write(self):
        # With dry_run=True, consolidate_old_sessions must not write any rows
        # to the summaries table — it only prints what would happen.

        # Insert an old session.
        _insert_session(self.conn, "dry-sess", _VERY_OLD_DATE)

        # Patch _call_ollama so it would return something if called.
        with mock.patch(
            "memory.consolidation._call_ollama",
            return_value="Should not be stored.",
        ):
            # dry_run=True — nothing should be written.
            consolidate_old_sessions(self.conn, days_threshold=1, dry_run=True)

        # get_summary must return None because no row was inserted.
        stored = get_summary(self.conn, "dry-sess")
        self.assertIsNone(stored)

    def test_consolidate_skips_session_when_ollama_fails(self):
        # If ollama raises RuntimeError, the session must be skipped without
        # crashing the entire consolidation run.

        # Insert two old sessions.
        _insert_session(self.conn, "fail-sess", _VERY_OLD_DATE)
        _insert_session(self.conn, "ok-sess",   _VERY_OLD_DATE)

        call_count = 0

        def side_effect(prompt):
            # Raise on the first call (fail-sess), succeed on the second (ok-sess).
            # We track call count via a closure variable (nonlocal keyword).
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("ollama timed out")
            return "OK summary."

        with mock.patch("memory.consolidation._call_ollama", side_effect=side_effect):
            result = consolidate_old_sessions(self.conn, days_threshold=1, dry_run=False)

        # One summarised, one skipped due to error.
        self.assertEqual(result["summarised"], 1)
        self.assertEqual(result["skipped"],    1)


# ---------------------------------------------------------------------------
# Test group: prune_old_transcripts (main prune function)
# ---------------------------------------------------------------------------

class TestPruneOldTranscripts(unittest.TestCase):

    def setUp(self):
        # Fresh database before each test.
        self.conn = init_db(":memory:")

    def test_prune_old_transcripts_clears_transcript(self):
        # After inserting a summary for an old session, prune_old_transcripts
        # must set the transcript to '[]'.

        _insert_session(self.conn, "prune-me", _VERY_OLD_DATE)
        insert_summary(self.conn, "prune-me", "A summary exists.", "llama3.2:3b")

        prune_old_transcripts(self.conn, days_threshold=1, dry_run=False)

        # The transcript column must now be '[]'.
        row = self.conn.execute(
            "SELECT transcript FROM sessions WHERE session_id = ?", ("prune-me",)
        ).fetchone()
        self.assertEqual(row["transcript"], "[]")

    def test_prune_dry_run_does_not_write(self):
        # With dry_run=True, no transcripts must be modified.

        _insert_session(self.conn, "keep-me", _VERY_OLD_DATE)
        insert_summary(self.conn, "keep-me", "Summary.", "llama3.2:3b")

        prune_old_transcripts(self.conn, days_threshold=1, dry_run=True)

        # The transcript must still contain the original content.
        row = self.conn.execute(
            "SELECT transcript FROM sessions WHERE session_id = ?", ("keep-me",)
        ).fetchone()
        # '[]' means it was pruned — it must NOT be '[]'.
        self.assertNotEqual(row["transcript"], "[]")
        # The stored JSON must still be parseable and non-empty.
        turns = json.loads(row["transcript"])
        self.assertGreater(len(turns), 0)


if __name__ == "__main__":
    unittest.main()
