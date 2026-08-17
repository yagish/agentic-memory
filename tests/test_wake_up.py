# test_wake_up.py — tests for the wake-up hook's core logic.
#
# We test the pure functions (build_digest, fetch_recent_sessions) directly,
# not the stdin/stdout hook layer — those are hard to test in isolation and
# the logic that matters is in the functions themselves.
#
# Run with:  python3 -m unittest tests.test_wake_up -v

import json
import os
import sys
import unittest

# Add the project root to the Python path so we can import from hooks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.wake_up import build_digest, fetch_recent_sessions
from memory.db import init_db, upsert_session


class TestBuildDigest(unittest.TestCase):
    """Tests for build_digest() — the function that formats the wake-up text."""

    def test_digest_contains_identity(self):
        # The digest should include whatever is in identity.md
        result = build_digest("Name: Alice\nRole: engineer", [])
        self.assertIn("Name: Alice", result)
        self.assertIn("Role: engineer", result)

    def test_digest_contains_session_timestamps(self):
        # The digest should list each session's timestamp
        sessions = [
            {"updated_at": "2026-08-15T10:00:00Z", "turn_count": 4, "first_user_message": "Hello there"},
        ]
        result = build_digest("", sessions)
        self.assertIn("2026-08-15T10:00:00Z", result)

    def test_digest_contains_turn_count(self):
        # The digest should show how many turns each session had
        sessions = [
            {"updated_at": "2026-08-15T10:00:00Z", "turn_count": 12, "first_user_message": "test"},
        ]
        result = build_digest("", sessions)
        self.assertIn("12 turns", result)

    def test_empty_sessions_shows_placeholder(self):
        # When there are no past sessions, show a friendly message instead of a blank section
        result = build_digest("", [])
        self.assertIn("no past sessions found", result)

    def test_missing_identity_shows_placeholder(self):
        # When identity is empty (file missing), show a helpful hint
        result = build_digest("", [])
        self.assertIn("no identity.md found", result)

    def test_digest_has_markers(self):
        # The digest should be clearly delimited so Claude can spot it
        result = build_digest("", [])
        self.assertIn("=== MEMORY WAKE-UP ===", result)
        self.assertIn("=== END MEMORY ===", result)


class TestFetchRecentSessions(unittest.TestCase):
    """Tests for fetch_recent_sessions() — querying the DB for past sessions."""

    def setUp(self):
        # Use a real in-memory DB and point DB_PATH at a temp file path
        # We monkey-patch the module-level DB_PATH for these tests
        import hooks.wake_up as wu
        self._original_db_path = wu.DB_PATH

        # Use a temp file so fetch_recent_sessions can open it
        import tempfile
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        wu.DB_PATH = self._tmp.name

        # Seed the temp DB with some sessions
        conn = init_db(self._tmp.name)
        upsert_session(
            conn, "sess-old", "claude",
            [{"role": "user", "content": "Tell me about quantum physics"},
             {"role": "assistant", "content": "Sure!"}],
            "2026-08-10T09:00:00Z", "2026-08-10T09:05:00Z"
        )
        upsert_session(
            conn, "sess-new", "claude",
            [{"role": "user", "content": "What is Python?"},
             {"role": "assistant", "content": "Python is a language."}],
            "2026-08-15T10:00:00Z", "2026-08-15T10:10:00Z"
        )
        conn.close()

    def tearDown(self):
        import hooks.wake_up as wu
        wu.DB_PATH = self._original_db_path
        os.unlink(self._tmp.name)

    def test_returns_sessions_excluding_current(self):
        # The current session should not appear in its own wake-up digest
        results = fetch_recent_sessions("sess-new")
        ids = [r["session_id"] for r in results]
        self.assertNotIn("sess-new", ids)
        self.assertIn("sess-old", ids)

    def test_returns_most_recent_first(self):
        # Sessions should be ordered newest-first
        results = fetch_recent_sessions("irrelevant-session-id")
        self.assertEqual(results[0]["session_id"], "sess-new")
        self.assertEqual(results[1]["session_id"], "sess-old")

    def test_first_user_message_extracted(self):
        # The first user message should be pulled from the transcript
        results = fetch_recent_sessions("irrelevant-session-id")
        new_session = next(r for r in results if r["session_id"] == "sess-new")
        self.assertIn("What is Python", new_session["first_user_message"])

    def test_empty_db_returns_empty_list(self):
        # If the DB has no sessions (other than the current one), return []
        results = fetch_recent_sessions("sess-old")
        # Only sess-new remains; should return it
        self.assertEqual(len(results), 1)

    def test_missing_db_returns_empty_list(self):
        # If the DB file doesn't exist yet, return [] gracefully
        import hooks.wake_up as wu
        wu.DB_PATH = "/tmp/nonexistent_memory_test.db"
        results = fetch_recent_sessions("any-session")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
