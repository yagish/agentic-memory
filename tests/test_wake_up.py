# test_wake_up.py — tests for the wake-up hook's core logic.
#
# We test the pure functions (build_digest, fetch_recent_sessions,
# fetch_relevant_context, get_wake_up_digest) directly, not the
# stdin/stdout hook layer — those are hard to test in isolation and
# the logic that matters is in the functions themselves.
#
# Run with:  python3 -m unittest tests.test_wake_up -v

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add the project root to the Python path so we can import from hooks/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.wake_up import (
    build_digest,
    fetch_recent_sessions,
    fetch_relevant_context,
    get_wake_up_digest,
)
from memory.db import init_db, insert_fact, upsert_session


# ---------------------------------------------------------------------------
# Existing tests — kept exactly as before (Phase 1–9 coverage)
# ---------------------------------------------------------------------------


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
        # Use a real in-memory DB and point DB_PATH at a temp file path.
        # We monkey-patch the module-level DB_PATH for these tests.
        import hooks.wake_up as wu
        self._original_db_path = wu.DB_PATH

        # Use a temp file so fetch_recent_sessions can open it
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


# ---------------------------------------------------------------------------
# New tests — Phase 10: relevance-based wake-up
# ---------------------------------------------------------------------------


class TestRelevanceWakeUp(unittest.TestCase):
    """
    Tests for Phase 10: fetch_relevant_context(), get_wake_up_digest(),
    and the extended build_digest() with facts support.

    We use a temp-file DB (not :memory:) so that fetch_recent_sessions()
    — which opens its own connection via the monkey-patched DB_PATH — can
    read the same sessions we seeded.  fetch_relevant_context receives an
    explicit conn passed from the test, so its DB access is independent.
    """

    def setUp(self):
        # Monkey-patch DB_PATH so fetch_recent_sessions uses our temp file.
        import hooks.wake_up as wu
        self._wu = wu
        self._original_db_path = wu.DB_PATH

        # Create a temp database file for the duration of this test.
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        wu.DB_PATH = self._tmp.name

        # Open a connection to the temp DB and seed it with two sessions:
        # sess-auth — old, about authentication bugs
        # sess-ui   — recent, about UI/CSS styling
        self._conn = init_db(self._tmp.name)
        upsert_session(
            self._conn, "sess-auth", "claude",
            [{"role": "user", "content": "authentication bug with login form"},
             {"role": "assistant", "content": "Let me look at your auth code."}],
            "2026-01-01T09:00:00Z", "2026-01-01T09:05:00Z",
        )
        upsert_session(
            self._conn, "sess-ui", "claude",
            [{"role": "user", "content": "CSS button styling and UI layout"},
             {"role": "assistant", "content": "Sure, let's look at your CSS."}],
            "2026-08-15T10:00:00Z", "2026-08-15T10:10:00Z",
        )

    def tearDown(self):
        # Restore the original DB_PATH and clean up the temp file.
        self._conn.close()
        self._wu.DB_PATH = self._original_db_path
        os.unlink(self._tmp.name)

    # --- test 1 ---

    @patch("hooks.wake_up.semantic_search_chunks")
    @patch("hooks.wake_up.search_facts")
    def test_relevance_beats_recency(self, mock_facts, mock_chunks):
        """
        When semantic search says the authentication session is more relevant
        (lower cosine distance) than the UI session, the authentication session
        should appear first in the digest — even though the UI session is newer.
        """
        # Mock chunk search: authentication session is more relevant (distance 0.1)
        # than the UI session (distance 0.9).
        mock_chunks.return_value = [
            {
                "session_id": "sess-auth",
                "chunk_index": 0,
                "distance":    0.1,        # low distance = highly relevant
                "text":        "authentication bug with login form",
                "snippet":     "login authentication bug",
            },
            {
                "session_id": "sess-ui",
                "chunk_index": 0,
                "distance":    0.9,        # high distance = less relevant
                "text":        "CSS button styling",
                "snippet":     "CSS button styling UI",
            },
        ]
        # Facts search returns nothing for this prompt.
        mock_facts.return_value = []

        digest = get_wake_up_digest(
            self._conn, "current-session", "authentication bug", ""
        )

        # Both snippets must appear in the digest.
        auth_pos = digest.find("login authentication bug")
        ui_pos   = digest.find("CSS button styling UI")
        self.assertGreater(auth_pos, -1, "Authentication snippet should appear in digest")
        self.assertGreater(ui_pos, -1, "UI snippet should appear in digest")

        # Authentication session (relevant) should be listed before UI session (recent).
        self.assertLess(
            auth_pos, ui_pos,
            "Relevant authentication session should appear before more-recent UI session",
        )

    # --- test 2 ---

    def test_fallback_on_empty_prompt(self):
        """
        When the prompt is empty, get_wake_up_digest skips semantic search entirely
        and falls back to the recency-based approach (last N sessions by updated_at).
        The digest should show the most-recent session first and omit the [L2] section.
        """
        # Pass an empty prompt — semantic search should NOT be called.
        digest = get_wake_up_digest(
            self._conn, "current-session", "", ""
        )

        # Recency heading must be present (not the relevance heading).
        self.assertIn("[L1 — Recent Sessions]", digest)

        # No facts section in recency fallback.
        self.assertNotIn("[L2", digest)

        # UI session (2026-08-15) is more recent — it should appear before
        # the authentication session (2026-01-01) in the digest.
        ui_pos   = digest.find("2026-08-15")
        auth_pos = digest.find("2026-01-01")
        self.assertGreater(ui_pos, -1, "Recent UI session timestamp should appear in digest")
        self.assertGreater(auth_pos, -1, "Older auth session timestamp should appear in digest")
        self.assertLess(
            ui_pos, auth_pos,
            "More-recent session should appear first in recency fallback",
        )

    # --- test 3 ---

    @patch("hooks.wake_up.semantic_search_chunks")
    def test_fallback_on_import_error(self, mock_chunks):
        """
        When semantic_search_chunks raises ImportError (e.g. sentence-transformers
        not installed), get_wake_up_digest must fall back gracefully to recency.
        The function should return a valid digest string — not crash.
        """
        # Simulate sentence-transformers being absent.
        mock_chunks.side_effect = ImportError(
            "sentence-transformers is not installed. Run: pip3 install sentence-transformers"
        )

        # This call must not raise — it should catch the ImportError and fall back.
        digest = get_wake_up_digest(
            self._conn, "current-session", "authentication bug", ""
        )

        # Must be a non-empty string with the standard markers.
        self.assertIn("=== MEMORY WAKE-UP ===", digest)
        self.assertIn("=== END MEMORY ===", digest)

        # Fallback digest must not contain an L2 facts section.
        self.assertNotIn("[L2", digest)

    # --- test 4 ---

    @patch("hooks.wake_up.semantic_search_chunks")
    @patch("hooks.wake_up.search_facts")
    def test_facts_section_present(self, mock_facts, mock_chunks):
        """
        When search_facts returns results, build_digest includes a
        [L2 — Relevant Facts] section with bullet points.
        """
        # Return one matching chunk so the relevance path is taken.
        mock_chunks.return_value = [
            {
                "session_id": "sess-auth",
                "chunk_index": 0,
                "distance":    0.2,
                "text":        "authentication bug with login form",
                "snippet":     "login bug",
            },
        ]
        # Return one fact with tags.
        mock_facts.return_value = [
            {
                "id":         "fact-1",
                "content":    "User prefers OAuth login over password",
                "tags":       ["auth", "preferences"],
                "source":     "agent",
                "session_id": "sess-auth",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "snippet":    "User prefers [OAuth] login",
            },
        ]

        digest = get_wake_up_digest(
            self._conn, "current-session", "authentication login", ""
        )

        # Facts section must be present.
        self.assertIn("[L2 — Relevant Facts]", digest)

        # The fact content must appear in the digest.
        self.assertIn("User prefers OAuth login over password", digest)

        # Tags must appear in the expected format.
        self.assertIn("[tags: auth, preferences]", digest)

    # --- test 5 ---

    def test_facts_section_absent_on_fallback(self):
        """
        The [L2 — Relevant Facts] section must NOT appear when the recency
        fallback is used (e.g. when the prompt is empty).
        """
        # Empty prompt triggers recency fallback unconditionally.
        digest = get_wake_up_digest(
            self._conn, "current-session", "", ""
        )

        # No L2 section should appear anywhere in the digest.
        self.assertNotIn("[L2", digest)
        self.assertNotIn("Relevant Facts", digest)


# ---------------------------------------------------------------------------
# Phase 15: injection-size logging test
# ---------------------------------------------------------------------------

import io
from unittest.mock import MagicMock, patch

import hooks.wake_up as _wu_module


class TestInjectionSizeLogged(unittest.TestCase):
    """Test that main() logs the injection size via log_retrieval."""

    def test_injection_size_logged(self):
        """
        When main() runs for a new session, it must call log_retrieval with
        tool='wake_up_injection' and result_size = len(digest) // 4.
        We mock get_wake_up_digest so the digest is a fixed string and we
        mock init_db so a fake connection is provided (conn is not None).
        """
        # A digest of exactly 400 characters → 400 // 4 = 100 est. tokens.
        fake_digest = "x" * 400

        # Use a session ID that is unlikely to collide with a real flag file.
        session_id = "test-injection-logging-phase15"
        flag_path  = f"/tmp/memory_injected_{session_id}"

        # Remove a stale flag file from a previous test run if one exists.
        if os.path.exists(flag_path):
            os.remove(flag_path)

        try:
            payload = json.dumps({"session_id": session_id, "prompt": "hello"})

            # os.path.exists is called twice in main():
            #   1. flag_path check  → must return False so we don't skip
            #   2. DB_PATH check    → must return True so conn is attempted
            # We fake DB_PATH to a sentinel value and match only that.
            fake_db_path = "/fake/nonexistent_for_test.db"

            def fake_exists(path):
                """Return True only for our fake DB path; False for everything else."""
                return path == fake_db_path

            with patch("sys.stdin",  io.StringIO(payload)), \
                 patch("sys.exit"), \
                 patch.object(_wu_module, "DB_PATH",    fake_db_path), \
                 patch("os.path.exists",                side_effect=fake_exists), \
                 patch.object(_wu_module, "init_db",    return_value=MagicMock()) as mock_init_db, \
                 patch.object(_wu_module, "get_wake_up_digest", return_value=fake_digest), \
                 patch.object(_wu_module, "log_retrieval") as mock_log_retrieval, \
                 patch.object(_wu_module, "activity_log"):

                _wu_module.main()

            # log_retrieval must have been called exactly once.
            mock_log_retrieval.assert_called_once()

            # Verify the positional arguments:
            #   arg[0] = conn (MagicMock)
            #   arg[1] = tool = 'wake_up_injection'
            #   arg[2] = query = None
            #   arg[3] = result_size = 100  (400 chars // 4)
            call_args = mock_log_retrieval.call_args[0]
            self.assertEqual(call_args[1], "wake_up_injection")
            self.assertIsNone(call_args[2])
            self.assertEqual(call_args[3], 100)

        finally:
            # Clean up the flag file written by main() during the test.
            if os.path.exists(flag_path):
                os.remove(flag_path)


if __name__ == "__main__":
    unittest.main()
