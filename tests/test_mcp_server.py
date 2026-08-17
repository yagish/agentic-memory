# test_mcp_server.py — tests for the MCP server tool handlers.
#
# Run with:  python3 -m unittest tests.test_mcp_server -v
#
# We test the three tool functions directly (memory_status, memory_search,
# memory_get_session) by temporarily pointing them at a temporary database,
# without going through the MCP wire protocol at all.

import json
import os
import sys
import tempfile
import unittest

# Add the project root so we can import both memory.db and memory.mcp_server.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import init_db, upsert_session


# Sample sessions we'll seed the test DB with.
SESSION_A_ID = "session-aaa"
SESSION_A_TRANSCRIPT = [
    {"role": "user",      "content": "Explain quantum entanglement please."},
    {"role": "assistant", "content": "Quantum entanglement links two particles so measuring one instantly affects the other."},
]

SESSION_B_ID = "session-bbb"
SESSION_B_TRANSCRIPT = [
    {"role": "user",      "content": "What is the speed of light?"},
    {"role": "assistant", "content": "The speed of light in a vacuum is 299,792,458 metres per second."},
    {"role": "user",      "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome."},
]


class TestMcpServerTools(unittest.TestCase):

    def setUp(self):
        """
        Before each test:
        1. Create a temporary database file on disk (not in-memory, because
           the server module opens its own connection via DB_PATH).
        2. Seed it with two sessions.
        3. Monkey-patch the DB_PATH inside mcp_server so it uses the temp file.
        """
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()

        # Seed the temp DB.
        conn = init_db(self.tmp_db.name)
        upsert_session(conn, SESSION_A_ID, "claude", SESSION_A_TRANSCRIPT,
                       "2026-08-01T10:00:00Z", "2026-08-01T10:05:00Z")
        upsert_session(conn, SESSION_B_ID, "claude", SESSION_B_TRANSCRIPT,
                       "2026-08-10T09:00:00Z", "2026-08-10T09:10:00Z")
        conn.close()

        # Import (or reload) mcp_server and redirect its DB_PATH to our temp file.
        import memory.mcp_server as srv
        self._original_db_path = srv.DB_PATH
        srv.DB_PATH = self.tmp_db.name
        self.srv = srv

    def tearDown(self):
        # Restore the original DB_PATH and delete the temp file.
        self.srv.DB_PATH = self._original_db_path
        os.unlink(self.tmp_db.name)

    # --- memory_status ---

    def test_status_returns_required_keys(self):
        result = self.srv.memory_status()
        for key in ("total_sessions", "total_turns", "oldest_session", "newest_session"):
            self.assertIn(key, result, f"missing key: {key}")

    def test_status_counts_are_correct(self):
        result = self.srv.memory_status()
        # We seeded 2 sessions with 2 + 4 = 6 turns total.
        self.assertEqual(result["total_sessions"], 2)
        self.assertEqual(result["total_turns"], 6)

    def test_status_counts_are_non_negative(self):
        result = self.srv.memory_status()
        self.assertGreaterEqual(result["total_sessions"], 0)
        self.assertGreaterEqual(result["total_turns"], 0)

    # --- memory_search ---

    def test_search_finds_matching_session(self):
        results = self.srv.memory_search("entanglement")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], SESSION_A_ID)

    def test_search_returns_required_keys(self):
        results = self.srv.memory_search("speed of light")
        self.assertEqual(len(results), 1)
        for key in ("session_id", "agent", "updated_at", "snippet"):
            self.assertIn(key, results[0], f"missing key: {key}")

    def test_search_no_match_returns_empty_list(self):
        results = self.srv.memory_search("photosynthesis")
        self.assertEqual(results, [])

    # --- memory_get_session ---

    def test_get_session_returns_full_transcript(self):
        result = self.srv.memory_get_session(SESSION_A_ID)
        self.assertEqual(result["session_id"], SESSION_A_ID)
        self.assertEqual(result["turn_count"], 2)
        self.assertEqual(result["transcript"], SESSION_A_TRANSCRIPT)

    def test_get_session_unknown_id_returns_error(self):
        result = self.srv.memory_get_session("does-not-exist")
        self.assertIn("error", result)

    def test_get_session_returns_all_required_keys(self):
        result = self.srv.memory_get_session(SESSION_B_ID)
        for key in ("session_id", "agent", "started_at", "updated_at", "turn_count", "transcript"):
            self.assertIn(key, result, f"missing key: {key}")


if __name__ == "__main__":
    unittest.main()
