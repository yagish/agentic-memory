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


class TestFactTools(unittest.TestCase):
    """
    Tests for the four fact MCP tools added in Phase 8:
    memory_save_fact, memory_update_fact, memory_delete_fact, memory_list_facts.

    Each test creates a fresh temporary database so tests are fully isolated.
    """

    def setUp(self):
        """
        Create a temporary database file and redirect mcp_server to use it.
        We don't seed any sessions here — each test creates facts as needed.
        """
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()

        # Import (or reload) the server module and redirect its DB_PATH.
        import memory.mcp_server as srv
        self._original_db_path = srv.DB_PATH
        srv.DB_PATH = self.tmp_db.name
        self.srv = srv

    def tearDown(self):
        # Restore the real DB_PATH and clean up the temporary file.
        self.srv.DB_PATH = self._original_db_path
        os.unlink(self.tmp_db.name)

    # --- memory_save_fact ---

    def test_save_fact_returns_id_content_tags(self):
        # memory_save_fact must return a dict with all three expected keys.
        result = self.srv.memory_save_fact("Python is interpreted.", tags=["python"])
        self.assertIn("id", result)
        self.assertIn("content", result)
        self.assertIn("tags", result)

    def test_save_fact_id_is_non_empty_string(self):
        # The returned id must be a non-empty string (a UUID).
        result = self.srv.memory_save_fact("Go is compiled.")
        self.assertIsInstance(result["id"], str)
        self.assertGreater(len(result["id"]), 0)

    def test_save_fact_tags_default_to_empty_list(self):
        # When no tags are passed, the returned tags must be an empty list — not None.
        result = self.srv.memory_save_fact("Rust has no GC.")
        self.assertEqual(result["tags"], [])

    # --- memory_update_fact ---

    def test_update_fact_valid_id_returns_updated_true(self):
        # Updating an existing fact must return {"updated": True, "fact_id": <id>}.
        saved = self.srv.memory_save_fact("original content")
        result = self.srv.memory_update_fact(saved["id"], content="new content")
        self.assertTrue(result["updated"])
        self.assertEqual(result["fact_id"], saved["id"])

    def test_update_fact_unknown_id_returns_updated_false(self):
        # Updating a non-existent fact must return {"updated": False, ...}.
        result = self.srv.memory_update_fact("nonexistent-uuid", content="anything")
        self.assertFalse(result["updated"])
        self.assertEqual(result["fact_id"], "nonexistent-uuid")

    # --- memory_delete_fact ---

    def test_delete_fact_returns_deleted_true(self):
        # Deleting an existing fact must return {"deleted": True, "fact_id": <id>}.
        saved = self.srv.memory_save_fact("a fact to delete")
        result = self.srv.memory_delete_fact(saved["id"])
        self.assertTrue(result["deleted"])
        self.assertEqual(result["fact_id"], saved["id"])

    def test_delete_fact_actually_removes_from_list(self):
        # After deletion, the fact must not appear in memory_list_facts.
        saved = self.srv.memory_save_fact("fact that will be deleted")
        self.srv.memory_delete_fact(saved["id"])
        remaining = self.srv.memory_list_facts()
        ids = [f["id"] for f in remaining]
        self.assertNotIn(saved["id"], ids)

    def test_delete_fact_unknown_id_returns_deleted_false(self):
        # Deleting a non-existent fact must return {"deleted": False, ...}.
        result = self.srv.memory_delete_fact("does-not-exist")
        self.assertFalse(result["deleted"])

    # --- memory_list_facts ---

    def test_list_facts_no_tag_returns_all(self):
        # memory_list_facts with no tag filter must return all saved facts.
        self.srv.memory_save_fact("fact one", tags=["a"])
        self.srv.memory_save_fact("fact two", tags=["b"])
        results = self.srv.memory_list_facts()
        self.assertEqual(len(results), 2)

    def test_list_facts_with_tag_filters_correctly(self):
        # memory_list_facts with a tag must only return facts that carry that tag.
        self.srv.memory_save_fact("tagged with python", tags=["python"])
        self.srv.memory_save_fact("tagged with go", tags=["go"])
        results = self.srv.memory_list_facts(tag="python")
        # Only one fact has the "python" tag.
        self.assertEqual(len(results), 1)
        self.assertIn("python", results[0]["tags"])

    def test_list_facts_result_has_required_keys(self):
        # Every fact in the list must contain all keys callers depend on.
        self.srv.memory_save_fact("a well-structured fact", tags=["test"])
        results = self.srv.memory_list_facts()
        self.assertEqual(len(results), 1)
        for key in ("id", "content", "tags", "source", "session_id",
                    "created_at", "updated_at"):
            self.assertIn(key, results[0], f"missing key: {key}")


if __name__ == "__main__":
    unittest.main()
