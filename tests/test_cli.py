# test_cli.py — tests for the Phase 6 CLI commands and retrieval logging.
#
# Run with:  python3 -m unittest tests.test_cli -v

import json
import sys
import os
import unittest
from io import StringIO   # StringIO lets us capture print() output in tests

# Add the project root to the module search path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import init_db, upsert_session, log_retrieval

# Import the CLI command functions directly so we can call them without subprocess.
from cli import cmd_status, cmd_search, cmd_tail, cmd_get_session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_args(**kwargs):
    """
    Build a simple namespace object that mimics what argparse returns.
    CLI functions receive an `args` object with attributes — this creates one cheaply.
    """
    class Args:
        pass
    a = Args()
    for k, v in kwargs.items():
        setattr(a, k, v)
    return a


def capture(fn, *args, **kwargs):
    """
    Run a function and return whatever it printed to stdout as a string.
    We temporarily replace sys.stdout with a StringIO buffer.
    """
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared test database setup
# ---------------------------------------------------------------------------

class CliTestBase(unittest.TestCase):
    """
    Base class for CLI tests. Sets up an in-memory database with sample data
    and patches cli.py's DB_PATH to point to our test connection.
    """

    def setUp(self):
        # Create a fresh in-memory database for each test.
        self.conn = init_db(":memory:")

        # Seed two sessions with different topics.
        upsert_session(
            self.conn, "sess-space", "claude",
            [
                {"role": "user",      "content": "Tell me about rocket propulsion."},
                {"role": "assistant", "content": "Rockets work by expelling mass."},
            ],
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
        )
        upsert_session(
            self.conn, "sess-cooking", "claude",
            [
                {"role": "user",      "content": "How do I make pasta?"},
                {"role": "assistant", "content": "Boil water, add salt, cook pasta."},
            ],
            "2026-01-02T00:00:00Z", "2026-01-02T00:01:00Z",
        )

        # Patch the CLI module so its get_conn() returns our test connection.
        import cli as cli_module
        self._orig_get_conn = cli_module.get_conn
        cli_module.get_conn = lambda: self.conn

    def tearDown(self):
        # Restore the original get_conn so other tests are unaffected.
        import cli as cli_module
        cli_module.get_conn = self._orig_get_conn
        self.conn.close()


# ---------------------------------------------------------------------------
# Tests: status command
# ---------------------------------------------------------------------------

class TestCmdStatus(CliTestBase):

    def test_shows_session_count(self):
        # status should report 2 sessions (we seeded 2 above).
        output = capture(cmd_status, make_args())
        self.assertIn("2", output)

    def test_shows_turn_count(self):
        # Each session has 2 turns → total 4 turns.
        output = capture(cmd_status, make_args())
        self.assertIn("4", output)

    def test_output_contains_headers(self):
        output = capture(cmd_status, make_args())
        self.assertIn("Sessions stored", output)
        self.assertIn("Turns stored", output)
        self.assertIn("Tokens", output)


# ---------------------------------------------------------------------------
# Tests: search command
# ---------------------------------------------------------------------------

class TestCmdSearch(CliTestBase):

    def test_search_finds_matching_session(self):
        # "rocket" only appears in sess-space.
        output = capture(cmd_search, make_args(query="rocket", limit=10))
        self.assertIn("sess-space", output)

    def test_search_no_match_says_no_results(self):
        output = capture(cmd_search, make_args(query="photosynthesis", limit=10))
        self.assertIn("No results", output)

    def test_search_shows_snippet(self):
        output = capture(cmd_search, make_args(query="pasta", limit=10))
        self.assertIn("Excerpt", output)


# ---------------------------------------------------------------------------
# Tests: tail command
# ---------------------------------------------------------------------------

class TestCmdTail(CliTestBase):

    def test_tail_shows_sessions(self):
        output = capture(cmd_tail, make_args(n=10))
        # Both sessions should appear.
        self.assertIn("sess-space", output)
        self.assertIn("sess-cooking", output)

    def test_tail_respects_limit(self):
        # With n=1, only one session should appear.
        output = capture(cmd_tail, make_args(n=1))
        # sess-cooking is more recent so it should be the one shown.
        self.assertIn("sess-cooking", output)
        self.assertNotIn("sess-space", output)

    def test_tail_shows_preview(self):
        output = capture(cmd_tail, make_args(n=10))
        # The first user message from each session should appear as preview.
        self.assertIn("rocket", output)


# ---------------------------------------------------------------------------
# Tests: get-session command
# ---------------------------------------------------------------------------

class TestCmdGetSession(CliTestBase):

    def test_get_session_shows_transcript(self):
        output = capture(cmd_get_session, make_args(session_id="sess-space"))
        # The session ID, role labels, and content should all appear.
        self.assertIn("sess-space", output)
        self.assertIn("USER", output)
        self.assertIn("rocket", output)

    def test_get_session_unknown_exits(self):
        # For an unknown session_id, the command should exit with code 1.
        with self.assertRaises(SystemExit) as ctx:
            cmd_get_session(make_args(session_id="nonexistent"))
        self.assertEqual(ctx.exception.code, 1)


# ---------------------------------------------------------------------------
# Tests: log_retrieval (the db function, not the CLI)
# ---------------------------------------------------------------------------

class TestLogRetrieval(unittest.TestCase):

    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_log_creates_row(self):
        log_retrieval(self.conn, "memory_search", "rockets", 512)
        count = self.conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
        self.assertEqual(count, 1)

    def test_log_stores_correct_fields(self):
        log_retrieval(self.conn, "memory_search", "pasta", 256)
        row = self.conn.execute(
            "SELECT tool, query, result_size FROM retrievals"
        ).fetchone()
        self.assertEqual(row["tool"], "memory_search")
        self.assertEqual(row["query"], "pasta")
        self.assertEqual(row["result_size"], 256)

    def test_multiple_logs(self):
        log_retrieval(self.conn, "memory_search", "a", 100)
        log_retrieval(self.conn, "memory_status", None, 50)
        log_retrieval(self.conn, "memory_get_session", "sess-1", 800)
        count = self.conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
        self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()
