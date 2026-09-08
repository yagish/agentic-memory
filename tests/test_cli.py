import os
import sys
import unittest
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import cmd_get_session, cmd_search, cmd_status, cmd_tail
from memory.db import init_db, insert_fact, insert_episodic, upsert_session, upsert_working_memory


def make_args(**kwargs):
    class Args:
        pass

    args = Args()
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def capture(fn, *args, **kwargs):
    buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    try:
        fn(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return buffer.getvalue()


class CliTestBase(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        upsert_session(
            self.conn,
            "sess-space",
            "claude",
            [
                {"role": "user", "content": "Tell me about rocket propulsion."},
                {"role": "assistant", "content": "Rockets work by expelling mass."},
            ],
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
        )
        upsert_session(
            self.conn,
            "sess-cooking",
            "claude",
            [
                {"role": "user", "content": "How do I make pasta?"},
                {"role": "assistant", "content": "Boil water, add salt, cook pasta."},
            ],
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:01:00Z",
        )
        insert_fact(
            self.conn,
            entity="user",
            attribute="name",
            value="Yash",
            semantic_content="My name is Yash. What's my name? Yash.",
            tags=["identity"],
            session_id="sess-space",
        )
        insert_episodic(
            self.conn,
            session_id="sess-cooking",
            title="Made pasta plan",
            abstract="Discussed how to cook pasta.",
            happened_at="2026-01-02T00:01:00Z",
            details={"outcomes": ["Recipe explained"]},
        )
        upsert_working_memory(
            self.conn,
            session_id="sess-cooking",
            current_goal="Cook pasta tonight",
            current_focus="Choosing the sauce",
            next_step="Boil the water",
            status="ready_to_resume",
            updated_at="2026-01-02T00:01:30Z",
            details={"active_tasks": ["Boil water", "Salt the pasta water"]},
        )

        import cli as cli_module

        self._orig_get_conn = cli_module.get_conn
        cli_module.get_conn = lambda: self.conn

    def tearDown(self):
        import cli as cli_module

        cli_module.get_conn = self._orig_get_conn
        self.conn.close()


class TestCmdStatus(CliTestBase):
    def test_shows_current_counts(self):
        output = capture(cmd_status, make_args())
        self.assertIn("Sessions : 2", output)
        self.assertIn("Turns    : 4", output)
        self.assertIn("Facts    : 1", output)
        self.assertIn("Episodes : 1", output)
        self.assertIn("Working : 1", output)


class TestCmdSearch(CliTestBase):
    def test_search_finds_matching_session(self):
        output = capture(cmd_search, make_args(query="rocket", limit=10))
        self.assertIn("sess-space", output)

    def test_search_no_match_says_no_results(self):
        output = capture(cmd_search, make_args(query="photosynthesis", limit=10))
        self.assertIn("No results", output)


class TestCmdTail(CliTestBase):
    def test_tail_shows_recent_session_preview(self):
        output = capture(cmd_tail, make_args(n=1))
        self.assertIn("sess-cooking", output)
        self.assertIn("How do I make pasta?", output)
        self.assertNotIn("sess-space", output)


class TestCmdGetSession(CliTestBase):
    def test_get_session_outputs_json(self):
        output = capture(cmd_get_session, make_args(session_id="sess-space"))
        self.assertIn('"session_id": "sess-space"', output)
        self.assertIn('"turn_count": 2', output)

    def test_get_session_unknown_exits(self):
        with self.assertRaises(SystemExit) as ctx:
            cmd_get_session(make_args(session_id="missing"))
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
