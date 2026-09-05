import json
import unittest

from memory.db import init_db
from memory.ingest_pipeline import ingest_session


class TestIngestPipeline(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        self.turns = [
            {"role": "user", "content": "How do I run the tests?"},
            {"role": "assistant", "content": "Run python3 -m pytest -q."},
        ]

    def tearDown(self):
        self.conn.close()

    def test_ingest_session_stores_session_only(self):
        outcome = ingest_session(
            self.conn,
            session_id="sess-1",
            agent="assistant",
            turns=self.turns,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
        )

        self.assertEqual(outcome.session_id, "sess-1")
        self.assertEqual(outcome.turn_count, 2)
        self.assertFalse(outcome.embedding_stored)
        self.assertEqual(outcome.chunk_count, 0)
        self.assertEqual(outcome.warnings, [])

        row = self.conn.execute(
            "SELECT transcript, turn_count FROM sessions WHERE session_id = ?",
            ("sess-1",),
        ).fetchone()
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual(json.loads(row["transcript"]), self.turns)


if __name__ == "__main__":
    unittest.main()
