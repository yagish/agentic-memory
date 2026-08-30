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

    def test_ingest_session_stores_session_and_chunks(self):
        def fake_embed(text: str) -> list[float]:
            return [0.1] * 384

        outcome = ingest_session(
            self.conn,
            session_id="sess-1",
            agent="assistant",
            turns=self.turns,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
            embed_fn=fake_embed,
        )

        self.assertEqual(outcome.session_id, "sess-1")
        self.assertEqual(outcome.turn_count, 2)
        self.assertTrue(outcome.embedding_stored)
        self.assertEqual(outcome.chunk_count, 1)
        self.assertEqual(outcome.warnings, [])

        row = self.conn.execute(
            "SELECT transcript, turn_count FROM sessions WHERE session_id = ?",
            ("sess-1",),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual(json.loads(row["transcript"]), self.turns)

        cache_row = self.conn.execute(
            "SELECT prompt, response, source_session_id FROM response_cache"
        ).fetchone()
        self.assertIsNotNone(cache_row)
        self.assertEqual(cache_row["prompt"], "How do I run the tests?")
        self.assertEqual(cache_row["response"], "Run python3 -m pytest -q.")
        self.assertEqual(cache_row["source_session_id"], "sess-1")

        chunk_count = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?",
            ("sess-1",),
        ).fetchone()[0]
        self.assertEqual(chunk_count, 1)

    def test_ingest_session_keeps_session_when_embedding_fails(self):
        def failing_embed(_text: str) -> list[float]:
            raise RuntimeError("embed boom")

        outcome = ingest_session(
            self.conn,
            session_id="sess-2",
            agent="assistant",
            turns=self.turns,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
            embed_fn=failing_embed,
        )

        self.assertFalse(outcome.embedding_stored)
        self.assertEqual(outcome.chunk_count, 0)
        self.assertEqual(len(outcome.warnings), 2)
        self.assertEqual(outcome.warnings[0].stage, "embedding")
        self.assertEqual(outcome.warnings[1].stage, "chunking")

        row = self.conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            ("sess-2",),
        ).fetchone()
        self.assertIsNotNone(row)

    def test_ingest_session_replaces_old_chunks(self):
        def fake_embed(text: str) -> list[float]:
            return [0.2] * 384

        ingest_session(
            self.conn,
            session_id="sess-3",
            agent="assistant",
            turns=self.turns,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
            embed_fn=fake_embed,
        )

        longer_turns = self.turns + [
            {"role": "user", "content": "Anything else?"},
            {"role": "assistant", "content": "Nope."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "You're welcome."},
            {"role": "user", "content": "Bye."},
        ]

        outcome = ingest_session(
            self.conn,
            session_id="sess-3",
            agent="assistant",
            turns=longer_turns,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:02:00Z",
            embed_fn=fake_embed,
        )

        chunk_rows = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?",
            ("sess-3",),
        ).fetchone()[0]
        self.assertEqual(chunk_rows, outcome.chunk_count)
        self.assertEqual(chunk_rows, 2)


if __name__ == "__main__":
    unittest.main()
