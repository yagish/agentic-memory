import unittest

from memory.contracts import ExtractedSessionMemory
from memory.db import init_db
from memory.session.repository import (
    build_session_memory_semantic_text,
    list_session_memory_rows,
    retrieve_session_memories,
    save_extracted_session_memory,
)


class TestSessionMemoryRepository(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_build_semantic_text_is_deterministic(self):
        memory = ExtractedSessionMemory(
            title="Auth middleware refactor",
            summary="Moved token validation into shared middleware and fixed the login redirect loop locally.",
            what_was_tried=("Moved token validation into shared middleware", "Retested the login loop locally"),
            outcomes=("Redirect loop stopped reproducing locally",),
            left_off_at="Regression coverage is still missing for refresh-token and expired-session flows",
            next_steps=("Add regression tests",),
        )

        semantic_text = build_session_memory_semantic_text(memory)
        self.assertIn("Auth middleware refactor", semantic_text)
        self.assertIn("Left off at: Regression coverage is still missing", semantic_text)
        self.assertIn("Tried: Moved token validation into shared middleware", semantic_text)
        self.assertIn("Outcomes: Redirect loop stopped reproducing locally", semantic_text)
        self.assertIn("Next session: Add regression tests", semantic_text)

    def test_save_extracted_session_memory_persists_structured_summary(self):
        memory = ExtractedSessionMemory(
            title="Auth middleware refactor",
            summary="Moved token validation into shared middleware and fixed the login redirect loop locally.",
            what_was_tried=("Moved token validation into shared middleware",),
            outcomes=("Redirect loop stopped reproducing locally",),
            left_off_at="Regression coverage is still missing for refresh-token and expired-session flows",
            next_steps=("Add regression tests",),
            confidence=0.91,
        )

        saved_id = save_extracted_session_memory(
            self.conn,
            memory,
            session_id="session-123",
            updated_at="2026-09-06T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
            source="session_test",
        )

        rows = list_session_memory_rows(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], saved_id)
        self.assertEqual(rows[0]["title"], "Auth middleware refactor")
        self.assertEqual(rows[0]["left_off_at"], "Regression coverage is still missing for refresh-token and expired-session flows")
        self.assertEqual(rows[0]["next_steps"], ["Add regression tests"])
        self.assertEqual(rows[0]["source"], "session_test")

    def test_retrieve_session_memories_filters_by_similarity_and_excludes_current_session(self):
        save_extracted_session_memory(
            self.conn,
            ExtractedSessionMemory(
                title="Auth middleware refactor",
                summary="Moved token validation into shared middleware and fixed the login redirect loop locally.",
                left_off_at="Regression coverage is still missing for refresh-token and expired-session flows",
                next_steps=("Add regression tests",),
            ),
            session_id="session-auth",
            updated_at="2026-09-06T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
        )
        save_extracted_session_memory(
            self.conn,
            ExtractedSessionMemory(
                title="Billing copy update",
                summary="Adjusted invoice wording for support.",
                left_off_at="Awaiting support review",
                next_steps=("Wait for support review",),
            ),
            session_id="session-billing",
            updated_at="2026-09-06T10:05:00Z",
            embed_fn=lambda text: [0.0, 1.0],
        )

        results = retrieve_session_memories(
            self.conn,
            "pick up auth middleware refactor where i left off",
            embed_fn=lambda prompt: [1.0, 0.0],
            min_similarity=0.76,
            source="test-harness",
            exclude_session_id="session-current",
        )

        self.assertEqual([row["session_id"] for row in results], ["session-auth"])
        self.assertEqual(results[0]["title"], "Auth middleware refactor")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)


if __name__ == "__main__":
    unittest.main()
