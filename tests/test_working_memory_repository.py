import unittest

from memory.contracts import ExtractedWorkingMemory
from memory.db import init_db
from memory.working_memory.repository import (
    build_working_memory_semantic_text,
    list_working_memories,
    retrieve_working_memory,
    save_extracted_working_memory,
)


class TestWorkingMemoryRepository(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_build_semantic_text_is_deterministic(self):
        memory = ExtractedWorkingMemory(
            current_goal="Finish auth middleware refactor",
            current_focus="Regression coverage for refresh-token flows",
            active_tasks=("Add refresh-token tests", "Add expired-session tests"),
            constraints=("Keep branch fix/auth-middleware",),
            next_step="Write the refresh-token regression test",
            status="ready_to_resume",
        )

        semantic_text = build_working_memory_semantic_text(memory)
        self.assertIn("Current goal: Finish auth middleware refactor", semantic_text)
        self.assertIn("Current focus: Regression coverage for refresh-token flows", semantic_text)
        self.assertIn("Active tasks: Add refresh-token tests; Add expired-session tests", semantic_text)
        self.assertIn("Constraints: Keep branch fix/auth-middleware", semantic_text)
        self.assertIn("Status: ready_to_resume", semantic_text)

    def test_save_extracted_working_memory_persists_structured_snapshot(self):
        memory = ExtractedWorkingMemory(
            current_goal="Finish auth middleware refactor",
            current_focus="Regression coverage for refresh-token flows",
            active_tasks=("Add refresh-token tests", "Add expired-session tests"),
            constraints=("Keep branch fix/auth-middleware",),
            next_step="Write the refresh-token regression test",
            status="ready_to_resume",
            confidence=0.91,
        )

        saved_id = save_extracted_working_memory(
            self.conn,
            memory,
            session_id="session-123",
            updated_at="2026-09-06T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
            source="working_test",
        )

        rows = list_working_memories(self.conn)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], saved_id)
        self.assertEqual(rows[0]["current_goal"], "Finish auth middleware refactor")
        self.assertEqual(rows[0]["current_focus"], "Regression coverage for refresh-token flows")
        self.assertEqual(rows[0]["active_tasks"], ["Add refresh-token tests", "Add expired-session tests"])
        self.assertEqual(rows[0]["constraints"], ["Keep branch fix/auth-middleware"])
        self.assertEqual(rows[0]["status"], "ready_to_resume")
        self.assertEqual(rows[0]["source"], "working_test")

    def test_retrieve_working_memory_returns_row_for_session(self):
        save_extracted_working_memory(
            self.conn,
            ExtractedWorkingMemory(
                current_goal="Finish auth middleware refactor",
                current_focus="Regression coverage for refresh-token flows",
                active_tasks=("Add refresh-token tests",),
                next_step="Write the refresh-token regression test",
                status="ready_to_resume",
            ),
            session_id="session-123",
            updated_at="2026-09-06T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
        )

        row = retrieve_working_memory(self.conn, session_id="session-123", source="test-harness")

        self.assertIsNotNone(row)
        self.assertEqual(row["session_id"], "session-123")
        self.assertEqual(row["current_goal"], "Finish auth middleware refactor")
        self.assertEqual(row["status"], "ready_to_resume")
        self.assertEqual(row["similarity"], 1.0)


if __name__ == "__main__":
    unittest.main()
