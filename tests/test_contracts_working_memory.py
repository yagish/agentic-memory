import unittest

from pydantic import ValidationError

from memory.contracts import WorkingMemory


class TestWorkingMemoryContract(unittest.TestCase):
    def test_accepts_valid_working_memory_and_normalizes_fields(self):
        memory = WorkingMemory.model_validate(
            {
                "current_goal": "  Finish auth middleware refactor  ",
                "current_focus": "  Regression coverage for refresh-token flows  ",
                "active_tasks": [" Add refresh-token tests ", "Add refresh-token tests", "Add expired-session tests"],
                "constraints": [" Keep branch fix/auth-middleware ", "Keep branch fix/auth-middleware"],
                "next_step": "  Write the refresh-token regression test  ",
                "status": " Ready To Resume ",
                "confidence": 0.9,
                "source_session_id": "session-123",
                "updated_at": "2026-09-06T10:00:00Z",
            }
        )

        self.assertEqual(memory.current_goal, "Finish auth middleware refactor")
        self.assertEqual(memory.current_focus, "Regression coverage for refresh-token flows")
        self.assertEqual(memory.active_tasks, ("Add refresh-token tests", "Add expired-session tests"))
        self.assertEqual(memory.constraints, ("Keep branch fix/auth-middleware",))
        self.assertEqual(memory.next_step, "Write the refresh-token regression test")
        self.assertEqual(memory.status, "ready_to_resume")

    def test_rejects_missing_required_field(self):
        with self.assertRaises(ValidationError):
            WorkingMemory.model_validate(
                {
                    "current_focus": "Regression coverage",
                    "active_tasks": ["Add tests"],
                    "next_step": "Write tests",
                    "status": "in_progress",
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-06T10:00:00Z",
                }
            )

    def test_rejects_empty_active_tasks(self):
        with self.assertRaises(ValidationError):
            WorkingMemory.model_validate(
                {
                    "current_goal": "Finish auth middleware refactor",
                    "current_focus": "Regression coverage",
                    "active_tasks": [],
                    "next_step": "Write tests",
                    "status": "in_progress",
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-06T10:00:00Z",
                }
            )

    def test_rejects_unknown_status(self):
        with self.assertRaises(ValidationError):
            WorkingMemory.model_validate(
                {
                    "current_goal": "Finish auth middleware refactor",
                    "current_focus": "Regression coverage",
                    "active_tasks": ["Add tests"],
                    "next_step": "Write tests",
                    "status": "paused",
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-06T10:00:00Z",
                }
            )

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            WorkingMemory.model_validate(
                {
                    "current_goal": "Finish auth middleware refactor",
                    "current_focus": "Regression coverage",
                    "active_tasks": ["Add tests"],
                    "next_step": "Write tests",
                    "status": "in_progress",
                    "owner": "platform",
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-06T10:00:00Z",
                }
            )


if __name__ == "__main__":
    unittest.main()
