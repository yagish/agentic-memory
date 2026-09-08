import unittest

from pydantic import ValidationError

from memory.contracts import SessionMemory


class TestSessionMemoryContract(unittest.TestCase):
    def test_accepts_valid_session_memory_and_normalizes_lists(self):
        memory = SessionMemory.model_validate(
            {
                "title": "  Auth middleware refactor  ",
                "summary": "  Moved token validation into shared middleware and fixed the login redirect loop locally.  ",
                "what_was_tried": [" Move token validation into shared middleware ", "Move token validation into shared middleware", "Retest the login loop locally"],
                "outcomes": [" Redirect loop stopped reproducing locally ", "Redirect loop stopped reproducing locally"],
                "left_off_at": "  Regression coverage is still missing for refresh-token and expired-session flows  ",
                "next_steps": [" Add regression tests ", "Add regression tests", "Verify expired-session flow"],
                "confidence": 0.92,
                "source_session_id": "session-123",
                "updated_at": "2026-09-06T10:00:00Z",
            }
        )

        self.assertEqual(memory.title, "Auth middleware refactor")
        self.assertEqual(memory.summary, "Moved token validation into shared middleware and fixed the login redirect loop locally.")
        self.assertEqual(memory.what_was_tried, ("Move token validation into shared middleware", "Retest the login loop locally"))
        self.assertEqual(memory.outcomes, ("Redirect loop stopped reproducing locally",))
        self.assertEqual(memory.left_off_at, "Regression coverage is still missing for refresh-token and expired-session flows")
        self.assertEqual(memory.next_steps, ("Add regression tests", "Verify expired-session flow"))

    def test_rejects_missing_required_field(self):
        with self.assertRaises(ValidationError):
            SessionMemory.model_validate(
                {
                    "summary": "Moved token validation into shared middleware.",
                    "left_off_at": "Regression coverage is still missing.",
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-06T10:00:00Z",
                }
            )

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            SessionMemory.model_validate(
                {
                    "title": "Auth middleware refactor",
                    "summary": "Moved token validation into shared middleware.",
                    "left_off_at": "Regression coverage is still missing.",
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-06T10:00:00Z",
                    "owner": "platform",
                }
            )


if __name__ == "__main__":
    unittest.main()
