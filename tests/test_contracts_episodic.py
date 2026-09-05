import unittest

from pydantic import ValidationError

from memory.contracts import EpisodicMemory


class TestEpisodicMemoryContract(unittest.TestCase):
    def test_accepts_valid_episodic_memory_and_normalizes_lists(self):
        memory = EpisodicMemory.model_validate(
            {
                "title": "  Auth middleware decision  ",
                "abstract": "  Moved token validation into shared middleware. Login loop is fixed.  ",
                "participants": [" Yash ", "Dana", "Yash"],
                "decisions": [" Move token validation into shared middleware "],
                "outcomes": [" Login loop fixed ", "Login loop fixed"],
                "follow_ups": [" Add regression tests "],
                "confidence": 0.88,
                "source_session_id": "session-123",
                "happened_at": "2026-09-04T10:00:00Z",
            }
        )

        self.assertEqual(memory.title, "Auth middleware decision")
        self.assertEqual(
            memory.abstract,
            "Moved token validation into shared middleware. Login loop is fixed.",
        )
        self.assertEqual(memory.participants, ("Yash", "Dana"))
        self.assertEqual(memory.decisions, ("Move token validation into shared middleware",))
        self.assertEqual(memory.outcomes, ("Login loop fixed",))
        self.assertEqual(memory.follow_ups, ("Add regression tests",))
        self.assertEqual(memory.confidence, 0.88)
        self.assertEqual(memory.source_session_id, "session-123")

    def test_rejects_missing_required_field(self):
        with self.assertRaises(ValidationError):
            EpisodicMemory.model_validate(
                {
                    "abstract": "Resolved the auth bug.",
                    "source_session_id": "session-123",
                    "happened_at": "2026-09-04T10:00:00Z",
                }
            )

    def test_rejects_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            EpisodicMemory.model_validate(
                {
                    "title": "Auth fix",
                    "abstract": "Resolved the auth bug.",
                    "confidence": 1.5,
                    "source_session_id": "session-123",
                    "happened_at": "2026-09-04T10:00:00Z",
                }
            )

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            EpisodicMemory.model_validate(
                {
                    "title": "Auth fix",
                    "abstract": "Resolved the auth bug.",
                    "source_session_id": "session-123",
                    "happened_at": "2026-09-04T10:00:00Z",
                    "cluster_id": "cluster-1",
                }
            )


if __name__ == "__main__":
    unittest.main()
