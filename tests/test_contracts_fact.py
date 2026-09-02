import unittest

from pydantic import ValidationError

from memory.contracts import FactMemory


class TestFactMemoryContract(unittest.TestCase):
    def test_accepts_valid_fact_memory(self):
        fact = FactMemory.model_validate(
            {
                "entity": "user",
                "attribute": "name",
                "value": "Yash",
                "confidence": 0.95,
                "source_session_id": "session-123",
                "valid_from": "2026-09-02T10:00:00Z",
            }
        )

        self.assertEqual(fact.entity, "user")
        self.assertEqual(fact.attribute, "name")
        self.assertEqual(fact.value, "Yash")
        self.assertEqual(fact.confidence, 0.95)
        self.assertEqual(fact.source_session_id, "session-123")
        self.assertEqual(fact.valid_to, None)

    def test_rejects_missing_required_field(self):
        with self.assertRaises(ValidationError):
            FactMemory.model_validate(
                {
                    "entity": "user",
                    "value": "Yash",
                    "confidence": 0.95,
                    "source_session_id": "session-123",
                    "valid_from": "2026-09-02T10:00:00Z",
                }
            )

    def test_rejects_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            FactMemory.model_validate(
                {
                    "entity": "user",
                    "attribute": "name",
                    "value": "Yash",
                    "confidence": 1.5,
                    "source_session_id": "session-123",
                    "valid_from": "2026-09-02T10:00:00Z",
                }
            )

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            FactMemory.model_validate(
                {
                    "entity": "user",
                    "attribute": "name",
                    "value": "Yash",
                    "confidence": 0.95,
                    "source_session_id": "session-123",
                    "valid_from": "2026-09-02T10:00:00Z",
                    "nickname": "y",
                }
            )


if __name__ == "__main__":
    unittest.main()
