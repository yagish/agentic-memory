import unittest

from pydantic import ValidationError

from memory.contracts import ProceduralMemory


class TestProceduralMemoryContract(unittest.TestCase):
    def test_accepts_valid_procedural_memory_and_normalizes_lists(self):
        memory = ProceduralMemory.model_validate(
            {
                "title": "  Web deploy workflow  ",
                "summary": "  Use this when deploying the web service to staging and production.  ",
                "steps": [" Build the Docker image ", "Run alembic upgrade", "Build the Docker image"],
                "trigger_phrases": [" how do i deploy the web service ", "deploy workflow", "deploy workflow"],
                "tools": [" Docker ", "alembic", "Docker"],
                "confidence": 0.9,
                "source_session_id": "session-123",
                "updated_at": "2026-09-04T10:00:00Z",
            }
        )

        self.assertEqual(memory.title, "Web deploy workflow")
        self.assertEqual(memory.summary, "Use this when deploying the web service to staging and production.")
        self.assertEqual(memory.steps, ("Build the Docker image", "Run alembic upgrade"))
        self.assertEqual(memory.trigger_phrases, ("how do i deploy the web service", "deploy workflow"))
        self.assertEqual(memory.tools, ("Docker", "alembic"))
        self.assertEqual(memory.confidence, 0.9)
        self.assertEqual(memory.source_session_id, "session-123")

    def test_rejects_missing_required_field(self):
        with self.assertRaises(ValidationError):
            ProceduralMemory.model_validate(
                {
                    "summary": "Deploy the service safely.",
                    "steps": ["Build image"],
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-04T10:00:00Z",
                }
            )

    def test_rejects_empty_steps(self):
        with self.assertRaises(ValidationError):
            ProceduralMemory.model_validate(
                {
                    "title": "Deploy workflow",
                    "summary": "Deploy the service safely.",
                    "steps": [],
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-04T10:00:00Z",
                }
            )

    def test_rejects_unknown_fields(self):
        with self.assertRaises(ValidationError):
            ProceduralMemory.model_validate(
                {
                    "title": "Deploy workflow",
                    "summary": "Deploy the service safely.",
                    "steps": ["Build image"],
                    "source_session_id": "session-123",
                    "updated_at": "2026-09-04T10:00:00Z",
                    "owner": "platform",
                }
            )


if __name__ == "__main__":
    unittest.main()
