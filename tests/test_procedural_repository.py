import unittest

from memory.contracts import ExtractedProcedure
from memory.db import init_db
from memory.procedural.repository import (
    build_procedural_semantic_text,
    list_session_procedures,
    retrieve_procedural_memories,
    save_extracted_procedure,
)


class TestProceduralRepository(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_build_semantic_text_is_deterministic(self):
        procedure = ExtractedProcedure(
            title="Web deploy workflow",
            summary="Use this when deploying the web service.",
            steps=("Build the Docker image", "Run alembic upgrade"),
            trigger_phrases=("how do i deploy the web service",),
            tools=("Docker", "alembic"),
        )

        semantic_text = build_procedural_semantic_text(procedure)
        self.assertIn("Web deploy workflow", semantic_text)
        self.assertIn("Steps: Build the Docker image; Run alembic upgrade", semantic_text)
        self.assertIn("Useful for: how do i deploy the web service", semantic_text)
        self.assertIn("Tools: Docker; alembic", semantic_text)

    def test_save_extracted_procedure_persists_structured_procedure(self):
        procedure = ExtractedProcedure(
            title="Web deploy workflow",
            summary="Use this when deploying the web service.",
            steps=("Build the Docker image", "Run alembic upgrade"),
            trigger_phrases=("how do i deploy the web service",),
            tools=("Docker",),
            confidence=0.9,
        )

        saved_id = save_extracted_procedure(
            self.conn,
            procedure,
            session_id="session-123",
            updated_at="2026-09-04T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
            source="procedural_test",
        )

        rows = list_session_procedures(self.conn, session_id="session-123")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], saved_id)
        self.assertEqual(rows[0]["title"], "Web deploy workflow")
        self.assertEqual(rows[0]["summary"], "Use this when deploying the web service.")
        self.assertEqual(rows[0]["steps"], ["Build the Docker image", "Run alembic upgrade"])
        self.assertEqual(rows[0]["trigger_phrases"], ["how do i deploy the web service"])
        self.assertEqual(rows[0]["tools"], ["Docker"])
        self.assertEqual(rows[0]["source"], "procedural_test")

    def test_retrieve_procedural_memories_filters_by_similarity(self):
        save_extracted_procedure(
            self.conn,
            ExtractedProcedure(
                title="Web deploy workflow",
                summary="Use this when deploying the web service.",
                steps=("Build the Docker image", "Run alembic upgrade"),
            ),
            session_id="session-deploy",
            updated_at="2026-09-04T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
        )
        save_extracted_procedure(
            self.conn,
            ExtractedProcedure(
                title="Invoice copy update",
                summary="Use this when editing billing email wording.",
                steps=("Open the billing templates", "Update the copy"),
            ),
            session_id="session-billing",
            updated_at="2026-09-04T10:05:00Z",
            embed_fn=lambda text: [0.0, 1.0],
        )

        results = retrieve_procedural_memories(
            self.conn,
            "how do i deploy the web service?",
            embed_fn=lambda prompt: [1.0, 0.0],
            min_similarity=0.72,
            source="test-harness",
        )

        self.assertEqual([row["session_id"] for row in results], ["session-deploy"])
        self.assertEqual(results[0]["title"], "Web deploy workflow")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)


if __name__ == "__main__":
    unittest.main()
