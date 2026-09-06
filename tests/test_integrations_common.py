import tempfile
import unittest
from unittest.mock import patch

from integrations.common import (
    RecallOutcome,
    decide_prompt_memory_action,
    open_existing_memory_db,
    open_memory_db_for_ingest,
    save_session_to_memory,
)
from memory.db import get_session_by_id
from memory.retrieval import WakeUpContext


class TestDecidePromptMemoryAction(unittest.TestCase):
    def test_returns_direct_answer_for_fact_only_hits(self):
        context = WakeUpContext(
            None,
            None,
            [],
            [],
            [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
            [],
        )

        with patch("integrations.common.render_fact_answer", return_value="Your name is Yash.") as render_fact_answer:
            outcome = decide_prompt_memory_action("what is my name?", context)

        self.assertIsInstance(outcome, RecallOutcome)
        self.assertEqual(outcome.action, "answer")
        self.assertEqual(outcome.answer, "Your name is Yash.")
        self.assertEqual(outcome.injection, "")
        render_fact_answer.assert_called_once_with("what is my name?", ["user.name = Yash"])

    def test_returns_noop_when_fact_hits_do_not_directly_answer_prompt(self):
        context = WakeUpContext(
            None,
            None,
            [],
            [],
            [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
            [],
        )

        with patch("integrations.common.render_fact_answer", return_value=""):
            outcome = decide_prompt_memory_action("how do i deploy this service?", context)

        self.assertEqual(outcome.action, "noop")
        self.assertEqual(outcome.answer, "")
        self.assertEqual(outcome.injection, "")

    def test_returns_injection_when_context_should_enrich_prompt(self):
        context = WakeUpContext(
            None,
            None,
            [],
            [{"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.91}],
            [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
            [{"id": "proc-1", "title": "Deploy service", "summary": "Use this when releasing auth.", "steps": ["Build the image"], "similarity": 0.87}],
        )

        outcome = decide_prompt_memory_action("continue fixing auth", context)

        self.assertEqual(outcome.action, "inject")
        self.assertEqual(outcome.answer, "")
        self.assertIn("Recent related episode: Resolved auth bug. Fixed the login loop.", outcome.injection)
        self.assertIn("Relevant how-to pattern: Deploy service. Use this when releasing auth.", outcome.injection)
        self.assertIn("Remembered fact: user.name = Yash.", outcome.injection)


class TestSharedSaveHelpers(unittest.TestCase):
    def test_open_existing_returns_none_when_db_missing(self):
        self.assertIsNone(open_existing_memory_db("/tmp/definitely-missing-agentic-memory.db"))

    def test_save_session_to_memory_persists_session(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            conn = open_memory_db_for_ingest(tmp.name)
            try:
                outcome = save_session_to_memory(
                    conn,
                    session_id="pi-session-1",
                    agent="pi",
                    turns=[
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ],
                    started_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:01:00+00:00",
                    metadata={"integration": "pi"},
                )
            finally:
                conn.close()

            conn = open_memory_db_for_ingest(tmp.name)
            try:
                session = get_session_by_id(conn, "pi-session-1")
            finally:
                conn.close()

        self.assertEqual(outcome.turn_count, 2)
        self.assertIsNotNone(session)
        self.assertEqual(session["agent"], "pi")


if __name__ == "__main__":
    unittest.main()
