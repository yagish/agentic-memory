import unittest
from unittest.mock import MagicMock, patch

from memory.retrieval import (
    WakeUpContext,
    build_fact_query,
    build_wake_up_injection,
    retrieve_wake_up_context,
)


class TestBuildFactQuery(unittest.TestCase):
    def test_strips_punctuation_and_dedupes_terms(self):
        result = build_fact_query("What is my name, my role, and my name?")
        self.assertEqual(result, "what OR my OR name OR role OR and")

    def test_returns_empty_when_no_meaningful_terms(self):
        result = build_fact_query("?!")
        self.assertEqual(result, "")

class TestBuildWakeUpInjection(unittest.TestCase):
    def test_returns_empty_when_no_sections_exist(self):
        result = build_wake_up_injection(
            WakeUpContext(None, None, [], [], [])
        )
        self.assertEqual(result, "")

    def test_cache_hit_includes_from_memory_instruction(self):
        result = build_wake_up_injection(
            WakeUpContext(
                {"similarity": 1.0, "response": "The authentication bug is fixed."},
                None,
                [],
                [],
                [],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("You answered this question before (100% match)", result)
        self.assertIn("Previous answer:", result)
        self.assertIn("The authentication bug is fixed.", result)

    def test_includes_only_fact_sections_in_facts_only_mode(self):
        result = build_wake_up_injection(
            WakeUpContext(
                None,
                {"summary": "Current task context"},
                [{"similarity": 0.82, "content": "Past summary"}],
                [
                    {"content": "user.name = Yash"},
                    {"content": "user.timezone = EST"},
                ],
                [{"title": "Deploy workflow", "steps": "1. Build\n2. Ship"}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Remembered fact: user.name = Yash.", result)
        self.assertIn("Remembered fact: user.timezone = EST.", result)
        self.assertNotIn("Current task context", result)
        self.assertNotIn("Relevant prior conversation", result)
        self.assertNotIn("Relevant how-to pattern", result)


class TestRetrieveWakeUpContext(unittest.TestCase):
    def test_retrieval_only_searches_facts(self):
        conn = MagicMock()

        with patch("memory.retrieval.search_facts_semantic", return_value=[
            {"id": "fact-1", "content": "user.name = Yash", "similarity": 0.98}
        ]) as search_facts:
            context = retrieve_wake_up_context(
                conn,
                "what is my name?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        self.assertIsNone(context.cache_hit)
        self.assertIsNone(context.working_mem)
        self.assertEqual(context.enrichment, [])
        self.assertEqual(context.procedural, [])
        self.assertEqual(context.facts[0]["id"], "fact-1")
        self.assertEqual(context.warnings, [])
        search_facts.assert_called_once_with(conn, [0.1, 0.2], limit=5)

    def test_non_fact_prompt_still_searches_facts(self):
        conn = MagicMock()

        with patch("memory.retrieval.search_facts_semantic", return_value=[]) as search_facts:
            context = retrieve_wake_up_context(
                conn,
                "continue implementing this",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.facts, [])
        self.assertEqual(context.enrichment, [])
        self.assertEqual(context.procedural, [])
        self.assertEqual(context.warnings, [])
        search_facts.assert_called_once_with(conn, [0.1], limit=5)

    def test_non_fatal_fact_errors_become_warnings(self):
        conn = MagicMock()

        with patch("memory.retrieval.search_facts_semantic", side_effect=RuntimeError("facts boom")):
            context = retrieve_wake_up_context(
                conn,
                "what is my name?",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.cache_hit, None)
        self.assertEqual(context.enrichment, [])
        self.assertEqual(context.facts, [])
        self.assertEqual([warning.stage for warning in context.warnings], ["facts"])


if __name__ == "__main__":
    unittest.main()
