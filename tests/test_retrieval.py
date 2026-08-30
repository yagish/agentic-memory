import unittest
from contextlib import ExitStack
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
        self.assertNotIn("Return", result)

    def test_includes_working_memory_facts_and_procedural_sections(self):
        result = build_wake_up_injection(
            WakeUpContext(
                None,
                {"summary": "Current task context"},
                [{"similarity": 0.82, "content": "Past summary"}],
                [{"content": "User prefers dark mode"}],
                [{"title": "Deploy workflow", "steps": "1. Build\n2. Ship"}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Current task context", result)
        self.assertIn("Relevant prior conversation (82% match)", result)
        self.assertIn("User prefers dark mode.", result)
        self.assertIn("Relevant how-to pattern", result)
        self.assertIn("Deploy workflow", result)
        self.assertNotIn("verbatim", result)


class TestRetrieveWakeUpContext(unittest.TestCase):
    def test_cache_hit_and_working_memory_are_selected(self):
        conn = MagicMock()

        with patch("memory.retrieval.find_cached_response", return_value=None), \
             patch("memory.retrieval.search_compacted_sessions", return_value=[
                 {"id": "cs-1", "cluster_id": "cluster-1", "content": "Task: fix auth bug", "similarity": 0.97}
             ]), patch("memory.retrieval.search_working_memory_semantic", return_value=[
                 {"id": "wm-1", "summary": "Current task context", "similarity": 0.91}
             ]) as search_working_memory, \
             patch("memory.retrieval.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval.search_procedural_semantic", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "fix auth bug",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        self.assertIsNone(context.cache_hit)
        self.assertEqual(context.working_mem["id"], "wm-1")
        self.assertEqual(context.enrichment[0]["id"], "cs-1")
        self.assertEqual(context.facts, [])
        self.assertEqual(context.procedural, [])
        self.assertEqual(context.warnings, [])
        search_working_memory.assert_called_once_with(conn, [0.1, 0.2], limit=1)

    def test_exact_cached_response_short_circuits_other_retrieval(self):
        conn = MagicMock()

        with patch("memory.retrieval.find_cached_response", return_value={
            "id": "cache-1", "response": "Use the existing deployment workflow.", "similarity": 1.0
        }), patch("memory.retrieval.search_compacted_sessions") as search_compacted:
            context = retrieve_wake_up_context(
                conn,
                "How do I deploy?",
                include_working_memory=True,
                embed_fn=lambda _prompt: self.fail("cache hit should not embed"),
            )

        self.assertEqual(context.cache_hit["id"], "cache-1")
        self.assertIsNone(context.working_mem)
        search_compacted.assert_not_called()

    def test_procedural_search_only_for_how_to_prompts(self):
        conn = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("memory.retrieval.find_cached_response", return_value=None))
            stack.enter_context(patch("memory.retrieval.search_compacted_sessions", return_value=[]))
            stack.enter_context(patch("memory.retrieval.search_facts_semantic", return_value=[]))
            search_procedural = stack.enter_context(
                patch("memory.retrieval.search_procedural_semantic", return_value=[])
            )
            retrieve_wake_up_context(
                conn,
                "hello there",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )
        search_procedural.assert_not_called()

        with ExitStack() as stack:
            stack.enter_context(patch("memory.retrieval.find_cached_response", return_value=None))
            stack.enter_context(patch("memory.retrieval.search_compacted_sessions", return_value=[]))
            stack.enter_context(patch("memory.retrieval.search_facts_semantic", return_value=[]))
            search_procedural = stack.enter_context(
                patch("memory.retrieval.search_procedural_semantic", return_value=[])
            )
            retrieve_wake_up_context(
                conn,
                "how should i deploy this?",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )
        search_procedural.assert_called_once_with(conn, [0.1], min_confidence=0.6, limit=3)

    def test_non_fatal_retrieval_errors_become_warnings(self):
        conn = MagicMock()

        with patch("memory.retrieval.find_cached_response", return_value=None), \
             patch("memory.retrieval.search_compacted_sessions", side_effect=RuntimeError("compacted boom")), \
             patch("memory.retrieval.search_facts_semantic", side_effect=RuntimeError("facts boom")):
            context = retrieve_wake_up_context(
                conn,
                "what is my name?",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.cache_hit, None)
        self.assertEqual(context.enrichment, [])
        self.assertEqual(context.facts, [])
        self.assertEqual([warning.stage for warning in context.warnings], ["compacted_search", "facts"])


if __name__ == "__main__":
    unittest.main()
