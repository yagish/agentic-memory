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
            WakeUpContext(None, None, [], [], [], [])
        )
        self.assertEqual(result, "")

    def test_includes_all_memory_sections(self):
        result = build_wake_up_injection(
            WakeUpContext(
                {"id": "cs-1", "similarity": 0.98, "content": "Task: Fix auth middleware"},
                {"id": "wm-1", "summary": "Current task is cleaning up auth middleware", "similarity": 0.88},
                [{"id": "cs-2", "similarity": 0.83, "content": "Task: Add JWT refresh flow"}],
                [{"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.79}],
                [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.97}],
                [{"id": "proc-1", "title": "Deploy service", "steps": "1. Build\n2. Ship", "similarity": 0.86}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Relevant prior session (98% match): Task: Fix auth middleware.", result)
        self.assertIn("Current task context: Current task is cleaning up auth middleware.", result)
        self.assertIn("Related prior session (83% match): Task: Add JWT refresh flow.", result)
        self.assertIn("Recent related episode: Resolved auth bug. Fixed the login loop.", result)
        self.assertIn("Remembered fact: user.name = Yash.", result)
        self.assertIn("Relevant how-to pattern: Deploy service. 1. Build 2. Ship.", result)


class TestRetrieveWakeUpContext(unittest.TestCase):
    def test_retrieval_searches_all_memory_layers(self):
        conn = MagicMock()

        with patch("memory.retrieval.search_compacted_sessions", return_value=[
            {"id": "cs-1", "content": "Task: Fix auth middleware", "similarity": 0.98},
            {"id": "cs-2", "content": "Task: Add JWT refresh flow", "similarity": 0.83},
        ]) as search_compacted, \
             patch("memory.retrieval.increment_compacted_hit") as increment_hit, \
             patch("memory.retrieval.search_working_memory_semantic", return_value=[
                 {"id": "wm-1", "summary": "Current task is auth cleanup", "similarity": 0.88}
             ]) as search_working, \
             patch("memory.retrieval.search_episodic_semantic", return_value=[
                 {"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.79}
             ]) as search_episodic, \
             patch("memory.retrieval.search_facts_semantic", return_value=[
                 {"id": "fact-1", "content": "user.name = Yash", "similarity": 0.97},
                 {"id": "fact-2", "content": "user.timezone = EST", "similarity": 0.55},
             ]) as search_facts, \
             patch("memory.retrieval.search_procedural_semantic", return_value=[
                 {"id": "proc-1", "title": "Deploy service", "steps": "1. Build\n2. Ship", "similarity": 0.86, "confidence": 0.8}
             ]) as search_procedural:
            context = retrieve_wake_up_context(
                conn,
                "how do i deploy the auth service?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        self.assertEqual(context.cache_hit["id"], "cs-1")
        self.assertEqual([item["id"] for item in context.enrichment], ["cs-2"])
        self.assertEqual(context.working_mem["id"], "wm-1")
        self.assertEqual([item["id"] for item in context.episodic], ["ep-1"])
        self.assertEqual([item["id"] for item in context.facts], ["fact-1"])
        self.assertEqual([item["id"] for item in context.procedural], ["proc-1"])
        self.assertEqual(context.warnings, [])
        search_compacted.assert_called_once_with(conn, [0.1, 0.2], limit=4)
        increment_hit.assert_called_once_with(conn, "cs-1")
        search_working.assert_called_once_with(conn, [0.1, 0.2], limit=1)
        search_episodic.assert_called_once_with(conn, [0.1, 0.2], limit=3)
        search_facts.assert_called_once_with(conn, [0.1, 0.2], limit=5)
        search_procedural.assert_called_once_with(conn, [0.1, 0.2], limit=3)

    def test_non_procedural_prompt_skips_procedural_search(self):
        conn = MagicMock()

        with patch("memory.retrieval.search_compacted_sessions", return_value=[]), \
             patch("memory.retrieval.search_episodic_semantic", return_value=[]), \
             patch("memory.retrieval.search_facts_semantic", return_value=[]), \
             patch("memory.retrieval.search_working_memory_semantic", return_value=[]), \
             patch("memory.retrieval.search_procedural_semantic") as search_procedural:
            context = retrieve_wake_up_context(
                conn,
                "what is my name?",
                include_working_memory=False,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.procedural, [])
        self.assertEqual(context.warnings, [])
        search_procedural.assert_not_called()

    def test_non_fatal_layer_errors_become_warnings(self):
        conn = MagicMock()

        with patch("memory.retrieval.search_compacted_sessions", side_effect=RuntimeError("compacted boom")), \
             patch("memory.retrieval.search_episodic_semantic", side_effect=RuntimeError("episodic boom")), \
             patch("memory.retrieval.search_facts_semantic", side_effect=RuntimeError("facts boom")), \
             patch("memory.retrieval.search_working_memory_semantic", side_effect=RuntimeError("working boom")), \
             patch("memory.retrieval.search_procedural_semantic", side_effect=RuntimeError("procedural boom")):
            context = retrieve_wake_up_context(
                conn,
                "how do i deploy?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.cache_hit, None)
        self.assertEqual(context.working_mem, None)
        self.assertEqual(context.enrichment, [])
        self.assertEqual(context.episodic, [])
        self.assertEqual(context.facts, [])
        self.assertEqual(context.procedural, [])
        self.assertEqual(
            [warning.stage for warning in context.warnings],
            ["compacted_sessions", "working_memory", "episodic", "facts", "procedural"],
        )


if __name__ == "__main__":
    unittest.main()
