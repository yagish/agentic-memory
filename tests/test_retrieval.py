import unittest
from unittest.mock import MagicMock, patch

from memory.retrieval import (
    WakeUpContext,
    build_wake_up_injection,
    retrieve_wake_up_context,
)


class TestBuildWakeUpInjection(unittest.TestCase):
    def test_returns_empty_when_no_sections_exist(self):
        result = build_wake_up_injection(
            WakeUpContext(None, None, [], [], [], [])
        )
        self.assertEqual(result, "")

    def test_includes_episodic_fact_and_procedural_sections(self):
        result = build_wake_up_injection(
            WakeUpContext(
                {"id": "cs-1", "similarity": 0.98, "content": "Task: Fix auth middleware"},
                {"id": "wm-1", "summary": "Current task is cleaning up auth middleware", "similarity": 0.88},
                [{"id": "cs-2", "similarity": 0.83, "content": "Task: Add JWT refresh flow"}],
                [{
                    "id": "ep-1",
                    "title": "Resolved auth bug",
                    "abstract": "Fixed the login loop.",
                    "decisions": ["Move token validation into shared middleware"],
                    "outcomes": ["Login loop stopped reproducing"],
                    "follow_ups": ["Add regression coverage"],
                    "similarity": 0.79,
                }],
                [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.97}],
                [{"id": "proc-1", "title": "Deploy service", "summary": "Use this when releasing the auth service.", "steps": ["Build the image", "Ship to staging"], "similarity": 0.86}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Recent related episode: Resolved auth bug. Fixed the login loop.", result)
        self.assertIn("Decision: Move token validation into shared middleware.", result)
        self.assertIn("Outcome: Login loop stopped reproducing.", result)
        self.assertIn("Follow-up: Add regression coverage.", result)
        self.assertIn("Relevant how-to pattern: Deploy service. Use this when releasing the auth service.", result)
        self.assertIn("Steps: Build the image; Ship to staging.", result)
        self.assertIn("Remembered fact: user.name = Yash.", result)
        self.assertNotIn("Relevant prior session", result)
        self.assertNotIn("Current task context", result)
        self.assertNotIn("Related prior session", result)


class TestRetrieveWakeUpContext(unittest.TestCase):
    def test_retrieval_fetches_episodic_facts_and_procedural_memory(self):
        conn = MagicMock()

        with patch("memory.retrieval.retrieve_episodic_memories", return_value=[
            {"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.79}
        ]) as retrieve_episodic, \
             patch("memory.retrieval.search_facts_semantic", return_value=[
                 {"id": "fact-1", "content": "user.name = Yash", "similarity": 0.45},
                 {"id": "fact-2", "content": "user.timezone = EST", "similarity": 0.24},
             ]) as search_facts_semantic, \
             patch("memory.retrieval.retrieve_procedural_memories", return_value=[
                 {"id": "proc-1", "title": "Deploy service", "summary": "Use this when releasing the auth service.", "steps": ["Build the image"], "similarity": 0.88}
             ]) as retrieve_procedural:
            context = retrieve_wake_up_context(
                conn,
                "how do i deploy the auth service?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1, 0.2],
            )

        self.assertIsNone(context.cache_hit)
        self.assertIsNone(context.working_mem)
        self.assertEqual(context.enrichment, [])
        self.assertEqual([item["id"] for item in context.episodic], ["ep-1"])
        self.assertEqual([item["id"] for item in context.facts], ["fact-1"])
        self.assertEqual([item["id"] for item in context.procedural], ["proc-1"])
        self.assertEqual(context.warnings, [])
        retrieve_episodic.assert_called_once_with(
            conn,
            "how do i deploy the auth service?",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.72,
            limit=3,
            source="wake_up",
        )
        search_facts_semantic.assert_called_once_with(conn, [0.1, 0.2], limit=5)
        retrieve_procedural.assert_called_once_with(
            conn,
            "how do i deploy the auth service?",
            query_vector=[0.1, 0.2],
            embed_fn=unittest.mock.ANY,
            min_similarity=0.74,
            limit=2,
            source="wake_up",
        )

    def test_fact_retrieval_returns_no_hits_when_semantic_search_misses(self):
        conn = MagicMock()

        with patch("memory.retrieval.retrieve_episodic_memories", return_value=[]), \
             patch("memory.retrieval.search_facts_semantic", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "what is my favorite language?",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual(context.facts, [])

    def test_recent_episode_fallback_returns_substantive_recent_work_items(self):
        conn = MagicMock()

        recent_rows = [
            {
                "id": "ep-ignored",
                "title": "Name inquiry",
                "abstract": "The user asked about their name, but it was not stored in the assistant's memory.",
                "decisions": [],
                "outcomes": [],
                "follow_ups": [],
            },
            {
                "id": "ep-1",
                "title": "YAML error correction",
                "abstract": "The user encountered a YAML error and fixed it by correcting the nesting.",
                "decisions": ["Fix the YAML nesting"],
                "outcomes": ["Corrected the YAML file"],
                "follow_ups": [],
            },
            {
                "id": "ep-2",
                "title": "Token validation move",
                "abstract": "Moved token validation into shared auth middleware and fixed the login loop.",
                "decisions": ["Move token validation into shared auth middleware"],
                "outcomes": ["Login loop fixed"],
                "follow_ups": ["Add regression tests"],
            },
        ]

        with patch("memory.retrieval.retrieve_episodic_memories", return_value=[]), \
             patch("memory.retrieval.list_recent_episodic_memories", return_value=recent_rows) as list_recent, \
             patch("memory.retrieval.search_facts_semantic", return_value=[]):
            context = retrieve_wake_up_context(
                conn,
                "what was i working on last",
                include_working_memory=True,
                embed_fn=lambda prompt: [0.1],
            )

        self.assertEqual([item["id"] for item in context.episodic], ["ep-1", "ep-2"])
        list_recent.assert_called_once_with(conn, limit=5, source="wake_up_recent")

    def test_non_fatal_layer_errors_become_warnings(self):
        conn = MagicMock()

        with patch("memory.retrieval.retrieve_episodic_memories", side_effect=RuntimeError("episodic boom")), \
             patch("memory.retrieval.search_facts_semantic", side_effect=RuntimeError("facts boom")), \
             patch("memory.retrieval.retrieve_procedural_memories", side_effect=RuntimeError("procedural boom")):
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
            ["episodic", "facts", "procedural"],
        )


if __name__ == "__main__":
    unittest.main()
