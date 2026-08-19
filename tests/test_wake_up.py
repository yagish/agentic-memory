# test_wake_up.py — tests for the wake-up hook's current retrieval pipeline.

import io
import json
import os
import sys
import unittest
from unittest.mock import ANY, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.wake_up import _build_fact_query, _build_injection, main
import hooks.wake_up as _wu


class TestBuildFactQuery(unittest.TestCase):
    def test_strips_punctuation_and_dedupes_terms(self):
        result = _build_fact_query("What is my name, my role, and my name?")
        self.assertEqual(result, "what OR my OR name OR role OR and")

    def test_returns_empty_when_no_meaningful_terms(self):
        result = _build_fact_query("?!")
        self.assertEqual(result, "")


class TestBuildInjection(unittest.TestCase):
    def test_returns_empty_when_no_sections_exist(self):
        result = _build_injection(None, None, [], [], [])
        self.assertEqual(result, "")

    def test_cache_hit_includes_from_memory_instruction(self):
        result = _build_injection(
            {"similarity": 0.97, "content": "Task: fix auth bug"},
            None,
            [],
            [],
            [],
        )
        self.assertIn("[Cached Session — 97% match]", result)
        self.assertIn("Task: fix auth bug", result)
        self.assertIn("[From Memory]", result)

    def test_includes_working_memory_facts_and_procedural_sections(self):
        result = _build_injection(
            None,
            {"summary": "Current task context"},
            [{"similarity": 0.82, "content": "Past summary"}],
            [{"content": "User prefers dark mode"}],
            [{"title": "Deploy workflow", "steps": "1. Build\n2. Ship"}],
        )
        self.assertIn("[Working Memory — current task context]", result)
        self.assertIn("[Relevant Past Work]", result)
        self.assertIn("(82% match)", result)
        self.assertIn("[Relevant Facts — authoritative]", result)
        self.assertIn("User prefers dark mode", result)
        self.assertIn("[How-To Patterns]", result)
        self.assertIn("Deploy workflow", result)


class TestMainIntegration(unittest.TestCase):
    def _run_main(
        self,
        *,
        prompt="hello",
        session_id="test-sess",
        search_results=None,
        working_mem=None,
        facts=None,
        procedural=None,
        first_message=True,
    ):
        payload = json.dumps({"session_id": session_id, "prompt": prompt})
        output = io.StringIO()
        conn = MagicMock()

        with patch("sys.stdin", io.StringIO(payload)), \
             patch("sys.stdout", output), \
             patch("sys.exit", side_effect=SystemExit), \
             patch.object(_wu, "DB_PATH", "/fake/test.db"), \
             patch("os.path.exists", return_value=True), \
             patch.object(_wu, "init_db", return_value=conn), \
             patch.object(_wu, "embed", return_value=[0.1, 0.2]), \
             patch.object(_wu, "search_compacted_sessions", return_value=search_results or []), \
             patch.object(_wu, "increment_compacted_hit") as increment_hit, \
             patch.object(_wu, "_is_first_message", return_value=first_message), \
             patch.object(_wu, "get_active_working_memory", return_value=working_mem), \
             patch.object(_wu, "search_facts", return_value=facts or []), \
             patch.object(_wu, "search_procedural", return_value=procedural or []), \
             patch.object(_wu, "log_retrieval") as log_retrieval, \
             patch.object(_wu, "activity_log"):
            with self.assertRaises(SystemExit):
                main()

        return {
            "output": json.loads(output.getvalue()),
            "conn": conn,
            "increment_hit": increment_hit,
            "log_retrieval": log_retrieval,
        }

    def test_outputs_empty_object_when_nothing_matches(self):
        result = self._run_main(prompt="hello", first_message=False)
        self.assertEqual(result["output"], {})
        result["conn"].close.assert_called_once()

    def test_cache_hit_is_injected_and_hit_count_incremented(self):
        result = self._run_main(
            prompt="fix auth bug",
            search_results=[
                {"id": "cs-1", "cluster_id": "cluster-1", "content": "Task: fix auth bug", "similarity": 0.97}
            ],
            first_message=False,
        )
        suffix = result["output"]["hookSpecificOutput"]["userPromptSuffix"]
        self.assertEqual(result["output"]["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit")
        self.assertIn("[Cached Session — 97% match]", suffix)
        self.assertIn("Task: fix auth bug", suffix)
        result["increment_hit"].assert_called_once_with(result["conn"], "cs-1")
        result["log_retrieval"].assert_called_once()

    def test_working_memory_only_on_first_message(self):
        search_results = [
            {"id": "cs-2", "cluster_id": "cluster-2", "content": "Task: prior work", "similarity": 0.82}
        ]

        first = self._run_main(
            prompt="continue the task",
            search_results=search_results,
            working_mem={"summary": "Current task context"},
            first_message=True,
        )
        first_suffix = first["output"]["hookSpecificOutput"]["userPromptSuffix"]
        self.assertIn("Current task context", first_suffix)

        later = self._run_main(
            prompt="continue the task",
            search_results=search_results,
            working_mem={"summary": "Current task context"},
            first_message=False,
        )
        later_suffix = later["output"]["hookSpecificOutput"]["userPromptSuffix"]
        self.assertNotIn("Current task context", later_suffix)

    def test_procedural_search_only_for_how_to_prompts(self):
        with patch("sys.stdin", io.StringIO(json.dumps({"session_id": "s1", "prompt": "hello there"}))), \
             patch("sys.stdout", io.StringIO()), \
             patch("sys.exit", side_effect=SystemExit), \
             patch.object(_wu, "DB_PATH", "/fake/test.db"), \
             patch("os.path.exists", return_value=True), \
             patch.object(_wu, "init_db", return_value=MagicMock()), \
             patch.object(_wu, "embed", return_value=[0.1]), \
             patch.object(_wu, "search_compacted_sessions", return_value=[]), \
             patch.object(_wu, "_is_first_message", return_value=False), \
             patch.object(_wu, "search_facts", return_value=[]), \
             patch.object(_wu, "search_procedural", return_value=[]) as search_procedural:
            with self.assertRaises(SystemExit):
                main()
        search_procedural.assert_not_called()

        with patch("sys.stdin", io.StringIO(json.dumps({"session_id": "s2", "prompt": "how should i deploy this?"}))), \
             patch("sys.stdout", io.StringIO()), \
             patch("sys.exit", side_effect=SystemExit), \
             patch.object(_wu, "DB_PATH", "/fake/test.db"), \
             patch("os.path.exists", return_value=True), \
             patch.object(_wu, "init_db", return_value=MagicMock()), \
             patch.object(_wu, "embed", return_value=[0.1]), \
             patch.object(_wu, "search_compacted_sessions", return_value=[]), \
             patch.object(_wu, "_is_first_message", return_value=False), \
             patch.object(_wu, "search_facts", return_value=[]), \
             patch.object(_wu, "search_procedural", return_value=[]) as search_procedural, \
             patch.object(_wu, "log_retrieval"), \
             patch.object(_wu, "activity_log"):
            with self.assertRaises(SystemExit):
                main()
        search_procedural.assert_called_once_with(ANY, "how should i deploy this?", min_confidence=0.6, limit=3)
        

if __name__ == "__main__":
    unittest.main()
