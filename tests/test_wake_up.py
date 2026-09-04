# test_wake_up.py — tests for the wake-up hook adapter.

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.wake_up import _build_fact_query, _build_injection, main
from memory.retrieval import WakeUpContext
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
        result = _build_injection(WakeUpContext(None, None, [], [], []))
        self.assertEqual(result, "")

    def test_cache_hit_includes_from_memory_instruction(self):
        result = _build_injection(
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

    def test_includes_only_facts_in_facts_only_mode(self):
        result = _build_injection(
            WakeUpContext(
                None,
                {"summary": "Current task context"},
                [{"similarity": 0.82, "content": "Past summary"}],
                [{"content": "user.name = Yash"}],
                [{"title": "Deploy workflow", "steps": "1. Build\n2. Ship"}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Remembered fact: user.name = Yash.", result)
        self.assertNotIn("Current task context", result)
        self.assertNotIn("Relevant prior conversation", result)
        self.assertNotIn("Relevant how-to pattern", result)


class TestMainIntegration(unittest.TestCase):
    def _run_main(
        self,
        *,
        prompt="hello",
        session_id="test-sess",
        context=None,
        first_message=True,
    ):
        payload = json.dumps({"session_id": session_id, "prompt": prompt})
        stdout = io.StringIO()
        stderr = io.StringIO()
        conn = MagicMock()
        retrieval_context = context or WakeUpContext(None, None, [], [], [])

        exit_code = None
        with patch("sys.stdin", io.StringIO(payload)), \
             patch("sys.stdout", stdout), \
             patch("sys.stderr", stderr), \
             patch("sys.exit", side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code))), \
             patch.object(_wu, "DB_PATH", "/fake/test.db"), \
             patch("os.path.exists", return_value=True), \
             patch.object(_wu, "open_db", return_value=conn), \
             patch.object(_wu, "_is_first_message", return_value=first_message), \
             patch.object(_wu, "retrieve_wake_up_context", return_value=retrieval_context) as retrieve_context, \
             patch.object(_wu, "log_retrieval") as log_retrieval, \
             patch.object(_wu, "activity_log"):
            with self.assertRaises(SystemExit) as raised:
                main()
            exit_code = raised.exception.code

        return {
            "stdout": stdout.getvalue(),
            "stderr": stderr.getvalue(),
            "exit_code": exit_code,
            "conn": conn,
            "retrieve_context": retrieve_context,
            "log_retrieval": log_retrieval,
        }

    def test_no_fact_found_short_circuits_with_error(self):
        result = self._run_main(prompt="hello", first_message=False)
        self.assertEqual(result["stdout"], "")
        error_payload = json.loads(result["stderr"])
        self.assertEqual(error_payload["error"]["message"], "No fact found.")
        self.assertEqual(result["exit_code"], 2)
        result["conn"].close.assert_called_once()

    def test_fact_hit_short_circuits_with_fact_response(self):
        result = self._run_main(
            prompt="what is my name?",
            context=WakeUpContext(
                None,
                None,
                [],
                [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
                [],
            ),
            first_message=False,
        )
        self.assertEqual(result["stdout"], "")
        error_payload = json.loads(result["stderr"])
        self.assertEqual(error_payload["error"]["message"], "user.name = Yash")
        self.assertEqual(result["exit_code"], 2)
        result["retrieve_context"].assert_called_once_with(
            result["conn"],
            "what is my name?",
            include_working_memory=False,
        )
        result["log_retrieval"].assert_called_once_with(
            result["conn"],
            "wake_up_fact_hit",
            "what is my name?",
            len("user.name = Yash") // 4,
        )

    def test_retrieval_is_called_with_first_message_flag(self):
        first = self._run_main(prompt="hello there", first_message=True)
        first["retrieve_context"].assert_called_once_with(
            first["conn"],
            "hello there",
            include_working_memory=True,
        )

        later = self._run_main(prompt="hello there", first_message=False)
        later["retrieve_context"].assert_called_once_with(
            later["conn"],
            "hello there",
            include_working_memory=False,
        )


if __name__ == "__main__":
    unittest.main()
