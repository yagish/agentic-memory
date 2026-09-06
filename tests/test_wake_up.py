# test_wake_up.py — tests for the wake-up hook adapter.

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.claude.wake_up import _build_injection, main
from memory.retrieval import WakeUpContext
import integrations.claude.wake_up as _wu


class TestBuildInjection(unittest.TestCase):
    def test_returns_empty_when_no_sections_exist(self):
        result = _build_injection(WakeUpContext(None, None, [], [], [], []))
        self.assertEqual(result, "")

    def test_includes_episodic_fact_and_procedural_sections(self):
        result = _build_injection(
            WakeUpContext(
                {"id": "cs-1", "similarity": 0.98, "content": "Task: Fix auth middleware"},
                {"id": "wm-1", "summary": "Current task is cleaning up auth middleware", "similarity": 0.88},
                [{"id": "cs-2", "similarity": 0.83, "content": "Task: Add JWT refresh flow"}],
                [{"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.79}],
                [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.97}],
                [{"id": "proc-1", "title": "Deploy workflow", "summary": "Use this when deploying auth.", "steps": ["Build", "Ship"], "similarity": 0.86}],
            )
        )
        self.assertTrue(result.startswith("[Memory context: "))
        self.assertIn("Recent related episode: Resolved auth bug. Fixed the login loop.", result)
        self.assertIn("Relevant how-to pattern: Deploy workflow. Use this when deploying auth.", result)
        self.assertIn("Steps: Build; Ship.", result)
        self.assertIn("Remembered fact: user.name = Yash.", result)
        self.assertNotIn("Relevant prior session", result)
        self.assertNotIn("Current task context", result)
        self.assertNotIn("Related prior session", result)


class TestMainIntegration(unittest.TestCase):
    def _run_main(
        self,
        *,
        prompt="hello",
        session_id="test-sess",
        server_response=None,
        first_message=True,
    ):
        payload = json.dumps({"session_id": session_id, "prompt": prompt})
        stdout = io.StringIO()
        stderr = io.StringIO()
        conn = MagicMock()
        response = server_response or {
            "action": "noop",
            "warnings": [],
            "context": {"facts": [], "episodic": [], "procedural": []},
        }

        exit_code = None
        with patch("sys.stdin", io.StringIO(payload)), \
             patch("sys.stdout", stdout), \
             patch("sys.stderr", stderr), \
             patch("sys.exit", side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code))), \
             patch.object(_wu, "DB_PATH", "/fake/test.db"), \
             patch.object(_wu, "_open_connection", return_value=conn), \
             patch.object(_wu, "_is_first_message", return_value=first_message), \
             patch.object(_wu, "_recall_via_server", return_value=response) as recall_via_server, \
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
            "recall_via_server": recall_via_server,
            "log_retrieval": log_retrieval,
        }

    def test_no_memory_found_allows_prompt(self):
        result = self._run_main(prompt="hello", first_message=False)
        self.assertEqual(result["stderr"], "")
        self.assertEqual(json.loads(result["stdout"]), {})
        self.assertEqual(result["exit_code"], 0)
        result["conn"].close.assert_called_once()

    def test_fact_hit_returns_documented_block_json_when_only_facts_match(self):
        result = self._run_main(
            prompt="what is my name?",
            server_response={
                "action": "answer",
                "answer": "Your name is Yash.",
                "warnings": [],
                "context": {
                    "facts": [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
                    "episodic": [],
                    "procedural": [],
                },
            },
            first_message=False,
        )
        self.assertEqual(result["stderr"], "")
        payload = json.loads(result["stdout"])
        self.assertEqual(
            payload,
            {
                "decision": "block",
                "reason": "Your name is Yash.",
                "suppressOriginalPrompt": True,
            },
        )
        self.assertEqual(result["exit_code"], 0)
        result["recall_via_server"].assert_called_once()
        result["log_retrieval"].assert_called_once_with(
            result["conn"],
            "wake_up_fact_hit",
            "what is my name?",
            len("Your name is Yash.") // 4,
        )

    def test_fact_hit_with_episodic_context_allows_prompt(self):
        result = self._run_main(
            prompt="continue fixing auth",
            server_response={
                "action": "inject",
                "injection": "[Memory context: foo]",
                "warnings": [],
                "context": {
                    "facts": [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
                    "episodic": [{"id": "ep-1", "title": "Resolved auth bug", "abstract": "Fixed the login loop.", "similarity": 0.79}],
                    "procedural": [],
                },
            },
            first_message=False,
        )

        self.assertEqual(json.loads(result["stdout"]), {})
        result["log_retrieval"].assert_called_once_with(
            result["conn"],
            "wake_up_context_found",
            "continue fixing auth",
            unittest.mock.ANY,
        )

    def test_logs_fact_lookup_query_results_and_renderer_io(self):
        payload = json.dumps({"session_id": "test-sess", "prompt": "what is my name and where do i live?"})
        stdout = io.StringIO()
        stderr = io.StringIO()
        conn = MagicMock()

        with patch("sys.stdin", io.StringIO(payload)), \
             patch("sys.stdout", stdout), \
             patch("sys.stderr", stderr), \
             patch("sys.exit", side_effect=lambda code=0: (_ for _ in ()).throw(SystemExit(code))), \
             patch.object(_wu, "DB_PATH", "/fake/test.db"), \
             patch.object(_wu, "_open_connection", return_value=conn), \
             patch.object(_wu, "_is_first_message", return_value=False), \
             patch.object(_wu, "_recall_via_server", return_value={
                 "action": "answer",
                 "answer": "Your name is Yash and you live in Seattle.",
                 "warnings": [],
                 "context": {
                     "facts": [
                         {"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99},
                         {"id": "fact-2", "content": "user.location = Seattle", "similarity": 0.97},
                     ],
                     "episodic": [],
                     "procedural": [],
                 },
             }), \
             patch.object(_wu, "log_retrieval"), \
             patch.object(_wu, "activity_log"), \
             patch.object(_wu, "_log_info") as log_info:
            with self.assertRaises(SystemExit):
                main()

        logged_messages = [call.args[0] for call in log_info.call_args_list]
        self.assertTrue(any('fact lookup results=[{"content": "user.name = Yash"' in msg for msg in logged_messages))
        self.assertTrue(any('fact renderer input=["user.name = Yash", "user.location = Seattle"]' in msg for msg in logged_messages))
        self.assertTrue(any("fact renderer output='Your name is Yash and you live in Seattle.'" in msg for msg in logged_messages))

    def test_recall_server_is_called_with_first_message_flag(self):
        first = self._run_main(prompt="hello there", first_message=True)
        self.assertEqual(
            first["recall_via_server"].call_args.kwargs,
            {"include_working_memory": True},
        )

        later = self._run_main(prompt="hello there", first_message=False)
        self.assertEqual(
            later["recall_via_server"].call_args.kwargs,
            {"include_working_memory": False},
        )


if __name__ == "__main__":
    unittest.main()
