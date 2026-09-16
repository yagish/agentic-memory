import json
import os
import unittest
from pathlib import Path

from memory.llm.ollama import is_ollama_running
from memory.working_memory import (
    discover_working_memory_fixture_cases,
    extract_working_memory_from_session_text,
    normalized_contains,
)


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "working_memory"


class TestWorkingMemoryFixtureInventory(unittest.TestCase):
    def test_fixture_inventory_is_complete(self):
        sessions = {path.stem for path in (_FIXTURES_ROOT / "sessions").glob("*.txt")}
        expected = {path.stem for path in (_FIXTURES_ROOT / "expected").glob("*.json")}
        self.assertEqual(sessions, expected)
        self.assertGreaterEqual(len(sessions), 100)


@unittest.skipUnless(
    os.environ.get("MEMORY_RUN_OLLAMA_TESTS") == "1" and is_ollama_running(),
    "Set MEMORY_RUN_OLLAMA_TESTS=1 and ensure Ollama is running for working-memory extraction fixtures",
)
class TestWorkingMemoryExtractionFixtures(unittest.TestCase):
    def test_session_fixtures_extract_expected_working_memory_signals(self):
        cases = discover_working_memory_fixture_cases(_FIXTURES_ROOT)
        self.assertGreaterEqual(len(cases), 100)

        for case_name in cases:
            with self.subTest(case=case_name):
                session_text = (_FIXTURES_ROOT / "sessions" / f"{case_name}.txt").read_text()
                expected = json.loads(
                    (_FIXTURES_ROOT / "expected" / f"{case_name}.json").read_text()
                )

                working_memory = extract_working_memory_from_session_text(
                    session_text,
                    source="fixture-harness",
                )

                if expected.get("expect_none"):
                    self.assertIsNone(working_memory)
                    continue

                self.assertIsNotNone(working_memory)
                for snippet in expected.get("current_goal_contains", []):
                    self.assertTrue(normalized_contains(working_memory.current_goal, snippet))
                for snippet in expected.get("current_focus_contains", []):
                    self.assertTrue(normalized_contains(working_memory.current_focus, snippet))
                for snippet in expected.get("active_tasks_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in working_memory.active_tasks),
                        msg=f"missing active-task snippet: {snippet}",
                    )
                for snippet in expected.get("constraints_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in working_memory.constraints),
                        msg=f"missing constraint snippet: {snippet}",
                    )
                for snippet in expected.get("next_step_contains", []):
                    self.assertTrue(normalized_contains(working_memory.next_step, snippet))
                if expected.get("status"):
                    self.assertEqual(working_memory.status, expected["status"])


if __name__ == "__main__":
    unittest.main()
