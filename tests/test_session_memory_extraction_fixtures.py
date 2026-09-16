import json
import os
import unittest
from pathlib import Path

from memory.llm.ollama import is_ollama_running
from memory.session import (
    discover_session_memory_fixture_cases,
    extract_session_memory_from_session_text,
    normalized_contains,
)


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "session_memory"


class TestSessionMemoryFixtureInventory(unittest.TestCase):
    def test_fixture_inventory_is_complete(self):
        sessions = {path.stem for path in (_FIXTURES_ROOT / "sessions").glob("*.txt")}
        expected = {path.stem for path in (_FIXTURES_ROOT / "expected").glob("*.json")}
        self.assertEqual(sessions, expected)
        self.assertGreaterEqual(len(sessions), 100)


@unittest.skipUnless(
    os.environ.get("MEMORY_RUN_OLLAMA_TESTS") == "1" and is_ollama_running(),
    "Set MEMORY_RUN_OLLAMA_TESTS=1 and ensure Ollama is running for session-memory extraction fixtures",
)
class TestSessionMemoryExtractionFixtures(unittest.TestCase):
    def test_session_fixtures_extract_expected_session_memory_signals(self):
        cases = discover_session_memory_fixture_cases(_FIXTURES_ROOT)
        self.assertGreaterEqual(len(cases), 100)

        for case_name in cases:
            with self.subTest(case=case_name):
                session_text = (_FIXTURES_ROOT / "sessions" / f"{case_name}.txt").read_text()
                expected = json.loads(
                    (_FIXTURES_ROOT / "expected" / f"{case_name}.json").read_text()
                )

                session_memory = extract_session_memory_from_session_text(
                    session_text,
                    source="fixture-harness",
                )

                if expected.get("expect_none"):
                    self.assertIsNone(session_memory)
                    continue

                self.assertIsNotNone(session_memory)
                for snippet in expected.get("title_contains", []):
                    self.assertTrue(normalized_contains(session_memory.title, snippet))
                for snippet in expected.get("summary_contains", []):
                    self.assertTrue(normalized_contains(session_memory.summary, snippet))
                for snippet in expected.get("what_was_tried_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in session_memory.what_was_tried),
                        msg=f"missing tried snippet: {snippet}",
                    )
                for snippet in expected.get("outcomes_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in session_memory.outcomes),
                        msg=f"missing outcome snippet: {snippet}",
                    )
                for snippet in expected.get("left_off_at_contains", []):
                    self.assertTrue(normalized_contains(session_memory.left_off_at, snippet))
                for snippet in expected.get("next_steps_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in session_memory.next_steps),
                        msg=f"missing next-step snippet: {snippet}",
                    )


if __name__ == "__main__":
    unittest.main()
