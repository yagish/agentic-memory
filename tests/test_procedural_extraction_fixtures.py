import json
import os
import unittest
from pathlib import Path

from memory.ollama import is_ollama_running
from memory.procedural import (
    discover_procedural_fixture_cases,
    extract_procedure_from_session_text,
    normalized_contains,
)


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "procedural"


class TestProceduralFixtureInventory(unittest.TestCase):
    def test_fixture_inventory_is_complete(self):
        sessions = {path.stem for path in (_FIXTURES_ROOT / "sessions").glob("*.txt")}
        expected = {path.stem for path in (_FIXTURES_ROOT / "expected").glob("*.json")}
        self.assertEqual(sessions, expected)
        self.assertGreaterEqual(len(sessions), 100)


@unittest.skipUnless(
    os.environ.get("MEMORY_RUN_OLLAMA_TESTS") == "1" and is_ollama_running(),
    "Set MEMORY_RUN_OLLAMA_TESTS=1 and ensure Ollama is running for procedural extraction fixtures",
)
class TestProceduralExtractionFixtures(unittest.TestCase):
    def test_session_fixtures_extract_expected_procedural_signals(self):
        cases = discover_procedural_fixture_cases(_FIXTURES_ROOT)
        self.assertGreaterEqual(len(cases), 100)

        for case_name in cases:
            with self.subTest(case=case_name):
                session_text = (_FIXTURES_ROOT / "sessions" / f"{case_name}.txt").read_text()
                expected = json.loads(
                    (_FIXTURES_ROOT / "expected" / f"{case_name}.json").read_text()
                )

                procedure = extract_procedure_from_session_text(
                    session_text,
                    source="fixture-harness",
                )

                if expected.get("expect_none"):
                    self.assertIsNone(procedure)
                    continue

                self.assertIsNotNone(procedure)
                for snippet in expected.get("title_contains", []):
                    self.assertTrue(normalized_contains(procedure.title, snippet))
                for snippet in expected.get("summary_contains", []):
                    self.assertTrue(normalized_contains(procedure.summary, snippet))
                for snippet in expected.get("steps_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in procedure.steps),
                        msg=f"missing step snippet: {snippet}",
                    )
                for snippet in expected.get("trigger_phrases_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in procedure.trigger_phrases),
                        msg=f"missing trigger snippet: {snippet}",
                    )
                for snippet in expected.get("tools_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in procedure.tools),
                        msg=f"missing tool snippet: {snippet}",
                    )


if __name__ == "__main__":
    unittest.main()
