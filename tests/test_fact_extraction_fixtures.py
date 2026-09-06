import json
import os
import unittest
from pathlib import Path

from memory.facts import (
    discover_fact_fixture_cases,
    extract_facts_from_session_text,
    facts_to_semantic_core,
)
from memory.ollama import is_ollama_running


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "facts"


class TestFactExtractionFixtureInventory(unittest.TestCase):
    def test_fixture_pairs_are_present_and_growing(self):
        cases = discover_fact_fixture_cases(_FIXTURES_ROOT)
        self.assertGreaterEqual(len(cases), 50)

        sessions = {path.stem for path in (_FIXTURES_ROOT / "sessions").glob("*.txt")}
        expected = {path.stem for path in (_FIXTURES_ROOT / "expected").glob("*.json")}
        self.assertEqual(sessions, expected)


@unittest.skipUnless(
    os.environ.get("MEMORY_RUN_OLLAMA_TESTS") == "1" and is_ollama_running(),
    "Set MEMORY_RUN_OLLAMA_TESTS=1 and ensure Ollama is running for fact extraction fixtures",
)
class TestFactExtractionFixtures(unittest.TestCase):
    # This is the real acceptance test for fact extraction:
    # session fixture text -> real Qwen call -> extracted facts -> expected JSON.
    def test_session_fixtures_extract_expected_facts(self):
        cases = discover_fact_fixture_cases(_FIXTURES_ROOT)
        self.assertGreaterEqual(len(cases), 50)

        for case_name in cases:
            # Run each fixture as a subtest so one failure shows the exact case name.
            with self.subTest(case=case_name):
                session_text = (_FIXTURES_ROOT / "sessions" / f"{case_name}.txt").read_text()
                expected = json.loads(
                    (_FIXTURES_ROOT / "expected" / f"{case_name}.json").read_text()
                )

                facts = extract_facts_from_session_text(session_text)

                self.assertEqual(
                    facts_to_semantic_core(facts),
                    facts_to_semantic_core(expected),
                )


if __name__ == "__main__":
    unittest.main()
