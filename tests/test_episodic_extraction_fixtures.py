import json
import os
import unittest
from pathlib import Path

from memory.episodic import (
    discover_episodic_fixture_cases,
    extract_episode_from_session_text,
    normalized_contains,
)
from memory.ollama import is_ollama_running


_FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "episodic"


@unittest.skipUnless(
    os.environ.get("MEMORY_RUN_OLLAMA_TESTS") == "1" and is_ollama_running(),
    "Set MEMORY_RUN_OLLAMA_TESTS=1 and ensure Ollama is running for episodic extraction fixtures",
)
class TestEpisodicExtractionFixtures(unittest.TestCase):
    def test_session_fixtures_extract_expected_episode_signals(self):
        cases = discover_episodic_fixture_cases(_FIXTURES_ROOT)
        self.assertGreaterEqual(len(cases), 2)

        for case_name in cases:
            with self.subTest(case=case_name):
                session_text = (_FIXTURES_ROOT / "sessions" / f"{case_name}.txt").read_text()
                expected = json.loads(
                    (_FIXTURES_ROOT / "expected" / f"{case_name}.json").read_text()
                )

                episode = extract_episode_from_session_text(session_text, source="fixture-harness")

                for snippet in expected.get("title_contains", []):
                    self.assertTrue(normalized_contains(episode.title, snippet))
                for snippet in expected.get("abstract_contains", []):
                    self.assertTrue(normalized_contains(episode.abstract, snippet))
                for snippet in expected.get("decisions_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in episode.decisions),
                        msg=f"missing decision snippet: {snippet}",
                    )
                for snippet in expected.get("outcomes_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in episode.outcomes),
                        msg=f"missing outcome snippet: {snippet}",
                    )
                for snippet in expected.get("follow_ups_contains", []):
                    self.assertTrue(
                        any(normalized_contains(item, snippet) for item in episode.follow_ups),
                        msg=f"missing follow-up snippet: {snippet}",
                    )


if __name__ == "__main__":
    unittest.main()
