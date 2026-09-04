import unittest

from memory.facts import (
    build_fact_extraction_prompt,
    extract_facts_from_session_text,
    facts_to_semantic_core,
)
from memory.inference import GenerationResult


class TestFactExtractionHelpers(unittest.TestCase):
    # These tests avoid the real LLM. They explain/test the local helper logic:
    # JSON parsing, normalization, dedupe, metadata merging, and prompt shaping.
    def test_extract_facts_parses_wrapped_json_and_merges_duplicate_metadata(self):
        def fake_generate(_request):
            # Simulate a model response that wraps JSON in prose and repeats a fact.
            return GenerationResult(
                text="""Here are the facts I found:\n[
  {\"entity\": \"User\", \"attribute\": \"Name\", \"value\": \" Yash \", \"confidence\": 0.41},
  {\"entity\": \"user\", \"attribute\": \"name\", \"value\": \"Yash\", \"confidence\": 0.92, \"source_quote\": \"My name is Yash.\"},
  {\"entity\": \"user\", \"attribute\": \"timezone\", \"value\": \"PST\", \"evidence\": \"My timezone is PST.\"}
]\n""",
                model="fake-model",
            )

        facts = extract_facts_from_session_text(
            "User: My name is Yash.\nUser: My timezone is PST.",
            generate_fn=fake_generate,
        )

        self.assertEqual(
            facts_to_semantic_core(facts),
            [
                {"entity": "user", "attribute": "name", "value": "Yash"},
                {"entity": "user", "attribute": "timezone", "value": "PST"},
            ],
        )
        self.assertEqual(facts[0].confidence, 0.92)
        self.assertEqual(facts[0].source_quote, "My name is Yash.")
        self.assertEqual(facts[1].evidence, "My timezone is PST.")

    def test_prompt_includes_only_user_lines_when_speaker_prefixes_exist(self):
        # The extractor prompt should never include assistant turns.
        prompt = build_fact_extraction_prompt(
            "User: My name is Yash.\nAssistant: Your timezone is CET.\nUser: My shell is zsh."
        )

        self.assertIn("User: My name is Yash.", prompt)
        self.assertIn("User: My shell is zsh.", prompt)
        self.assertNotIn("Assistant: Your timezone is CET.", prompt)


if __name__ == "__main__":
    unittest.main()
