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

    def test_prompt_keeps_user_lines_and_includes_tool_trace_antiexample(self):
        prompt = build_fact_extraction_prompt(
            "User: <bash-input>pwd</bash-input>\nUser: Python is my favorite language."
        )

        transcript = prompt.split("Transcript:\n", 1)[1]
        self.assertIn("User: <bash-input>pwd</bash-input>", transcript)
        self.assertIn("User: Python is my favorite language.", transcript)
        self.assertIn("Ignore tool echo lines such as <bash-input>pwd</bash-input>", prompt)

    def test_prompt_biases_toward_stable_user_facts_and_empty_output_when_unsure(self):
        prompt = build_fact_extraction_prompt(
            "User: Remove ObsoleteTables from db.py.\nUser: Use --once as a force flag."
        )

        self.assertIn("Most valid facts are about the user", prompt)
        self.assertIn("Every fact must include: entity, attribute, value, source_quote.", prompt)
        self.assertIn("Never copy values from examples into the output.", prompt)
        self.assertIn("If the transcript is mostly current-session work instructions, return [].", prompt)
        self.assertIn("If unsure whether something is a durable fact, return [].", prompt)
        self.assertIn("User: Remove ObsoleteTables from db.py.", prompt)
        self.assertIn("User: Use --once as a force flag.", prompt)
        self.assertIn("Output:\n[]", prompt)

    def test_prompt_includes_language_preference_examples(self):
        prompt = build_fact_extraction_prompt("User: Python is my favorite language.")

        self.assertIn('"attribute": "favorite_language"', prompt)
        self.assertIn('"attribute": "preferred_language"', prompt)

    def test_shell_command_noise_is_filtered_out_of_extracted_facts(self):
        def fake_generate(_request):
            return GenerationResult(
                text="""[
  {\"entity\": \"shell\", \"attribute\": \"command\", \"value\": \"pwd\", \"source_quote\": \"<bash-input>pwd</bash-input>\"},
  {\"entity\": \"user\", \"attribute\": \"name\", \"value\": \"Yash\", \"source_quote\": \"My name is Yash.\"}
]""",
                model="fake-model",
            )

        facts = extract_facts_from_session_text(
            "User: <bash-input>pwd</bash-input>\nUser: My name is Yash.",
            generate_fn=fake_generate,
        )

        self.assertEqual(
            facts_to_semantic_core(facts),
            [{"entity": "user", "attribute": "name", "value": "Yash"}],
        )

    def test_open_ontology_keeps_non_machine_fact_shapes(self):
        def fake_generate(_request):
            return GenerationResult(
                text="""[
  {\"entity\": \"user\", \"attribute\": \"current_task\", \"value\": \"Fix auth bug\", \"source_quote\": \"Fix auth bug\"},
  {\"entity\": \"repo\", \"attribute\": \"name\", \"value\": \"agentic-memory\", \"source_quote\": \"The repo name is agentic-memory.\"}
]""",
                model="fake-model",
            )

        facts = extract_facts_from_session_text(
            "User: Fix auth bug\nUser: The repo name is agentic-memory.",
            generate_fn=fake_generate,
        )

        self.assertEqual(
            facts_to_semantic_core(facts),
            [
                {"entity": "user", "attribute": "current_task", "value": "Fix auth bug"},
                {"entity": "repo", "attribute": "name", "value": "agentic-memory"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
