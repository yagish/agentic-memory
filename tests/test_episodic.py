import unittest

from memory.episodic import (
    build_episodic_extraction_prompt,
    episode_to_semantic_core,
    extract_episode_from_session_text,
)
from memory.inference import GenerationResult


class TestEpisodicExtractionHelpers(unittest.TestCase):
    def test_extract_episode_parses_wrapped_json_and_normalizes_lists(self):
        def fake_generate(_request):
            return GenerationResult(
                text='''Here is the episode:\n{\n  "title": " Auth middleware decision ",\n  "abstract": " Moved token validation into shared middleware. Login loop is fixed. ",\n  "participants": [" Yash ", "Dana", "Yash"],\n  "decisions": [" Move token validation into shared middleware "],\n  "outcomes": [" Login loop fixed ", "Login loop fixed"],\n  "follow_ups": [" Add regression tests "],\n  "confidence": 0.91\n}\n''',
                model="fake-model",
            )

        episode = extract_episode_from_session_text(
            "User: We decided to move token validation into shared middleware.\nAssistant: Done. The login loop is fixed.\nUser: Add regression tests next.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertEqual(
            episode_to_semantic_core(episode),
            {
                "title": "Auth middleware decision",
                "abstract": "Moved token validation into shared middleware. Login loop is fixed.",
                "participants": ["Yash", "Dana"],
                "decisions": ["Move token validation into shared middleware"],
                "outcomes": ["Login loop fixed"],
                "follow_ups": ["Add regression tests"],
            },
        )
        self.assertEqual(episode.confidence, 0.91)

    def test_prompt_includes_both_user_and_assistant_turns(self):
        prompt = build_episodic_extraction_prompt(
            "User: We moved token validation into middleware.\nAssistant: I implemented it and fixed the loop."
        )

        self.assertIn("User: We moved token validation into middleware.", prompt)
        self.assertIn("Assistant: I implemented it and fixed the loop.", prompt)
        self.assertIn("Return only the JSON object.", prompt)

    def test_prompt_pushes_concrete_literals_and_named_people(self):
        prompt = build_episodic_extraction_prompt(
            "User: Jenna will own the postmortem.\nAssistant: The root cause was STRIPE_WEBHOOK_SECRET."
        )

        self.assertIn("preserve important literal details", prompt)
        self.assertIn("names, branch names, env vars, flags, file names, model names, percentages, dates, and numeric thresholds", prompt)
        self.assertIn("include named people when they materially participated", prompt)
        self.assertIn("Never copy values from examples unless they appear in the transcript.", prompt)
        self.assertIn('"participants": ["Jenna"]', prompt)
        self.assertIn("STRIPE_WEBHOOK_SECRET", prompt)


if __name__ == "__main__":
    unittest.main()
