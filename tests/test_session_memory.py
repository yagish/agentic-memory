import unittest

from memory.inference import GenerationResult
from memory.session import (
    build_session_memory_extraction_prompt,
    extract_session_memory_from_session_text,
    parse_extracted_session_memory,
    session_memory_to_semantic_core,
)


class TestSessionMemoryExtractionHelpers(unittest.TestCase):
    def test_extract_session_memory_parses_wrapped_json_and_normalizes_lists(self):
        def fake_generate(_request):
            return GenerationResult(
                text='''Session summary:\n{\n  "title": " Auth middleware refactor ",\n  "summary": " Moved token validation into shared middleware and fixed the login redirect loop locally. ",\n  "what_was_tried": [" Move token validation into shared middleware ", "Move token validation into shared middleware", "Retest the login loop locally"],\n  "outcomes": [" Redirect loop stopped reproducing locally ", "Redirect loop stopped reproducing locally"],\n  "left_off_at": " Regression coverage is still missing for refresh-token and expired-session flows ",\n  "next_steps": [" Add regression tests ", "Add regression tests", "Verify expired-session flow"],\n  "confidence": 0.95\n}\n''',
                model="fake-model",
            )

        memory = extract_session_memory_from_session_text(
            "User: We moved token validation into shared auth middleware.\nAssistant: The redirect loop stopped reproducing locally.\nUser: Next add regression tests for refresh-token and expired-session flows.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertEqual(
            session_memory_to_semantic_core(memory),
            {
                "title": "Auth middleware refactor",
                "summary": "Moved token validation into shared middleware and fixed the login redirect loop locally.",
                "what_was_tried": ["Move token validation into shared middleware", "Retest the login loop locally"],
                "outcomes": ["Redirect loop stopped reproducing locally"],
                "left_off_at": "Regression coverage is still missing for refresh-token and expired-session flows",
                "next_steps": ["Add regression tests", "Verify expired-session flow"],
            },
        )
        self.assertEqual(memory.confidence, 0.95)

    def test_empty_json_means_no_session_memory(self):
        self.assertIsNone(parse_extracted_session_memory("{}"))

    def test_extract_session_memory_returns_none_when_model_finds_no_resumable_work(self):
        def fake_generate(_request):
            return GenerationResult(text="{}", model="fake-model")

        memory = extract_session_memory_from_session_text(
            "User: Thanks for the explanation of embeddings.\nAssistant: Happy to help.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertIsNone(memory)

    def test_prompt_includes_transcript_and_output_shape(self):
        prompt = build_session_memory_extraction_prompt(
            "User: Finish the auth middleware refactor.\nAssistant: Next add regression tests."
        )

        self.assertIn("User: Finish the auth middleware refactor.", prompt)
        self.assertIn('"left_off_at": "where the work currently stands"', prompt)
        self.assertIn('"next_steps": ["useful next session step"]', prompt)
        self.assertIn("Return only the JSON object.", prompt)

    def test_prompt_pushes_resume_handoff_not_durable_fact_or_procedure(self):
        prompt = build_session_memory_extraction_prompt(
            "User: We still need BILLING_DB_URL before we can rerun staging validation."
        )

        self.assertIn("concise handoff for the overall session", prompt)
        self.assertIn("It is not for:", prompt)
        self.assertIn("durable repeatable procedures", prompt)
        self.assertIn("Preserve exact literals when central", prompt)


if __name__ == "__main__":
    unittest.main()
