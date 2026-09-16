import unittest

from memory.llm.inference import GenerationResult
from memory.working_memory import (
    build_working_memory_extraction_prompt,
    extract_working_memory_from_session_text,
    parse_extracted_working_memory,
    working_memory_to_semantic_core,
)


class TestWorkingMemoryExtractionHelpers(unittest.TestCase):
    def test_extract_working_memory_parses_wrapped_json_and_normalizes_lists(self):
        def fake_generate(_request):
            return GenerationResult(
                text='''Working context:\n{\n  "current_goal": " Finish auth middleware refactor ",\n  "current_focus": " Regression coverage for refresh-token flows ",\n  "active_tasks": [" Add refresh-token tests ", "Add refresh-token tests", "Add expired-session tests"],\n  "constraints": [" Keep branch fix/auth-middleware ", "Keep branch fix/auth-middleware"],\n  "next_step": " Write the refresh-token regression test ",\n  "status": " Ready To Resume ",\n  "confidence": 0.94\n}\n''',
                model="fake-model",
            )

        memory = extract_working_memory_from_session_text(
            "User: We moved token validation into shared auth middleware.\nUser: Next add regression tests for refresh-token and expired-session flows.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertEqual(
            working_memory_to_semantic_core(memory),
            {
                "current_goal": "Finish auth middleware refactor",
                "current_focus": "Regression coverage for refresh-token flows",
                "active_tasks": ["Add refresh-token tests", "Add expired-session tests"],
                "constraints": ["Keep branch fix/auth-middleware"],
                "next_step": "Write the refresh-token regression test",
                "status": "ready_to_resume",
            },
        )
        self.assertEqual(memory.confidence, 0.94)

    def test_empty_json_means_no_working_memory(self):
        self.assertIsNone(parse_extracted_working_memory("{}"))

    def test_extract_working_memory_returns_none_when_model_finds_no_active_context(self):
        def fake_generate(_request):
            return GenerationResult(text="{}", model="fake-model")

        memory = extract_working_memory_from_session_text(
            "User: We shipped the release and all smoke tests passed.",
            generate_fn=fake_generate,
            source="test-harness",
            session_id="session-123",
        )

        self.assertIsNone(memory)

    def test_prompt_includes_transcript_and_output_shape(self):
        prompt = build_working_memory_extraction_prompt(
            "User: Finish the auth middleware refactor.\nAssistant: Next add regression tests."
        )

        self.assertIn("User: Finish the auth middleware refactor.", prompt)
        self.assertIn('"current_goal": "short statement of the current objective"', prompt)
        self.assertIn('"status": "in_progress"', prompt)
        self.assertIn("Return only the JSON object.", prompt)

    def test_prompt_pushes_temporary_resume_context_not_durable_memory(self):
        prompt = build_working_memory_extraction_prompt(
            "User: We still need BILLING_DB_URL before we can rerun staging validation."
        )

        self.assertIn("current temporary active context", prompt)
        self.assertIn("It is not for:", prompt)
        self.assertIn("durable project procedures or runbooks", prompt)
        self.assertIn("Preserve exact literals when central", prompt)


if __name__ == "__main__":
    unittest.main()
