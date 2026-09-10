import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.facts.renderer import (
    _canonical_fallback,
    _normalize_rendered_answer,
    build_fact_render_prompt,
    render_fact_answer,
)


class TestFactRenderer(unittest.TestCase):
    def test_build_prompt_includes_question_and_facts(self):
        prompt = build_fact_render_prompt(
            user_prompt="what is my name and where do i live?",
            fact_contents=["user.name = Yagish", "user.location = West Chester, OH"],
        )
        self.assertIn("Question: what is my name and where do i live?", prompt)
        self.assertIn("Facts:\n- user.name = Yagish\n- user.location = West Chester, OH", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_normalize_renderer_output_strips_quotes_and_code_fences(self):
        self.assertEqual(_normalize_rendered_answer('"Your name is Yagish."'), "Your name is Yagish.")
        self.assertEqual(_normalize_rendered_answer("```\nYour name is Yagish.\n```"), "Your name is Yagish.")
        self.assertEqual(_normalize_rendered_answer("Your name is Yagish.\nYou live in Ohio."), "Your name is Yagish. You live in Ohio.")

    def test_render_fact_answer_renders_name_deterministically(self):
        result = render_fact_answer("what is my name?", ["user.name = Yagish"])
        self.assertEqual(result, "Your name is Yagish.")

    def test_render_fact_answer_renders_multiple_matched_facts(self):
        result = render_fact_answer(
            "what is my name and where do i live?",
            ["user.name = Yagish", "user.location = West Chester, OH"],
        )
        self.assertEqual(result, "Your name is Yagish. You live in West Chester, OH.")

    def test_render_fact_answer_returns_empty_when_prompt_does_not_match_facts(self):
        result = render_fact_answer("how do i deploy this service?", ["user.name = Yagish"])
        self.assertEqual(result, "")

    def test_render_fact_answer_returns_empty_for_non_canonical_fact_strings(self):
        result = render_fact_answer("what is my name?", ["My name is Yagish. What is my name? Yagish."])
        self.assertEqual(result, "")

    def test_canonical_fallback_joins_multiple_facts(self):
        self.assertEqual(
            _canonical_fallback(["user.name = Yagish", "user.location = West Chester, OH"]),
            "user.name = Yagish\nuser.location = West Chester, OH",
        )

    def test_render_fact_answer_stays_within_default_length_budget(self):
        result = render_fact_answer(
            "what languages do i prefer?",
            [
                "user.favorite_language = Python",
                "user.preferred_language = TypeScript",
                "user.preferred_language = Rust",
                "user.preferred_language = Go",
                "user.preferred_language = Kotlin",
            ],
        )

        self.assertTrue(result)
        self.assertLessEqual(len(result), 240)


if __name__ == "__main__":
    unittest.main()
