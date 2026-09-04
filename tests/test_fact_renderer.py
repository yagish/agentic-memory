# test_fact_renderer.py — tests for LLM-backed fact answer rendering.

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.fact_renderer import (
    _canonical_fallback,
    _normalize_rendered_answer,
    build_fact_render_prompt,
    render_fact_answer,
)
from memory.inference import GenerationResult, InferenceError


class TestFactRenderer(unittest.TestCase):
    def test_build_prompt_includes_question_and_facts(self):
        prompt = build_fact_render_prompt(
            user_prompt="what is my name and where do i live?",
            fact_contents=["user.name = Yagish", "user.location = West Chester, OH"],
        )
        self.assertIn("Question: what is my name and where do i live?", prompt)
        self.assertIn("Facts:\n- user.name = Yagish\n- user.location = West Chester, OH", prompt)
        self.assertIn("Answer only what the user asked.", prompt)
        self.assertIn("If the question has multiple parts, answer every part supported by the facts.", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

    def test_normalize_renderer_output_strips_quotes_and_code_fences(self):
        self.assertEqual(_normalize_rendered_answer('"Your name is Yagish."'), "Your name is Yagish.")
        self.assertEqual(_normalize_rendered_answer("```\nYour name is Yagish.\n```"), "Your name is Yagish.")

    def test_render_fact_answer_uses_local_llm_text(self):
        with patch(
            "memory.fact_renderer.generate_text",
            return_value=GenerationResult(text="Your name is Yagish.\n", model="qwen2.5:3b"),
        ) as generate_text:
            result = render_fact_answer("what is my name?", ["user.name = Yagish"])

        self.assertEqual(result, "Your name is Yagish.")
        request = generate_text.call_args.args[0]
        self.assertEqual(request.temperature, 0.0)
        self.assertIn("what is my name?", request.prompt)
        self.assertIn("- user.name = Yagish", request.prompt)

    def test_render_fact_answer_falls_back_to_canonical_fact_on_failure(self):
        with patch(
            "memory.fact_renderer.generate_text",
            side_effect=InferenceError("ollama unavailable"),
        ):
            result = render_fact_answer("what is my name?", ["user.name = Yagish"])

        self.assertEqual(result, "user.name = Yagish")

    def test_render_fact_answer_falls_back_when_renderer_returns_blank(self):
        with patch(
            "memory.fact_renderer.generate_text",
            return_value=GenerationResult(text="   ", model="qwen2.5:3b"),
        ):
            result = render_fact_answer("what is my name?", ["user.name = Yagish"])

        self.assertEqual(result, "user.name = Yagish")

    def test_render_fact_answer_keeps_model_output_when_non_blank(self):
        with patch(
            "memory.fact_renderer.generate_text",
            return_value=GenerationResult(text="My name is Yagish.", model="qwen2.5:3b"),
        ):
            result = render_fact_answer("whats my name", ["user.name = Yagish"])

        self.assertEqual(result, "My name is Yagish.")

    def test_canonical_fallback_joins_multiple_facts(self):
        self.assertEqual(
            _canonical_fallback(["user.name = Yagish", "user.location = West Chester, OH"]),
            "user.name = Yagish\nuser.location = West Chester, OH",
        )

    def test_render_fact_answer_passes_all_facts_to_model(self):
        with patch(
            "memory.fact_renderer.generate_text",
            return_value=GenerationResult(
                text="Your name is Yagish and you live in West Chester, OH.",
                model="qwen2.5:3b",
            ),
        ) as generate_text:
            result = render_fact_answer(
                "whats my name and where do i live",
                ["user.name = Yagish", "user.location = West Chester, OH"],
            )

        self.assertEqual(result, "Your name is Yagish and you live in West Chester, OH.")
        request = generate_text.call_args.args[0]
        self.assertIn("- user.name = Yagish", request.prompt)
        self.assertIn("- user.location = West Chester, OH", request.prompt)

    def test_render_fact_answer_passes_all_provided_facts_to_model(self):
        with patch(
            "memory.fact_renderer.generate_text",
            return_value=GenerationResult(
                text="Your name is Yagish and you live in West Chester, OH.",
                model="qwen2.5:3b",
            ),
        ) as generate_text:
            render_fact_answer(
                "whats my name and where do i live",
                [
                    "user.name = Yagish",
                    "user.location = West Chester, OH",
                    "user.timezone = EST",
                ],
            )

        prompt = generate_text.call_args.args[0].prompt
        self.assertIn("- user.name = Yagish", prompt)
        self.assertIn("- user.location = West Chester, OH", prompt)
        self.assertIn("- user.timezone = EST", prompt)


if __name__ == "__main__":
    unittest.main()
