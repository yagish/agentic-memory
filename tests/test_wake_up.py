# test_wake_up.py — tests for the wake-up hook's core logic.
#
# Tests cover _build_injection() (the formatting function) and main()
# (the stdin/stdout entry point).
#
# Run with:  python3 -m unittest tests.test_wake_up -v

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hooks.wake_up import _build_injection, main
import hooks.wake_up as _wu


class TestBuildInjection(unittest.TestCase):
    """Tests for _build_injection() — the function that formats the memory block."""

    def test_identity_appears_before_cached_answer(self):
        # Identity must come first so it cannot be overridden by a cached answer.
        result = _build_injection(
            identity="Name: Alice",
            facts=[],
            insights=None,
            cached_answer={"question": "hi", "answer": "hello", "similarity": 1.0},
        )
        identity_pos = result.find("Name: Alice")
        cached_pos   = result.find("[Cached Answer")
        self.assertGreater(identity_pos, -1)
        self.assertGreater(cached_pos, -1)
        self.assertLess(identity_pos, cached_pos,
                        "Identity must appear before the cached answer block")

    def test_identity_section_labelled_authoritative(self):
        result = _build_injection("Name: Alice", [], None)
        self.assertIn("[Identity & Preferences — always authoritative]", result)

    def test_cached_answer_block_present_when_provided(self):
        ca = {"question": "whats my name", "answer": "Alice", "similarity": 0.97}
        result = _build_injection("Name: Alice", [], None, cached_answer=ca)
        self.assertIn("[Cached Answer", result)
        self.assertIn("Q: whats my name", result)
        self.assertIn("A: Alice", result)

    def test_cached_answer_instruction_mentions_identity_conflict(self):
        # The instruction must tell Claude to ignore the cached answer if it
        # contradicts the Identity section.
        ca = {"question": "whats my name", "answer": "I don't know", "similarity": 1.0}
        result = _build_injection("Name: Alice", [], None, cached_answer=ca)
        self.assertIn("contradicts Identity", result)
        self.assertIn("ignore it", result)

    def test_from_memory_instruction_present_with_cached_answer(self):
        ca = {"question": "q", "answer": "a", "similarity": 0.95}
        result = _build_injection("Name: Alice", [], None, cached_answer=ca)
        self.assertIn("[From Memory]", result)

    def test_from_memory_instruction_absent_without_cached_answer(self):
        result = _build_injection("Name: Alice", [], None, cached_answer=None)
        self.assertNotIn("[From Memory]", result)

    def test_no_cached_answer_section_when_none(self):
        result = _build_injection("Name: Alice", [], None, cached_answer=None)
        self.assertNotIn("[Cached Answer", result)

    def test_facts_section_present_when_provided(self):
        facts = [{"content": "User prefers dark mode", "tags": ["ui"]}]
        result = _build_injection("", facts, None)
        self.assertIn("[Relevant Facts about You]", result)
        self.assertIn("User prefers dark mode", result)
        self.assertIn("[tags: ui]", result)

    def test_facts_section_absent_when_empty(self):
        result = _build_injection("Name: Alice", [], None)
        self.assertNotIn("[Relevant Facts about You]", result)

    def test_insights_section_present_when_provided(self):
        insights = [{"content": "User often asks about auth", "insight_type": "pattern", "confidence": 0.8}]
        result = _build_injection("", [], insights)
        self.assertIn("[Learned Patterns]", result)
        self.assertIn("User often asks about auth", result)

    def test_insights_section_absent_when_none(self):
        result = _build_injection("Name: Alice", [], None)
        self.assertNotIn("[Learned Patterns]", result)

    def test_output_has_delimiters(self):
        result = _build_injection("", [], None)
        self.assertIn("=== MEMORY ===", result)
        self.assertIn("=== END MEMORY ===", result)

    def test_similarity_formatted_as_percentage(self):
        ca = {"question": "q", "answer": "a", "similarity": 0.96}
        result = _build_injection("", [], None, cached_answer=ca)
        self.assertIn("96%", result)

    def test_empty_identity_omits_identity_section(self):
        result = _build_injection("", [], None)
        self.assertNotIn("[Identity & Preferences", result)


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main() via mocked stdin and DB."""

    def _run_main(self, session_id="test-sess", prompt="hello",
                  identity="Name: Alice", facts=None, insights=None,
                  cached_answer=None):
        """Run main() with full mocking, return parsed stdout JSON."""
        payload = json.dumps({"session_id": session_id, "prompt": prompt})
        fake_db = "/fake/test.db"

        def fake_exists(path):
            return path == fake_db or path.startswith("/tmp/memory_insights_injected_")

        output = io.StringIO()
        with patch("sys.stdin",  io.StringIO(payload)), \
             patch("sys.stdout", output), \
             patch("sys.exit"), \
             patch.object(_wu, "DB_PATH", fake_db), \
             patch("os.path.exists", side_effect=fake_exists), \
             patch.object(_wu, "init_db", return_value=MagicMock()), \
             patch.object(_wu, "_read_identity", return_value=identity), \
             patch.object(_wu, "find_direct_answer", return_value=cached_answer), \
             patch.object(_wu, "_fetch_relevant_facts", return_value=facts or []), \
             patch.object(_wu, "list_insights", return_value=insights or []), \
             patch.object(_wu, "log_retrieval"), \
             patch.object(_wu, "activity_log"):
            main()

        return json.loads(output.getvalue())

    def test_identity_injected_into_suffix(self):
        result = self._run_main(identity="Name: Yagish\nRole: Tech Lead")
        suffix = result["hookSpecificOutput"]["userPromptSuffix"]
        self.assertIn("Name: Yagish", suffix)
        self.assertIn("Role: Tech Lead", suffix)

    def test_output_has_hook_event_name(self):
        result = self._run_main()
        self.assertEqual(
            result["hookSpecificOutput"]["hookEventName"],
            "UserPromptSubmit",
        )

    def test_no_injection_when_nothing_to_inject(self):
        # When identity is empty and no facts/insights/cache → output {}
        # sys.exit must raise so we capture only the first print ({}).
        payload = json.dumps({"session_id": "test-empty", "prompt": "hi"})
        fake_db = "/fake/test.db"
        output  = io.StringIO()
        with patch("sys.stdin",  io.StringIO(payload)), \
             patch("sys.stdout", output), \
             patch("sys.exit",   side_effect=SystemExit), \
             patch.object(_wu, "DB_PATH", fake_db), \
             patch("os.path.exists", return_value=True), \
             patch.object(_wu, "init_db", return_value=MagicMock()), \
             patch.object(_wu, "_read_identity", return_value=""), \
             patch.object(_wu, "find_direct_answer", return_value=None), \
             patch.object(_wu, "_fetch_relevant_facts", return_value=[]), \
             patch.object(_wu, "list_insights", return_value=[]), \
             patch.object(_wu, "log_retrieval"), \
             patch.object(_wu, "activity_log"):
            with self.assertRaises(SystemExit):
                main()
        result = json.loads(output.getvalue())
        self.assertEqual(result, {})

    def test_cached_answer_appears_in_suffix(self):
        ca = {"question": "whats my name", "answer": "Alice", "similarity": 1.0}
        result = self._run_main(identity="Name: Alice", cached_answer=ca)
        suffix = result["hookSpecificOutput"]["userPromptSuffix"]
        self.assertIn("whats my name", suffix)
        self.assertIn("Alice", suffix)

    def test_identity_before_cached_answer_in_suffix(self):
        ca = {"question": "q", "answer": "old answer", "similarity": 0.95}
        result = self._run_main(identity="Name: Alice", cached_answer=ca)
        suffix = result["hookSpecificOutput"]["userPromptSuffix"]
        self.assertLess(suffix.find("Name: Alice"), suffix.find("[Cached Answer"))


if __name__ == "__main__":
    unittest.main()
