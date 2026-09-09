# test_save_hook.py — tests for the Claude Stop-hook adapter.
#
# Run with:  python3 -m unittest tests.test_save_hook -v
#
# We test save_session() and parse_transcript() directly — no need to
# simulate stdin or subprocess calls. The tests create real temporary files
# so the JSONL parsing is exercised exactly as it runs in production.

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from integrations.claude.save_hook import parse_transcript, save_session
from memory.db import init_db


def make_jsonl(path: str, turns: list[dict], session_id: str = "test-session-1") -> None:
    """
    Write a minimal JSONL transcript file that matches the real Claude Code format.

    Each turn becomes one line. User turns have a string content field;
    assistant turns have a list of content blocks (matching the real format).
    """
    with open(path, "w") as f:
        for i, turn in enumerate(turns):
            if turn["role"] == "user":
                obj = {
                    "type": "user",
                    "sessionId": session_id,
                    "timestamp": f"2026-08-16T10:0{i}:00.000Z",
                    "isMeta": False,
                    "message": {
                        "role": "user",
                        "content": turn["content"],
                    },
                }
            else:
                # Assistant content is an array of typed blocks in real transcripts.
                obj = {
                    "type": "assistant",
                    "sessionId": session_id,
                    "timestamp": f"2026-08-16T10:0{i}:30.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": turn["content"]},
                        ],
                    },
                }
            f.write(json.dumps(obj) + "\n")


# Sample conversation used across multiple tests.
SAMPLE_TURNS = [
    {"role": "user",      "content": "What is quantum entanglement?"},
    {"role": "assistant", "content": "Quantum entanglement is a phenomenon where two particles become correlated."},
    {"role": "user",      "content": "Can you give me an example?"},
    {"role": "assistant", "content": "Sure! If you measure one particle's spin, the other's spin is instantly determined."},
]


class TestParseTranscript(unittest.TestCase):

    def test_returns_correct_turns(self):
        # Write a JSONL file and confirm parse_transcript returns the right turns.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name

        try:
            make_jsonl(jsonl_path, SAMPLE_TURNS)
            turns, started_at, session_id = parse_transcript(jsonl_path)

            self.assertEqual(len(turns), 4)
            self.assertEqual(turns[0]["role"], "user")
            self.assertEqual(turns[0]["content"], "What is quantum entanglement?")
            self.assertEqual(turns[1]["role"], "assistant")
            self.assertIn("correlated", turns[1]["content"])
            self.assertEqual(session_id, "test-session-1")
            self.assertIsNotNone(started_at)
        finally:
            os.unlink(jsonl_path)

    def test_skips_meta_lines(self):
        # isMeta=True user lines (system events) should be excluded from turns.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name

        try:
            with open(jsonl_path, "w") as f:
                # A real meta line (isMeta=True)
                f.write(json.dumps({
                    "type": "user",
                    "sessionId": "s1",
                    "isMeta": True,
                    "message": {"role": "user", "content": "system event"},
                }) + "\n")
                # A real user message
                f.write(json.dumps({
                    "type": "user",
                    "sessionId": "s1",
                    "timestamp": "2026-08-16T10:00:00Z",
                    "message": {"role": "user", "content": "hello"},
                }) + "\n")

            turns, _, _ = parse_transcript(jsonl_path)
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["content"], "hello")
        finally:
            os.unlink(jsonl_path)

    def test_user_block_list_content_supported(self):
        # Real Claude transcripts store human prompts as a list of typed blocks.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name

        try:
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "type": "user",
                    "sessionId": "s1",
                    "timestamp": "2026-08-16T10:00:00Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "first line"},
                            {"type": "text", "text": "second line"},
                        ],
                    },
                }) + "\n")
                # Tool-result pseudo-user events should not become transcript turns.
                f.write(json.dumps({
                    "type": "user",
                    "sessionId": "s1",
                    "timestamp": "2026-08-16T10:00:01Z",
                    "message": {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "tool-1", "content": [{"type": "text", "text": "tool bytes"}]},
                        ],
                    },
                }) + "\n")

            turns, started_at, _ = parse_transcript(jsonl_path)
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["role"], "user")
            self.assertEqual(turns[0]["content"], "first line\nsecond line")
            self.assertEqual(started_at, "2026-08-16T10:00:00Z")
        finally:
            os.unlink(jsonl_path)

    def test_assistant_thinking_blocks_excluded(self):
        # Thinking blocks should be stripped; only text blocks are kept.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name

        try:
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "type": "assistant",
                    "sessionId": "s1",
                    "message": {
                        "role": "assistant",
                        "stop_reason": "end_turn",
                        "content": [
                            {"type": "thinking", "thinking": "internal reasoning here"},
                            {"type": "text",     "text": "visible reply to user"},
                        ],
                    },
                }) + "\n")

            turns, _, _ = parse_transcript(jsonl_path)
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["content"], "visible reply to user")
        finally:
            os.unlink(jsonl_path)

    def test_assistant_tool_use_preamble_excluded(self):
        # Skip assistant text emitted right before a tool call; keep final answer only.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name

        try:
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "type": "assistant",
                    "sessionId": "s1",
                    "message": {
                        "role": "assistant",
                        "stop_reason": "tool_use",
                        "content": [{"type": "text", "text": "Let me inspect the file."}],
                    },
                }) + "\n")
                f.write(json.dumps({
                    "type": "assistant",
                    "sessionId": "s1",
                    "message": {
                        "role": "assistant",
                        "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "Done. Here is the final answer."}],
                    },
                }) + "\n")

            turns, _, _ = parse_transcript(jsonl_path)
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["content"], "Done. Here is the final answer.")
        finally:
            os.unlink(jsonl_path)


class TestSaveSession(unittest.TestCase):

    def setUp(self):
        # Create a temporary JSONL transcript and a temporary DB for each test.
        self.tmp_jsonl = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        make_jsonl(self.tmp_jsonl.name, SAMPLE_TURNS, session_id="sess-abc")
        self.tmp_jsonl.close()

        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp_db.close()

    def tearDown(self):
        os.unlink(self.tmp_jsonl.name)
        os.unlink(self.tmp_db.name)

    def _run_save(self, session_id="sess-abc"):
        """Helper: call save_session with a payload pointing at the temp files."""
        # Monkey-patch the DB_PATH so the hook writes to our temp DB, not ~/.memory/memory.db
        import integrations.claude.save_hook as hook_module
        original_db_path = hook_module.DB_PATH
        hook_module.DB_PATH = self.tmp_db.name
        try:
            save_session({
                "session_id": session_id,
                "transcript_path": self.tmp_jsonl.name,
                "stop_hook_active": False,
            })
        finally:
            hook_module.DB_PATH = original_db_path

    def test_session_saved_to_db(self):
        # After save_session runs, the session row must exist in the DB.
        self._run_save()

        conn = init_db(self.tmp_db.name)
        row = conn.execute(
            "SELECT session_id, agent, turn_count, transcript FROM sessions WHERE session_id = 'sess-abc'"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(row, "session row was not written to DB")
        self.assertEqual(row["session_id"], "sess-abc")
        self.assertEqual(row["agent"], "claude")
        self.assertEqual(row["turn_count"], 4)

        stored = json.loads(row["transcript"])
        self.assertEqual(stored[0]["content"], "What is quantum entanglement?")

    def test_upsert_overwrites_on_second_save(self):
        # Saving the same session twice should update the row, not create a duplicate.
        self._run_save()
        self._run_save()

        conn = init_db(self.tmp_db.name)
        count = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = 'sess-abc'"
        ).fetchone()[0]
        conn.close()

        self.assertEqual(count, 1, "upsert created a duplicate row")


if __name__ == "__main__":
    unittest.main()
