import tempfile
import unittest
from unittest.mock import patch

from integrations.pi.adapter import handle_recall, handle_save
from memory.db import get_session_by_id, init_db


class TestPiAdapterRecall(unittest.TestCase):
    def test_recall_returns_noop_when_server_is_unavailable(self):
        with patch("integrations.pi.adapter.MemoryClient.recall", side_effect=ConnectionError("offline")):
            result = handle_recall({"prompt": "hello"}, db_path="/tmp/definitely-missing-agentic-memory.db")
        self.assertEqual(result["action"], "noop")
        self.assertEqual(result["server_required"], True)
        self.assertIn("offline", result["error"])

    def test_recall_returns_answer_from_server(self):
        with patch(
            "integrations.pi.adapter.MemoryClient.recall",
            return_value={
                "action": "answer",
                "answer": "Your name is Yash.",
                "facts_count": 1,
                "episodic_count": 0,
                "procedural_count": 0,
                "warnings": [],
                "context": {"facts": [{"id": "fact-1", "content": "user.name = Yash"}], "episodic": [], "procedural": []},
            },
        ) as recall, \
        patch("integrations.pi.adapter._wake_log_info") as wake_log_info:
            result = handle_recall({
                "prompt": "what is my name?",
                "session_id": "pi-session-1",
                "cwd": "/tmp/demo-repo/app",
                "repo_root": "/tmp/demo-repo",
                "git_remote": "git@github.com:example/demo.git",
                "git_branch": "main",
            })

        self.assertEqual(result["action"], "answer")
        self.assertEqual(result["answer"], "Your name is Yash.")
        self.assertEqual(result["facts_count"], 1)
        self.assertEqual(result["episodic_count"], 0)
        recall.assert_called_once_with(
            "what is my name?",
            include_working_memory=False,
            session_id="pi-session-1",
            agent="pi",
            project_id="git@github.com:example/demo.git",
            repo_root="/tmp/demo-repo",
            cwd="/tmp/demo-repo/app",
            git_remote="git@github.com:example/demo.git",
            git_branch="main",
        )
        wake_log_info.assert_called()


class TestPiAdapterSave(unittest.TestCase):
    def test_save_persists_session(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp, \
             patch("integrations.pi.adapter._save_log_info") as save_log_info:
            result = handle_save(
                {
                    "session_id": "pi-session-1",
                    "agent": "pi",
                    "turns": [
                        {"role": "user", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ],
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:01:00+00:00",
                    "metadata": {"integration": "pi"},
                    "repo_root": "/tmp/demo-repo",
                    "cwd": "/tmp/demo-repo/app",
                    "git_remote": "git@github.com:example/demo.git",
                    "git_branch": "main",
                },
                db_path=tmp.name,
            )

            conn = init_db(tmp.name)
            try:
                session = get_session_by_id(conn, "pi-session-1")
            finally:
                conn.close()

        self.assertTrue(result["ok"])
        self.assertEqual(result["turns_stored"], 2)
        self.assertEqual(session["agent"], "pi")
        self.assertEqual(session["project_id"], "git@github.com:example/demo.git")
        self.assertEqual(session["repo_root"], "/tmp/demo-repo")
        self.assertEqual(session["cwd"], "/tmp/demo-repo/app")
        self.assertEqual(session["git_remote"], "git@github.com:example/demo.git")
        self.assertEqual(session["git_branch"], "main")
        save_log_info.assert_called()

    def test_save_rejects_empty_turns(self):
        with patch("integrations.pi.adapter._save_log_error") as save_log_error:
            result = handle_save({"session_id": "pi-session-1", "turns": []})
        self.assertEqual(result, {"ok": False, "error": "turns must not be empty"})
        save_log_error.assert_called_once()


if __name__ == "__main__":
    unittest.main()
