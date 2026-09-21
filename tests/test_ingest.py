import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import memory.servers.ingest_server as ingest_server_module
from memory.servers.ingest_server import app
from memory.db import init_db, list_recent_latency_breakdowns, list_recall_events, list_retrieval_lane_metrics
from memory.retrieval import WakeUpContext


class TestIngestEndpoints(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        ingest_server_module.DB_PATH = self.db_path
        self.client = TestClient(app)

    def tearDown(self):
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_post_ingest_stores_session(self):
        response = self.client.post(
            "/ingest",
            json={
                "session_id": "sess-001",
                "agent": "cursor",
                "turns": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
                "metadata": {"source": "cursor"},
                "repo_root": "/tmp/demo-repo",
                "cwd": "/tmp/demo-repo/app",
                "git_remote": "git@github.com:example/demo.git",
                "git_branch": "main",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["turns_stored"], 2)

        conn = init_db(self.db_path)
        row = conn.execute(
            "SELECT session_id, turn_count, metadata, project_id, repo_root, cwd, git_remote, git_branch FROM sessions WHERE session_id = ?",
            ("sess-001",),
        ).fetchone()
        conn.close()
        self.assertEqual(row["session_id"], "sess-001")
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual(json.loads(row["metadata"]), {"source": "cursor"})
        self.assertEqual(row["project_id"], "git@github.com:example/demo.git")
        self.assertEqual(row["repo_root"], "/tmp/demo-repo")
        self.assertEqual(row["cwd"], "/tmp/demo-repo/app")
        self.assertEqual(row["git_remote"], "git@github.com:example/demo.git")
        self.assertEqual(row["git_branch"], "main")

    def test_post_ingest_invalid_role_rejected(self):
        response = self.client.post(
            "/ingest",
            json={"session_id": "sess-002", "turns": [{"role": "system", "content": "bad"}]},
        )
        self.assertEqual(response.status_code, 422)

    def test_post_recall_returns_answer(self):
        context = WakeUpContext(
            None,
            None,
            [],
            [],
            [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
            [],
        )
        with patch.object(ingest_server_module, "retrieve_prompt_memory", return_value=context) as retrieve_prompt_memory, \
             patch.object(ingest_server_module, "log_memory_answer") as log_memory_answer, \
             patch.object(ingest_server_module, "log_memory_injection") as log_memory_injection:
            response = self.client.post(
                "/recall",
                json={
                    "prompt": "what is my name?",
                    "session_id": "sess-123",
                    "agent": "claude",
                    "repo_root": "/tmp/demo-repo",
                    "cwd": "/tmp/demo-repo/app",
                    "git_remote": "git@github.com:example/demo.git",
                    "git_branch": "main",
                },
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["action"], "answer")
        self.assertEqual(body["answer"], "Your name is Yash.")
        self.assertEqual(body["facts_count"], 1)
        retrieve_prompt_memory.assert_called_once_with(
            unittest.mock.ANY,
            "what is my name?",
            include_working_memory=False,
            session_id="sess-123",
            project_context={
                "project_id": "git@github.com:example/demo.git",
                "repo_root": "/tmp/demo-repo",
                "cwd": "/tmp/demo-repo/app",
                "git_remote": "git@github.com:example/demo.git",
                "git_branch": "main",
            },
        )
        log_memory_answer.assert_called_once()
        log_memory_injection.assert_not_called()

    def test_post_recall_logs_memory_injection(self):
        context = WakeUpContext(
            None,
            {"id": "wm-1", "current_goal": "Finish dashboard", "current_focus": "Add savings tiles"},
            [],
            [],
            [],
            [],
        )
        with patch.object(ingest_server_module, "retrieve_prompt_memory", return_value=context), \
             patch.object(ingest_server_module, "log_memory_answer") as log_memory_answer, \
             patch.object(ingest_server_module, "log_memory_injection") as log_memory_injection:
            response = self.client.post(
                "/recall",
                json={"prompt": "continue", "session_id": "sess-456", "agent": "claude"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["action"], "inject")
        self.assertIn("Memory context", body["injection"])
        log_memory_injection.assert_called_once()
        log_memory_answer.assert_not_called()

    def test_post_recall_persists_db_telemetry(self):
        context = WakeUpContext(
            None,
            None,
            [],
            [],
            [{"id": "fact-1", "content": "user.name = Yash", "similarity": 0.99}],
            [],
            timings={
                "prompt_embedding_ms": 12.5,
                "memory_search_ms": 44.2,
                "retrieval_total_ms": 60.0,
            },
        )
        with patch.object(ingest_server_module, "retrieve_prompt_memory", return_value=context):
            response = self.client.post(
                "/recall",
                json={"prompt": "what is my name?", "session_id": "sess-telemetry", "agent": "claude"},
            )

        self.assertEqual(response.status_code, 200)

        conn = init_db(self.db_path)
        try:
            recall_events = list_recall_events(conn, limit=10)
            self.assertEqual(len(recall_events), 2)
            self.assertEqual({event["event_type"] for event in recall_events}, {"request", "outcome"})
            outcome = next(event for event in recall_events if event["event_type"] == "outcome")
            self.assertEqual(outcome["action"], "answer")
            self.assertEqual(outcome["facts_count"], 1)

            latency_rows = list_recent_latency_breakdowns(conn, component="retrieval", operation="recall", limit=10)
            self.assertEqual({row["stage"] for row in latency_rows}, {"prompt_embedding", "memory_search", "retrieval_total"})

            lane_metrics = list_retrieval_lane_metrics(conn, limit=10)
            fact_lane = next(row for row in lane_metrics if row["lane"] == "facts")
            self.assertEqual(fact_lane["selected_count"], 1)
            self.assertEqual(fact_lane["hit_count"], 1)
        finally:
            conn.close()

    def test_status_ok(self):
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("embed_model_ready", response.json())


if __name__ == "__main__":
    unittest.main()
