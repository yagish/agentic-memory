import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import memory.ingest_server as ingest_server_module
from memory.ingest_server import app
from memory.db import init_db
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
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["turns_stored"], 2)

        conn = init_db(self.db_path)
        row = conn.execute(
            "SELECT session_id, turn_count, metadata FROM sessions WHERE session_id = ?",
            ("sess-001",),
        ).fetchone()
        conn.close()
        self.assertEqual(row["session_id"], "sess-001")
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual(json.loads(row["metadata"]), {"source": "cursor"})

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
        with patch.object(ingest_server_module, "retrieve_prompt_memory", return_value=context):
            response = self.client.post(
                "/recall",
                json={"prompt": "what is my name?"},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["action"], "answer")
        self.assertEqual(body["answer"], "Your name is Yash.")
        self.assertEqual(body["facts_count"], 1)

    def test_status_ok(self):
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertIn("embed_model_ready", response.json())


if __name__ == "__main__":
    unittest.main()
