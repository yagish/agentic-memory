import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.servers.client import MemoryClient


def _fake_response(data: dict):
    body_bytes = json.dumps(data).encode("utf-8")
    mock_response = MagicMock()
    mock_response.read.return_value = body_bytes
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_response
    mock_cm.__exit__.return_value = False
    return mock_cm


class TestMemoryClient(unittest.TestCase):
    def test_client_raises_connection_error(self):
        client = MemoryClient()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with self.assertRaises(ConnectionError) as ctx:
                client.save_session(
                    session_id="test-session",
                    agent="test-agent",
                    turns=[{"role": "user", "content": "Hello"}],
                )
        self.assertIn("python3 memory/ingest_server.py", str(ctx.exception))

    def test_client_raises_runtime_error_for_http_error(self):
        client = MemoryClient()
        error_body = io.BytesIO(json.dumps({"detail": {"ok": False, "error": "bad input"}}).encode("utf-8"))
        fake_http_error = urllib.error.HTTPError(
            url="http://localhost:7747/ingest",
            code=422,
            msg="Unprocessable Entity",
            hdrs=None,
            fp=error_body,
        )

        with patch("urllib.request.urlopen", side_effect=fake_http_error):
            with self.assertRaises(RuntimeError) as ctx:
                client.save_session(session_id="bad-session", agent="test-agent", turns=[])

        self.assertIn("HTTP 422", str(ctx.exception))
        self.assertIn("bad input", str(ctx.exception))

    def test_client_save_session_returns_dict(self):
        client = MemoryClient()
        with patch("urllib.request.urlopen", return_value=_fake_response({"ok": True, "session_id": "abc"})):
            result = client.save_session(
                session_id="abc",
                agent="cursor",
                turns=[
                    {"role": "user", "content": "What time is it?"},
                    {"role": "assistant", "content": "It is 12:00 PM."},
                ],
            )
        self.assertEqual(result, {"ok": True, "session_id": "abc"})

    def test_client_recall_returns_dict(self):
        client = MemoryClient()
        with patch("urllib.request.urlopen", return_value=_fake_response({"action": "answer", "answer": "Your name is Yash."})):
            result = client.recall("what is my name?")
        self.assertEqual(result, {"action": "answer", "answer": "Your name is Yash."})

    def test_client_status_returns_dict(self):
        client = MemoryClient()
        with patch("urllib.request.urlopen", return_value=_fake_response({"status": "ok"})):
            result = client.status()
        self.assertEqual(result, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
