# test_client.py — automated tests for the MemoryClient thin client (Phase 16).
#
# Run with:
#   python3 -m pytest tests/test_client.py -v
#
# These tests use unittest.mock to replace urllib.request.urlopen so no real
# HTTP connection is made — the client is tested in isolation.

import io           # for creating a fake HTTP response body stream
import json         # for building fake JSON response bytes
import os           # for module path setup
import sys          # for module search path manipulation
import unittest     # the standard Python test framework
from unittest.mock import patch, MagicMock   # for replacing urlopen with a fake

import urllib.error  # for constructing fake urllib exceptions

# Add the project root to the module search path so 'memory.*' imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.client import MemoryClient   # the class we are testing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(data: dict):
    """
    Build a mock object that looks like the file-like object returned by
    urllib.request.urlopen().

    urllib's urlopen returns a context manager whose __enter__ yields an
    object with a .read() method.  We simulate that here.

    Args:
        data — dict to serialise as the fake response body

    Returns:
        A MagicMock usable as `with urlopen(...) as response: response.read()`.
    """
    # Encode the dict as UTF-8 JSON bytes — same as a real HTTP response.
    body_bytes = json.dumps(data).encode("utf-8")

    # Build the mock response object with a .read() method.
    mock_response = MagicMock()
    mock_response.read.return_value = body_bytes

    # Build the context manager wrapper so `with urlopen(...) as r` works.
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_response
    mock_cm.__exit__.return_value = False   # don't suppress exceptions

    return mock_cm


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestMemoryClient(unittest.TestCase):
    """Unit tests for MemoryClient using mocked urlopen."""

    # ------------------------------------------------------------------
    # Test 7: ConnectionError raised when server is not running
    # ------------------------------------------------------------------

    def test_client_raises_connection_error(self):
        """
        When urlopen raises urllib.error.URLError (e.g. server not running),
        MemoryClient.save_session() must re-raise it as ConnectionError with
        a helpful message telling the user how to start the server.
        """
        # urllib.error.URLError is what urlopen raises on connection failure.
        # reason can be any string — we use a typical "Connection refused" message.
        fake_url_error = urllib.error.URLError("Connection refused")

        client = MemoryClient()

        # Patch urlopen at the location it is looked up — inside urllib.request.
        with patch("urllib.request.urlopen", side_effect=fake_url_error):
            # save_session must raise ConnectionError, not URLError.
            with self.assertRaises(ConnectionError) as ctx:
                client.save_session(
                    session_id="test-session",
                    agent="test-agent",
                    turns=[{"role": "user", "content": "Hello"}],
                )

        # The error message must mention how to start the server.
        self.assertIn("ingest-server start", str(ctx.exception))

    # ------------------------------------------------------------------
    # Test 8: save_session returns the dict from the server on success
    # ------------------------------------------------------------------

    def test_client_raises_runtime_error_for_http_error(self):
        """
        When the server returns an HTTP status like 422 or 500, the client must
        surface that as a RuntimeError instead of pretending the server is down.
        """
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
                client.save_session(
                    session_id="bad-session",
                    agent="test-agent",
                    turns=[],
                )

        self.assertIn("HTTP 422", str(ctx.exception))
        self.assertIn("bad input", str(ctx.exception))

    def test_client_save_session_returns_dict(self):
        """
        When urlopen succeeds and returns {"ok": true, "session_id": "abc"},
        MemoryClient.save_session() must return that same dict unchanged.
        """
        # Build the fake server response.
        fake_response_data = {"ok": True, "session_id": "abc"}

        client = MemoryClient()

        # Patch urlopen to return the fake response without making a real connection.
        with patch("urllib.request.urlopen", return_value=_fake_response(fake_response_data)):
            result = client.save_session(
                session_id="abc",
                agent="cursor",
                turns=[
                    {"role": "user",      "content": "What time is it?"},
                    {"role": "assistant", "content": "It is 12:00 PM."},
                ],
            )

        # The returned dict must exactly match the fake server response.
        self.assertEqual(result, fake_response_data)

        # Confirm the individual fields are correct too.
        self.assertTrue(result["ok"])
        self.assertEqual(result["session_id"], "abc")


if __name__ == "__main__":
    unittest.main()
