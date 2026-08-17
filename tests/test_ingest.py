# test_ingest.py — automated tests for the HTTP ingest server (Phase 16).
#
# Run with:
#   python3 -m pytest tests/test_ingest.py -v
#
# These tests use FastAPI's TestClient (backed by starlette) which runs the
# ASGI app in-process — no real HTTP server is started, so the tests are fast
# and require no open ports.

import json          # for inspecting raw JSON responses
import os            # for database file path helpers
import sys           # for module search path manipulation
import tempfile      # for creating a temporary database file
import unittest      # the standard Python test framework

# Add the project root to the module search path so 'memory.*' imports work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch DB_PATH in ingest_server to use a temp file before importing the app.
# We do this by setting the environment variable that ingest_server reads,
# but ingest_server uses DB_PATH directly, so we monkey-patch after import.
from starlette.testclient import TestClient   # test HTTP client for ASGI apps

# Import the FastAPI app.  ingest_server.py uses a module-level DB_PATH
# constant; we override it per-test using the _override_db helper below.
import memory.ingest_server as ingest_server_module
from memory.ingest_server import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_db() -> str:
    """
    Create a temporary SQLite database file and return its path.

    Using a real file (not :memory:) because multiple connections may be
    opened during one request — in-memory DBs aren't shared between connections.
    """
    # mkstemp creates the file and returns a file descriptor we close immediately.
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return path


# Two sample turns — valid role values used in most tests.
SAMPLE_TURNS = [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestIngestEndpoints(unittest.TestCase):
    """Integration tests for POST /ingest and GET /status."""

    def setUp(self):
        """
        Before each test, create a fresh temporary DB and point ingest_server
        at it, then build a TestClient wrapping the FastAPI app.
        """
        # Create a fresh temp database for isolation between tests.
        self._db_path = _tmp_db()
        # Monkey-patch the module-level constant so the handlers use our DB.
        ingest_server_module.DB_PATH = self._db_path
        # TestClient gives us a requests-like interface to the ASGI app.
        self.client = TestClient(app)

    def tearDown(self):
        """Remove the temporary database file after each test."""
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Test 1: POST /ingest stores the session in the database
    # ------------------------------------------------------------------

    def test_post_ingest_stores_session(self):
        """
        A valid POST /ingest request must return {"ok": true} and the session
        must be findable in the sessions table.
        """
        payload = {
            "session_id": "sess-001",
            "agent":      "cursor",
            "turns":      SAMPLE_TURNS,
        }

        # Send the POST request via the test client.
        response = self.client.post("/ingest", json=payload)

        # 200 OK with an ok flag.
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["session_id"], "sess-001")

        # Verify the session was written to the database.
        from memory.db import init_db
        conn = init_db(self._db_path)
        row = conn.execute(
            "SELECT session_id, turn_count FROM sessions WHERE session_id = 'sess-001'"
        ).fetchone()
        conn.close()

        # The row must exist with the correct turn count.
        self.assertIsNotNone(row)
        self.assertEqual(row["turn_count"], len(SAMPLE_TURNS))

    # ------------------------------------------------------------------
    # Test 2: POST /ingest populates the chunks table
    # ------------------------------------------------------------------

    def test_post_ingest_stores_chunks(self):
        """
        After a successful ingest, the chunks table must contain at least one
        row for the ingested session.
        """
        payload = {
            "session_id": "sess-002",
            "agent":      "langchain",
            "turns":      SAMPLE_TURNS,
        }

        self.client.post("/ingest", json=payload)

        # Check that chunks were stored.
        from memory.db import init_db, get_chunks_for_session
        conn = init_db(self._db_path)
        chunks = get_chunks_for_session(conn, "sess-002")
        conn.close()

        # At least one chunk must have been created for these two turns.
        self.assertGreater(len(chunks), 0)

    # ------------------------------------------------------------------
    # Test 3: empty turns list is rejected with 422
    # ------------------------------------------------------------------

    def test_post_ingest_empty_turns_rejected(self):
        """
        Sending an empty turns list must result in a 422 Unprocessable Entity
        response — the server should never store an empty session.
        """
        payload = {
            "session_id": "sess-003",
            "turns":      [],   # invalid: no turns
        }

        response = self.client.post("/ingest", json=payload)

        # 422 is FastAPI/Pydantic's standard response for validation failures.
        self.assertEqual(response.status_code, 422)

    # ------------------------------------------------------------------
    # Test 4: invalid role is rejected with 422
    # ------------------------------------------------------------------

    def test_post_ingest_invalid_role_rejected(self):
        """
        A turn with a role other than 'user' or 'assistant' must be rejected
        with 422 — the validator on the Turn model enforces this.
        """
        payload = {
            "session_id": "sess-004",
            "turns": [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
        }

        response = self.client.post("/ingest", json=payload)

        # FastAPI propagates the Pydantic ValueError as a 422 response.
        self.assertEqual(response.status_code, 422)

    # ------------------------------------------------------------------
    # Test 5: GET /status returns session count after ingest
    # ------------------------------------------------------------------

    def test_get_status_returns_counts(self):
        """
        After ingesting a session, GET /status must return total_sessions >= 1.
        """
        # First ingest a session so there is data to count.
        self.client.post("/ingest", json={
            "session_id": "sess-005",
            "turns":      SAMPLE_TURNS,
        })

        # Now query the status endpoint.
        response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()

        # status must be "ok".
        self.assertEqual(body["status"], "ok")

        # total_sessions must be at least 1 since we just ingested one.
        self.assertGreaterEqual(body["total_sessions"], 1)

        # db_size_bytes must be a non-negative integer.
        self.assertIsInstance(body["db_size_bytes"], int)
        self.assertGreaterEqual(body["db_size_bytes"], 0)

    # ------------------------------------------------------------------
    # Test 6: POST /ingest accepts a metadata dict
    # ------------------------------------------------------------------

    def test_post_ingest_with_metadata(self):
        """
        A metadata dict in the request body must be stored in the sessions table
        as a JSON string and the endpoint must still return ok=True.
        """
        meta = {"source": "cursor", "project": "alpha", "user_id": 42}
        payload = {
            "session_id": "sess-006",
            "agent":      "cursor",
            "turns":      SAMPLE_TURNS,
            "metadata":   meta,
        }

        response = self.client.post("/ingest", json=payload)

        # The request must succeed.
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])

        # Verify the metadata was persisted correctly.
        from memory.db import init_db
        conn = init_db(self._db_path)
        row = conn.execute(
            "SELECT metadata FROM sessions WHERE session_id = 'sess-006'"
        ).fetchone()
        conn.close()

        # The metadata column must not be NULL.
        self.assertIsNotNone(row["metadata"])

        # Decoding the stored JSON must reproduce the original dict.
        stored_meta = json.loads(row["metadata"])
        self.assertEqual(stored_meta, meta)


if __name__ == "__main__":
    unittest.main()
