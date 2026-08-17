# test_db.py — automated tests for the storage layer (memory/db.py).
#
# Run with:  python3 -m unittest tests.test_db -v
#
# Each test creates a fresh in-memory database (no files on disk),
# runs an operation, and asserts the result is what we expect.
# If an assertion fails, the test fails and prints what went wrong.

import json
import unittest

# Import the three functions we want to test.
from memory.db import init_db, upsert_session, search


# ---------------------------------------------------------------------------
# Sample transcripts used across multiple tests.
# A transcript is a list of turns — each turn is a dict with "role" and "content".
# ---------------------------------------------------------------------------

TRANSCRIPT_A = [
    {"role": "user",      "content": "Tell me about the quantum entanglement experiment."},
    {"role": "assistant", "content": "Quantum entanglement is a phenomenon where particles become correlated."},
]

# TRANSCRIPT_B extends TRANSCRIPT_A with two more turns (simulates a continued session).
TRANSCRIPT_B = TRANSCRIPT_A + [
    {"role": "user",      "content": "Can you elaborate on the Bell inequality?"},
    {"role": "assistant", "content": "The Bell inequality tests whether local hidden variables can explain entanglement."},
]


# ---------------------------------------------------------------------------
# Test group 1: inserting and updating sessions
# ---------------------------------------------------------------------------

class TestUpsertAndRead(unittest.TestCase):

    def setUp(self):
        # setUp runs before each individual test method.
        # ":memory:" creates a temporary SQLite database that exists only in RAM
        # and is discarded after the test — no cleanup needed.
        self.conn = init_db(":memory:")

    def test_insert_and_read_verbatim(self):
        # Save a session, then read it back raw from the database.
        upsert_session(self.conn, "s1", "claude", TRANSCRIPT_A, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z")

        row = self.conn.execute(
            "SELECT transcript, turn_count FROM sessions WHERE session_id = 's1'"
        ).fetchone()

        # json.loads converts the stored JSON string back into a Python list.
        # We assert it's identical to what we put in — verbatim, nothing changed.
        self.assertEqual(json.loads(row["transcript"]), TRANSCRIPT_A)

        # TRANSCRIPT_A has 2 turns, so turn_count should be 2.
        self.assertEqual(row["turn_count"], 2)

    def test_upsert_updates_turn_count_and_transcript(self):
        # Save session s1 with the short transcript...
        upsert_session(self.conn, "s1", "claude", TRANSCRIPT_A, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z")

        # ...then save it again with the longer transcript (same session_id).
        # This should overwrite the first save, not create a duplicate row.
        upsert_session(self.conn, "s1", "claude", TRANSCRIPT_B, "2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z")

        row = self.conn.execute(
            "SELECT transcript, turn_count, updated_at FROM sessions WHERE session_id = 's1'"
        ).fetchone()

        # The stored transcript should now be TRANSCRIPT_B (the longer one).
        self.assertEqual(json.loads(row["transcript"]), TRANSCRIPT_B)

        # TRANSCRIPT_B has 4 turns.
        self.assertEqual(row["turn_count"], 4)

        # updated_at should reflect the second save, not the first.
        self.assertEqual(row["updated_at"], "2026-01-01T00:05:00Z")


# ---------------------------------------------------------------------------
# Test group 2: full-text search
# ---------------------------------------------------------------------------

class TestFTS5Search(unittest.TestCase):

    def setUp(self):
        # Create a fresh database and seed it with two sessions on different topics.
        self.conn = init_db(":memory:")

        # Session s1: quantum physics conversation.
        upsert_session(
            self.conn, "s1", "claude", TRANSCRIPT_A,
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z"
        )

        # Session s2: physics but different topic (speed of light).
        upsert_session(
            self.conn, "s2", "claude",
            [
                {"role": "user",      "content": "What is the speed of light?"},
                {"role": "assistant", "content": "The speed of light in a vacuum is approximately 299,792,458 metres per second."},
            ],
            "2026-01-02T00:00:00Z", "2026-01-02T00:01:00Z"
        )

    def test_search_returns_matching_session(self):
        # "entanglement" only appears in s1, so we should get exactly one result.
        results = search(self.conn, "entanglement")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "s1")

    def test_search_no_match_returns_empty(self):
        # A word that appears in neither session should return an empty list.
        results = search(self.conn, "photosynthesis")
        self.assertEqual(results, [])

    def test_search_result_has_required_keys(self):
        # Every result dict must have these four keys — callers depend on them.
        results = search(self.conn, "speed of light")
        self.assertEqual(len(results), 1)
        self.assertIn("session_id", results[0])
        self.assertIn("agent",      results[0])
        self.assertIn("updated_at", results[0])
        self.assertIn("snippet",    results[0])   # the highlighted excerpt


if __name__ == "__main__":
    unittest.main()
