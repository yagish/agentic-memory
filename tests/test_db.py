# test_db.py — automated tests for the storage layer (memory/db.py).
#
# Run with:  python3 -m unittest tests.test_db -v
#
# Each test creates a fresh in-memory database (no files on disk),
# runs an operation, and asserts the result is what we expect.
# If an assertion fails, the test fails and prints what went wrong.

import json
import unittest

# Import the storage and chunking functions we want to test.
from memory.db import (
    init_db,
    upsert_session,
    search,
    store_chunk,
    get_chunks_for_session,
    delete_chunks_for_session,
    chunk_transcript,
)


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


# ---------------------------------------------------------------------------
# Test group 3: chunk storage functions (Ticket 7-01)
# ---------------------------------------------------------------------------

class TestChunkStorage(unittest.TestCase):

    def setUp(self):
        # Create a fresh in-memory database and a session row to attach chunks to.
        # The chunks table has a REFERENCES sessions(session_id) foreign key.
        self.conn = init_db(":memory:")
        upsert_session(
            self.conn, "sess1", "claude", TRANSCRIPT_A,
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
        )

    def test_store_and_retrieve_two_chunks(self):
        # Store two chunks for the same session, then retrieve them.
        # We expect to get back exactly 2 rows in chunk_index order.
        store_chunk(self.conn, "sess1", 0, "first chunk text", None)
        store_chunk(self.conn, "sess1", 1, "second chunk text", None)

        chunks = get_chunks_for_session(self.conn, "sess1")

        # There must be exactly 2 chunks.
        self.assertEqual(len(chunks), 2)

        # chunk_index and text must match what we stored, in order.
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertEqual(chunks[0]["text"], "first chunk text")
        self.assertEqual(chunks[1]["chunk_index"], 1)
        self.assertEqual(chunks[1]["text"], "second chunk text")

    def test_delete_chunks_returns_empty_list(self):
        # After storing a chunk and then deleting all chunks for the session,
        # get_chunks_for_session should return an empty list.
        store_chunk(self.conn, "sess1", 0, "some text", None)
        delete_chunks_for_session(self.conn, "sess1")

        chunks = get_chunks_for_session(self.conn, "sess1")
        self.assertEqual(chunks, [])

    def test_store_chunk_upserts_on_same_id(self):
        # Storing a chunk twice with the same session_id + chunk_index should
        # overwrite the first row (upsert), not create a duplicate.
        store_chunk(self.conn, "sess1", 0, "original text", None)
        store_chunk(self.conn, "sess1", 0, "updated text", None)

        chunks = get_chunks_for_session(self.conn, "sess1")

        # Only one chunk row should exist — the second write replaced the first.
        self.assertEqual(len(chunks), 1)
        # The text should be the latest value, not the original.
        self.assertEqual(chunks[0]["text"], "updated text")


# ---------------------------------------------------------------------------
# Test group 4: chunk_transcript function (Ticket 7-02)
# ---------------------------------------------------------------------------

def _make_turns(n: int) -> list[dict]:
    """
    Helper: build a list of n alternating user/assistant turns.
    Each turn has a unique content string so we can track which chunk it lands in.
    """
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": f"turn {i}"} for i in range(n)]


class TestChunkTranscript(unittest.TestCase):

    def test_six_turns_window_six_gives_one_chunk(self):
        # With exactly window=6 turns and window=6, the whole transcript fits in
        # a single chunk — no stepping needed.
        turns = _make_turns(6)
        chunks = chunk_transcript(turns, window=6, overlap=1)

        # Exactly one chunk must be returned.
        self.assertEqual(len(chunks), 1)

        # That chunk must contain all 6 turns' content.
        for i in range(6):
            self.assertIn(f"turn {i}", chunks[0])

    def test_seven_turns_window_six_gives_two_chunks(self):
        # 7 turns, window=6, overlap=1 → step=5.
        # Chunk 0 covers turns 0-5; chunk 1 covers turns 5-6 (turn 5 repeated).
        turns = _make_turns(7)
        chunks = chunk_transcript(turns, window=6, overlap=1)

        # Two chunks must be produced.
        self.assertEqual(len(chunks), 2)

        # Chunk 1 must start at turn 5 (the overlap turn).
        self.assertIn("turn 5", chunks[1])

    def test_twelve_turns_window_six_gives_three_chunks(self):
        # 12 turns, window=6, overlap=1 → step=5.
        # Chunk 0: turns 0-5; chunk 1: turns 5-10; chunk 2: turns 10-11.
        turns = _make_turns(12)
        chunks = chunk_transcript(turns, window=6, overlap=1)

        # Three chunks must be produced.
        self.assertEqual(len(chunks), 3)

    def test_empty_turns_returns_single_empty_string(self):
        # An empty turn list must return [""] so callers never handle an empty list.
        chunks = chunk_transcript([], window=6, overlap=1)

        self.assertEqual(chunks, [""])

    def test_non_string_content_turns_are_skipped(self):
        # Turns with list/dict content (tool calls etc.) must be silently ignored.
        turns = [
            {"role": "user",      "content": "hello"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "x"}]},  # non-string
            {"role": "user",      "content": {"nested": "dict"}},                 # non-string
            {"role": "assistant", "content": "world"},
        ]
        # Only "hello" and "world" are string turns — they should both appear.
        chunks = chunk_transcript(turns, window=6, overlap=1)

        self.assertEqual(len(chunks), 1)
        self.assertIn("hello", chunks[0])
        self.assertIn("world", chunks[0])
        # The non-string content must not appear.
        self.assertNotIn("tool_use", chunks[0])
        self.assertNotIn("nested", chunks[0])


if __name__ == "__main__":
    unittest.main()
