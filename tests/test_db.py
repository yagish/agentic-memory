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
    insert_fact,
    update_fact,
    delete_fact,
    search_facts,
    list_facts,
    hybrid_search,
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


# ---------------------------------------------------------------------------
# Test group 5: facts CRUD functions (Ticket 8-01)
# ---------------------------------------------------------------------------

class TestFacts(unittest.TestCase):

    def setUp(self):
        # Each test gets its own fresh in-memory database — no state leaks between tests.
        self.conn = init_db(":memory:")

    def test_insert_fact_returns_string_id(self):
        # insert_fact should return a non-empty string (a UUID like "a3f2-...").
        fact_id = insert_fact(self.conn, "Python uses indentation for code blocks.")
        self.assertIsInstance(fact_id, str)
        self.assertGreater(len(fact_id), 0)

    def test_insert_fact_row_is_retrievable(self):
        # After inserting, the row should exist in the database with the correct content.
        fact_id = insert_fact(self.conn, "Rust has no garbage collector.", tags=["rust"])
        row = self.conn.execute(
            "SELECT content, source FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        # The row must exist — None means the insert failed.
        self.assertIsNotNone(row)
        # Content must be stored verbatim.
        self.assertEqual(row["content"], "Rust has no garbage collector.")
        # Default source is "manual".
        self.assertEqual(row["source"], "manual")

    def test_update_fact_changes_content_and_returns_true(self):
        # update_fact should overwrite the content field and return True.
        fact_id = insert_fact(self.conn, "original content")
        result = update_fact(self.conn, fact_id, content="updated content")

        # The return value signals success: True means the row was found and changed.
        self.assertTrue(result)

        # Read the row back directly to confirm the content was actually changed.
        row = self.conn.execute(
            "SELECT content FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        self.assertEqual(row["content"], "updated content")

    def test_update_fact_bumps_updated_at(self):
        # updated_at must be refreshed to a time >= created_at after an update.
        fact_id = insert_fact(self.conn, "a fact")
        row_before = self.conn.execute(
            "SELECT created_at FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()

        update_fact(self.conn, fact_id, content="changed")

        row_after = self.conn.execute(
            "SELECT updated_at FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()

        # ISO timestamp strings sort lexicographically by time, so >= works correctly.
        self.assertGreaterEqual(row_after["updated_at"], row_before["created_at"])

    def test_update_fact_returns_false_for_unknown_id(self):
        # update_fact on a non-existent id must return False — not raise an exception.
        result = update_fact(self.conn, "does-not-exist", content="anything")
        self.assertFalse(result)

    def test_delete_fact_removes_row_and_returns_true(self):
        # delete_fact should remove the row from the database and return True.
        fact_id = insert_fact(self.conn, "a fact to delete")
        result = delete_fact(self.conn, fact_id)

        # True means the row was found and deleted.
        self.assertTrue(result)

        # The row must no longer exist in the database.
        row = self.conn.execute(
            "SELECT id FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        self.assertIsNone(row)

    def test_delete_fact_returns_false_for_unknown_id(self):
        # Deleting a non-existent fact must return False — not raise an exception.
        result = delete_fact(self.conn, "does-not-exist")
        self.assertFalse(result)

    def test_search_facts_finds_keyword_in_content(self):
        # search_facts should locate a fact by a keyword that appears in its content.
        insert_fact(self.conn, "Python uses indentation for blocks.", tags=["python"])
        insert_fact(self.conn, "Rust is a systems programming language.")

        # "indentation" only appears in the first fact.
        results = search_facts(self.conn, "indentation")

        self.assertEqual(len(results), 1)
        self.assertIn("indentation", results[0]["content"])

    def test_search_facts_result_has_required_keys(self):
        # Every search result dict must carry all the keys callers depend on.
        insert_fact(self.conn, "Go is statically typed.", tags=["go"])
        results = search_facts(self.conn, "statically")

        self.assertEqual(len(results), 1)
        for key in ("id", "content", "tags", "source", "session_id",
                    "created_at", "updated_at", "snippet"):
            self.assertIn(key, results[0], f"missing key: {key}")

    def test_list_facts_no_tag_returns_all(self):
        # list_facts with no tag filter must return every stored fact.
        insert_fact(self.conn, "fact one", tags=["a"])
        insert_fact(self.conn, "fact two", tags=["b"])
        insert_fact(self.conn, "fact three")

        results = list_facts(self.conn)
        self.assertEqual(len(results), 3)

    def test_list_facts_with_tag_returns_only_matching(self):
        # list_facts filtered by tag must exclude facts that do not carry that tag.
        insert_fact(self.conn, "tagged with a and common", tags=["a", "common"])
        insert_fact(self.conn, "tagged with b and common", tags=["b", "common"])
        insert_fact(self.conn, "tagged with only a", tags=["a"])

        # Filter by "a" — should match 2 facts ("a and common", "only a").
        results = list_facts(self.conn, tag="a")
        self.assertEqual(len(results), 2)

        # Every returned fact must carry the "a" tag.
        for r in results:
            self.assertIn("a", r["tags"])

    def test_tags_round_trip_list_in_list_out(self):
        # Tags passed as a Python list must come back as a Python list — not a JSON string.
        tags_in = ["python", "databases", "sql"]
        fact_id = insert_fact(self.conn, "tags round-trip test", tags=tags_in)

        # list_facts uses the same JSON-parsing code as search_facts.
        results = list_facts(self.conn)
        self.assertEqual(len(results), 1)
        # Sort both sides so the comparison is order-independent.
        self.assertEqual(sorted(results[0]["tags"]), sorted(tags_in))

    def test_no_tags_defaults_to_empty_list(self):
        # When no tags are provided, the returned dict must have an empty list — not None.
        insert_fact(self.conn, "fact with no tags")
        results = list_facts(self.conn)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tags"], [])


# ---------------------------------------------------------------------------
# Test group: hybrid_search — Reciprocal Rank Fusion of FTS5 + semantic
# ---------------------------------------------------------------------------

class TestHybridSearch(unittest.TestCase):
    """
    Tests for hybrid_search() in memory/db.py.

    Because sentence-transformers may not be installed in CI, these tests
    treat the semantic leg as optional. The mock-based tests cover the case
    where it IS available; the fallback test covers the case where it is NOT.
    """

    def setUp(self):
        # A fresh in-memory database is created before each test.
        # This keeps tests fully isolated — writes in one test do not affect another.
        self.conn = init_db(":memory:")

    # ---- helper -------------------------------------------------------

    def _insert(self, session_id, text, agent="claude",
                 started="2026-01-01T00:00:00Z", updated="2026-01-01T00:01:00Z"):
        """
        Insert a minimal session with the given text as its transcript content.
        The transcript format is a list of turn dicts — we keep it simple here.
        """
        transcript = [{"role": "user", "content": text}]
        upsert_session(self.conn, session_id, agent, transcript, started, updated)

    # ---- tests --------------------------------------------------------

    def test_hybrid_prefers_both_match(self):
        # Insert two sessions. "quantum entanglement physics" is specific enough
        # that it should appear in FTS5 results for "quantum". A session that
        # matches only via keywords (not semantically) ranks lower overall.
        #
        # We mock semantic_search_chunks so the test does not require the
        # sentence-transformers library and is deterministic.
        import unittest.mock as mock

        # Session A: matches on keyword "quantum".
        # Session B: matches on keyword "quantum" too, but ranked lower in FTS5.
        self._insert("s-both",  "quantum entanglement physics")
        self._insert("s-kw",    "quantum mechanics theory")

        # Semantic results: only "s-both" appears (rank 0).
        # This simulates a session that matches on both keyword and meaning.
        fake_sem = [
            {
                "session_id":  "s-both",
                "chunk_index": 0,
                "distance":    0.1,       # low distance = close semantic match
                "text":        "quantum entanglement physics",
                "snippet":     "quantum entanglement physics",
            }
        ]

        # Patch semantic_search_chunks inside the memory.db module so our call
        # to hybrid_search uses the fake results instead of the real embedder.
        with mock.patch("memory.db.semantic_search_chunks", return_value=fake_sem):
            results = hybrid_search(self.conn, "quantum", limit=10)

        # "s-both" must appear before "s-kw" because it got contributions from
        # BOTH the FTS5 rank and the semantic rank.
        self.assertGreater(len(results), 0)
        session_ids = [r["session_id"] for r in results]
        self.assertIn("s-both", session_ids)

        # s-both must have a strictly higher rrf_score than s-kw (if s-kw appears).
        both_score = next(r["rrf_score"] for r in results if r["session_id"] == "s-both")
        kw_scores  = [r["rrf_score"] for r in results if r["session_id"] == "s-kw"]
        if kw_scores:
            # "s-kw" only gets FTS5 contribution; "s-both" gets FTS5 + semantic.
            self.assertGreater(both_score, kw_scores[0])

    def test_hybrid_deduplicates_sessions(self):
        # Insert a handful of sessions and check that no session_id appears twice.
        for i in range(5):
            self._insert(f"sess-{i}", f"python programming topic {i}")

        # Run hybrid_search — dedup must happen even if the same session_id appears
        # in both result lists.
        import unittest.mock as mock

        # Fake semantic results: deliberately repeat "sess-0" as two chunks.
        # After dedup, only one entry for "sess-0" should survive.
        fake_sem = [
            {"session_id": "sess-0", "chunk_index": 0, "distance": 0.1,
             "text": "python programming topic 0", "snippet": "topic 0"},
            {"session_id": "sess-0", "chunk_index": 1, "distance": 0.2,
             "text": "python programming topic 0b", "snippet": "topic 0b"},
        ]
        with mock.patch("memory.db.semantic_search_chunks", return_value=fake_sem):
            results = hybrid_search(self.conn, "python programming", limit=10)

        # Extract all session_id values and check for duplicates.
        session_ids = [r["session_id"] for r in results]
        # A set has no duplicates; if the list length equals the set length,
        # there are no duplicates.
        self.assertEqual(len(session_ids), len(set(session_ids)))

    def test_hybrid_fallback_fts_only(self):
        # When semantic_search_chunks raises ImportError (sentence-transformers
        # not installed), hybrid_search must still return FTS5 results.
        import unittest.mock as mock

        self._insert("fts-only-session", "machine learning neural network")

        # Simulate the ImportError that happens when sentence-transformers is absent.
        with mock.patch(
            "memory.db.semantic_search_chunks",
            side_effect=ImportError("sentence-transformers not installed"),
        ):
            results = hybrid_search(self.conn, "machine learning", limit=10)

        # The function must return results from FTS5 without raising.
        self.assertIsInstance(results, list)
        # "fts-only-session" must be in the results because it matched on keywords.
        session_ids = [r["session_id"] for r in results]
        self.assertIn("fts-only-session", session_ids)

    def test_hybrid_returns_limit(self):
        # Insert more sessions than the limit so we can verify truncation.
        for i in range(10):
            self._insert(f"limit-sess-{i}", f"database indexing query optimization topic {i}")

        import unittest.mock as mock

        # No semantic results — clean FTS5-only test.
        with mock.patch("memory.db.semantic_search_chunks", return_value=[]):
            results = hybrid_search(self.conn, "database indexing query", limit=3)

        # Exactly 3 results must be returned, not more.
        self.assertEqual(len(results), 3)

    def test_hybrid_snippet_present(self):
        # Every result dict must have a "snippet" key with a non-empty string.
        self._insert("snippet-sess", "photosynthesis light chlorophyll plants")

        import unittest.mock as mock

        with mock.patch("memory.db.semantic_search_chunks", return_value=[]):
            results = hybrid_search(self.conn, "photosynthesis", limit=5)

        # At least one result must be present (the session we inserted matches).
        self.assertGreater(len(results), 0)
        for r in results:
            # Every result must have a "snippet" key.
            self.assertIn("snippet", r)
            # The snippet must be a non-empty string — not None and not "".
            self.assertIsInstance(r["snippet"], str)
            self.assertGreater(len(r["snippet"]), 0)


if __name__ == "__main__":
    unittest.main()
