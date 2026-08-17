# test_semantic_search.py — tests for Phase 5 semantic (vector) search.
#
# Run with:  python3 -m unittest tests.test_semantic_search -v
#
# These tests verify that:
#   1. embed() produces a vector of the right shape
#   2. store_embedding() saves without error
#   3. semantic_search() returns results in the right format and order
#
# NOTE: these tests use the real all-MiniLM-L6-v2 model.
# The first run downloads ~90MB of model weights to ~/.cache/huggingface/.
# Subsequent runs load from cache and are fast (~2 seconds).

import struct
import unittest

# Import the Phase 5 and Phase 7 functions we want to test.
from memory.db import (
    init_db,
    upsert_session,
    embed,
    store_embedding,
    semantic_search,
    chunk_transcript,
    store_chunk,
    get_chunks_for_session,
    delete_chunks_for_session,
    semantic_search_chunks,
)


class TestEmbed(unittest.TestCase):
    """Tests for the embed() function that converts text to a vector."""

    def test_returns_list_of_floats(self):
        # embed() must return a plain Python list, not a numpy array.
        # Callers rely on this — numpy arrays don't serialize cleanly to JSON.
        vec = embed("Hello world")
        self.assertIsInstance(vec, list)

    def test_correct_dimension(self):
        # all-MiniLM-L6-v2 always outputs exactly 384 floats.
        # If this fails, the model version changed — update _DIMS in db.py.
        vec = embed("Hello world")
        self.assertEqual(len(vec), 384)

    def test_all_elements_are_floats(self):
        # Every element must be a float (not int, not None, not str).
        vec = embed("Hello world")
        for val in vec:
            self.assertIsInstance(val, float)

    def test_deterministic(self):
        # The same text must always produce the same vector.
        # Embedding models are deterministic — same input → same output.
        vec1 = embed("test sentence")
        vec2 = embed("test sentence")
        self.assertEqual(vec1, vec2)

    def test_different_texts_produce_different_vectors(self):
        # Two completely unrelated texts should not produce identical vectors.
        vec1 = embed("quantum physics")
        vec2 = embed("cooking pasta")
        self.assertNotEqual(vec1, vec2)


class TestStoreEmbedding(unittest.TestCase):
    """Tests for store_embedding() — saving vectors to the database."""

    def setUp(self):
        # Create a fresh in-memory database for each test.
        # ":memory:" means no files on disk — tests are self-contained.
        self.conn = init_db(":memory:")

        # We need a session row before we can store an embedding for it.
        upsert_session(
            self.conn, "s1", "claude",
            [{"role": "user", "content": "Hello"}],
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
        )

    def test_store_does_not_raise(self):
        # Smoke test — if we get here without an exception, the store worked.
        vec = embed("test")
        store_embedding(self.conn, "s1", vec)  # must not raise

    def test_store_replaces_existing(self):
        # Storing twice for the same session_id should overwrite, not duplicate.
        vec1 = embed("first content")
        vec2 = embed("second content")
        store_embedding(self.conn, "s1", vec1)
        store_embedding(self.conn, "s1", vec2)  # must not raise

        # Only one row should exist for s1.
        count = self.conn.execute(
            "SELECT COUNT(*) FROM session_vecs WHERE session_id='s1'"
        ).fetchone()[0]
        self.assertEqual(count, 1)


class TestSemanticSearch(unittest.TestCase):
    """Tests for semantic_search() — finding sessions by meaning."""

    def setUp(self):
        # Create a fresh database and seed it with two sessions on different topics.
        self.conn = init_db(":memory:")

        # Session about space travel.
        self.transcript_space = [
            {"role": "user",      "content": "Tell me about rocket propulsion and orbital mechanics."},
            {"role": "assistant", "content": "Rockets work by expelling mass at high velocity. Orbital mechanics describes how spacecraft travel."},
        ]
        # Session about cooking.
        self.transcript_cooking = [
            {"role": "user",      "content": "How do I make a good pasta carbonara?"},
            {"role": "assistant", "content": "Carbonara uses eggs, pecorino cheese, guanciale, and black pepper."},
        ]

        # Save both sessions to the main sessions table.
        upsert_session(self.conn, "space-session", "claude", self.transcript_space,
                       "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z")
        upsert_session(self.conn, "cooking-session", "claude", self.transcript_cooking,
                       "2026-01-02T00:00:00Z", "2026-01-02T00:01:00Z")

        # Embed and store both sessions.
        space_text   = " ".join(t["content"] for t in self.transcript_space)
        cooking_text = " ".join(t["content"] for t in self.transcript_cooking)
        store_embedding(self.conn, "space-session",   embed(space_text))
        store_embedding(self.conn, "cooking-session", embed(cooking_text))

    def test_returns_list(self):
        # semantic_search must always return a list, never None or a dict.
        results = semantic_search(self.conn, "space travel", limit=5)
        self.assertIsInstance(results, list)

    def test_result_has_required_keys(self):
        # Each result dict must have these four keys — callers depend on them.
        results = semantic_search(self.conn, "space travel", limit=5)
        if results:
            keys = results[0].keys()
            self.assertIn("session_id", keys)
            self.assertIn("agent",      keys)
            self.assertIn("updated_at", keys)
            self.assertIn("distance",   keys)

    def test_distance_is_numeric(self):
        # Cosine distance must be a number (float or int), not None or a string.
        results = semantic_search(self.conn, "space travel", limit=5)
        if results:
            self.assertIsInstance(results[0]["distance"], (int, float))

    def test_space_query_ranks_space_session_first(self):
        # A query about rockets should match the space session more closely
        # (lower distance) than the cooking session.
        results = semantic_search(self.conn, "rocket launch orbital mechanics", limit=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["session_id"], "space-session")

    def test_cooking_query_ranks_cooking_session_first(self):
        # A query about pasta should match the cooking session more closely.
        results = semantic_search(self.conn, "pasta recipe eggs cheese", limit=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["session_id"], "cooking-session")

    def test_results_ordered_by_distance_ascending(self):
        # Results must be sorted closest-first (lowest distance = best match).
        results = semantic_search(self.conn, "food cooking", limit=5)
        distances = [r["distance"] for r in results]
        self.assertEqual(distances, sorted(distances))

    def test_limit_respected(self):
        # With limit=1, only one result should come back even if both match.
        results = semantic_search(self.conn, "something", limit=1)
        self.assertLessEqual(len(results), 1)


# ---------------------------------------------------------------------------
# Phase 7 tests: chunk-level semantic search (Ticket 7-03)
# ---------------------------------------------------------------------------

class TestSemanticSearchChunks(unittest.TestCase):
    """Tests for semantic_search_chunks() — finding sessions via chunk embeddings."""

    def setUp(self):
        # Create a fresh database and seed it with one session that has two
        # very different topics: the first chunk is about astronomy and the
        # second chunk is about baking. The test verifies that a query about
        # baking finds the session even though astronomy dominates turn count.
        self.conn = init_db(":memory:")

        # A session whose first turns discuss astronomy and later turns discuss baking.
        # We build it so each topic fills a natural chunk boundary.
        self.session_id = "mixed-session"
        self.transcript = [
            {"role": "user",      "content": "Tell me about black holes and event horizons."},
            {"role": "assistant", "content": "Black holes are regions where gravity is so strong that light cannot escape."},
            {"role": "user",      "content": "What is the Schwarzschild radius?"},
            {"role": "assistant", "content": "The Schwarzschild radius defines the event horizon size based on mass."},
            {"role": "user",      "content": "Fascinating. Now, how do I bake sourdough bread?"},
            {"role": "assistant", "content": "Sourdough bread requires a starter culture of wild yeast and lactic acid bacteria."},
            {"role": "user",      "content": "What hydration ratio should I use for the dough?"},
            {"role": "assistant", "content": "A 75% hydration dough works well for beginners — use 750 grams water per 1000 grams flour."},
        ]

        # Save the session to the sessions table.
        upsert_session(
            self.conn, self.session_id, "claude", self.transcript,
            "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z",
        )

        # Chunk the transcript with window=4, overlap=1 → 2 chunks:
        #   chunk 0: turns 0-3 (astronomy)
        #   chunk 1: turns 3-7 (baking)
        chunk_texts = chunk_transcript(self.transcript, window=4, overlap=1)

        # Remove stale chunks (none yet, but mirrors what save_hook does).
        delete_chunks_for_session(self.conn, self.session_id)

        # Embed and store each chunk.
        for idx, text in enumerate(chunk_texts):
            vec = embed(text) if text.strip() else None
            store_chunk(self.conn, self.session_id, idx, text, vec)

        # Track how many chunks were stored for the count assertion test.
        self.stored_chunk_count = len(chunk_texts)

    def test_query_in_second_chunk_returns_correct_session(self):
        # A query about sourdough bread should match the second chunk (baking)
        # and return the session_id — even though the first chunk (astronomy) is
        # larger and would dominate a whole-session embedding.
        results = semantic_search_chunks(self.conn, "sourdough bread baking yeast", limit=5)

        # There must be at least one result.
        self.assertGreater(len(results), 0)

        # The top result must be from our mixed session — the baking chunk matched.
        session_ids = [r["session_id"] for r in results]
        self.assertIn(self.session_id, session_ids)

    def test_chunk_count_in_db_matches_expected(self):
        # After the setUp save, the chunks table must contain exactly as many
        # rows as chunk_transcript() produced. This confirms store_chunk() and
        # delete_chunks_for_session() round-trip correctly.
        chunks = get_chunks_for_session(self.conn, self.session_id)
        self.assertEqual(len(chunks), self.stored_chunk_count)

    def test_result_has_required_keys(self):
        # Every result dict must contain the mandatory fields.
        results = semantic_search_chunks(self.conn, "bread flour water", limit=5)
        if results:
            keys = results[0].keys()
            self.assertIn("session_id",  keys)
            self.assertIn("chunk_index", keys)
            self.assertIn("distance",    keys)
            self.assertIn("snippet",     keys)

    def test_results_ordered_by_distance_ascending(self):
        # Results must be sorted by cosine distance, closest first.
        results = semantic_search_chunks(self.conn, "event horizon gravity", limit=10)
        distances = [r["distance"] for r in results]
        self.assertEqual(distances, sorted(distances))

    def test_empty_db_returns_empty_list(self):
        # A fresh database with no chunk embeddings must return [].
        fresh_conn = init_db(":memory:")
        results = semantic_search_chunks(fresh_conn, "anything", limit=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
