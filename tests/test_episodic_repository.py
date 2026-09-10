import unittest

from memory.contracts import ExtractedEpisode
from memory.db import init_db
from memory.episodic.repository import (
    build_episodic_semantic_text,
    list_session_episodes,
    retrieve_episodic_memories,
    save_extracted_episode,
)


class TestEpisodicRepository(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_build_semantic_text_is_deterministic(self):
        episode = ExtractedEpisode(
            title="Auth middleware decision",
            abstract="Moved token validation into shared middleware. Login loop is fixed.",
            participants=("pi", "claude"),
            decisions=("Move token validation into shared middleware",),
            outcomes=("Login loop fixed",),
            follow_ups=("Add regression tests",),
        )

        semantic_text = build_episodic_semantic_text(episode)
        self.assertIn("Auth middleware decision", semantic_text)
        self.assertIn("Participants: pi; claude", semantic_text)
        self.assertIn("Decisions: Move token validation into shared middleware", semantic_text)
        self.assertIn("Outcomes: Login loop fixed", semantic_text)
        self.assertIn("Follow-ups: Add regression tests", semantic_text)

    def test_save_extracted_episode_persists_structured_episode(self):
        episode = ExtractedEpisode(
            title="Auth middleware decision",
            abstract="Moved token validation into shared middleware. Login loop is fixed.",
            decisions=("Move token validation into shared middleware",),
            outcomes=("Login loop fixed",),
            follow_ups=("Add regression tests",),
            confidence=0.9,
        )

        saved_id = save_extracted_episode(
            self.conn,
            episode,
            session_id="session-123",
            happened_at="2026-09-04T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
            source="episodic_test",
        )

        rows = list_session_episodes(self.conn, session_id="session-123")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], saved_id)
        self.assertEqual(rows[0]["title"], "Auth middleware decision")
        self.assertEqual(rows[0]["abstract"], "Moved token validation into shared middleware. Login loop is fixed.")
        self.assertEqual(rows[0]["decisions"], ["Move token validation into shared middleware"])
        self.assertEqual(rows[0]["outcomes"], ["Login loop fixed"])
        self.assertEqual(rows[0]["follow_ups"], ["Add regression tests"])
        self.assertEqual(rows[0]["source"], "episodic_test")
        self.assertIn("Decisions: Move token validation into shared middleware", rows[0]["semantic_text"])

    def test_retrieve_episodic_memories_filters_by_similarity(self):
        save_extracted_episode(
            self.conn,
            ExtractedEpisode(
                title="Auth middleware decision",
                abstract="Moved token validation into shared middleware. Login loop is fixed.",
            ),
            session_id="session-auth",
            happened_at="2026-09-04T10:00:00Z",
            embed_fn=lambda text: [1.0, 0.0],
        )
        save_extracted_episode(
            self.conn,
            ExtractedEpisode(
                title="Billing cleanup",
                abstract="Refactored invoice formatting and updated copy.",
            ),
            session_id="session-billing",
            happened_at="2026-09-04T10:05:00Z",
            embed_fn=lambda text: [0.0, 1.0],
        )

        results = retrieve_episodic_memories(
            self.conn,
            "what did we decide about auth middleware?",
            embed_fn=lambda prompt: [1.0, 0.0],
            min_similarity=0.72,
            source="test-harness",
        )

        self.assertEqual([row["session_id"] for row in results], ["session-auth"])
        self.assertEqual(results[0]["title"], "Auth middleware decision")
        self.assertGreaterEqual(results[0]["similarity"], 0.99)


if __name__ == "__main__":
    unittest.main()
