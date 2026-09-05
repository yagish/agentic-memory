import unittest

from memory.db import init_db, semantic_search, upsert_session
from memory.vectors import embed


class TestEmbed(unittest.TestCase):
    def test_returns_list_of_floats(self):
        vector = embed("Hello world")
        self.assertIsInstance(vector, list)
        self.assertTrue(vector)
        self.assertTrue(all(isinstance(value, float) for value in vector))

    def test_is_deterministic(self):
        self.assertEqual(embed("test sentence"), embed("test sentence"))


class TestSemanticSearch(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        upsert_session(
            self.conn,
            "space-session",
            "claude",
            [
                {"role": "user", "content": "Tell me about rocket propulsion and orbital mechanics."},
                {"role": "assistant", "content": "Rockets work by expelling mass at high velocity."},
            ],
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
        )
        upsert_session(
            self.conn,
            "cooking-session",
            "claude",
            [
                {"role": "user", "content": "How do I make a good pasta carbonara?"},
                {"role": "assistant", "content": "Use eggs, pecorino, guanciale, and pepper."},
            ],
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:01:00Z",
        )

    def tearDown(self):
        self.conn.close()

    def test_space_query_ranks_space_session_first(self):
        results = semantic_search(self.conn, "rocket launch orbital mechanics", limit=2)
        self.assertEqual(results[0]["session_id"], "space-session")

    def test_cooking_query_ranks_cooking_session_first(self):
        results = semantic_search(self.conn, "pasta recipe eggs cheese", limit=2)
        self.assertEqual(results[0]["session_id"], "cooking-session")

    def test_results_are_sorted_by_distance(self):
        results = semantic_search(self.conn, "food cooking", limit=5)
        distances = [row["distance"] for row in results]
        self.assertEqual(distances, sorted(distances))


if __name__ == "__main__":
    unittest.main()
