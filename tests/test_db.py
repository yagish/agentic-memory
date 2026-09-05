import json
import os
import sqlite3
import tempfile
import unittest

from memory.db import (
    bootstrap_db,
    ensure_schema,
    init_db,
    insert_episodic,
    insert_fact,
    log_retrieval,
    open_db,
    search,
    search_episodic_semantic,
    search_facts_semantic,
    semantic_search,
    upsert_session,
)
from memory.vectors import embed


class TestConnectionBootstrapSplit(unittest.TestCase):
    def test_open_db_does_not_apply_schema(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        try:
            conn = open_db(path)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            conn.close()
        finally:
            os.unlink(path)

    def test_bootstrap_db_creates_schema(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        try:
            conn = bootstrap_db(path)
            try:
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                self.assertTrue({"sessions", "facts", "episodic_memory"}.issubset(tables))
                self.assertTrue({"sessions", "facts", "episodic_memory"}.issubset(tables))
            finally:
                conn.close()
        finally:
            os.unlink(path)


class TestSessionStorage(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        self.transcript = [
            {"role": "user", "content": "Tell me about rockets."},
            {"role": "assistant", "content": "Rockets expel mass."},
        ]

    def tearDown(self):
        self.conn.close()

    def test_upsert_and_search(self):
        upsert_session(
            self.conn,
            "s1",
            "claude",
            self.transcript,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            metadata={"project": "space"},
        )
        row = self.conn.execute(
            "SELECT transcript, turn_count, metadata FROM sessions WHERE session_id = 's1'"
        ).fetchone()
        self.assertEqual(json.loads(row["transcript"]), self.transcript)
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual(json.loads(row["metadata"]), {"project": "space"})

        results = search(self.conn, "rockets")
        self.assertEqual([item["session_id"] for item in results], ["s1"])

    def test_semantic_search_returns_ranked_rows(self):
        upsert_session(
            self.conn,
            "space",
            "claude",
            [
                {"role": "user", "content": "Tell me about rocket propulsion and orbit transfers."},
                {"role": "assistant", "content": "Orbital mechanics explains transfer burns."},
            ],
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
        )
        upsert_session(
            self.conn,
            "cooking",
            "claude",
            [
                {"role": "user", "content": "How do I cook pasta carbonara?"},
                {"role": "assistant", "content": "Use eggs, cheese, and pepper."},
            ],
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:01:00Z",
        )

        results = semantic_search(self.conn, "rocket launch orbit", limit=2)
        self.assertEqual(results[0]["session_id"], "space")
        self.assertLessEqual(results[0]["distance"], results[1]["distance"])


class TestFactAndEpisodeStorage(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_semantic_fact_search(self):
        insert_fact(self.conn, "user.name = Yash", tags=["identity"], session_id="s1")
        insert_fact(self.conn, "project.language = Python", tags=["project"], session_id="s1")
        results = search_facts_semantic(self.conn, embed("what is the user's name"), limit=2)
        self.assertEqual(results[0]["content"], "user.name = Yash")

    def test_semantic_episode_search(self):
        insert_episodic(
            self.conn,
            session_id="s1",
            title="Fixed login loop",
            abstract="Moved token validation into middleware and fixed the loop.",
            happened_at="2026-01-01T00:00:00Z",
            details={"follow_ups": ["Add regression tests"]},
            embedding=embed("Fixed login loop middleware"),
        )
        insert_episodic(
            self.conn,
            session_id="s2",
            title="Planned pasta dinner",
            abstract="Chose a carbonara recipe.",
            happened_at="2026-01-02T00:00:00Z",
            embedding=embed("Pasta carbonara recipe"),
        )
        results = search_episodic_semantic(self.conn, embed("login middleware bug"), limit=2)
        self.assertEqual(results[0]["title"], "Fixed login loop")


class TestCompatibilityNoops(unittest.TestCase):
    def test_log_retrieval_is_noop(self):
        conn = init_db(":memory:")
        try:
            log_retrieval(conn, "wake_up", "query", 123)
            row = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='retrievals'"
            ).fetchone()
            self.assertEqual(row[0], 0)
        finally:
            conn.close()



if __name__ == "__main__":
    unittest.main()
