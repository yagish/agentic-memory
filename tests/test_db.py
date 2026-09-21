import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from memory.db import (
    bootstrap_db,
    ensure_schema,
    init_db,
    insert_episodic,
    insert_fact,
    insert_procedural,
    log_retrieval,
    open_db,
    get_session_by_id,
    prune_stale_session_memory,
    prune_stale_working_memory,
    search,
    search_episodic_fts,
    search_session_fts,
    search_episodic_semantic,
    search_facts_semantic,
    search_procedural_fts,
    search_procedural_semantic,
    search_session_memory_fts,
    search_session_memory_semantic,
    semantic_search,
    upsert_session,
    upsert_session_memory,
    upsert_working_memory,
)
from memory.vectors import embed
from scripts.migrate_facts_semantic_text import migrate_fact_schema


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
                self.assertTrue({"sessions", "facts", "episodic_memory", "procedural_memory", "working_memory", "session_memory"}.issubset(tables))
            finally:
                conn.close()
        finally:
            os.unlink(path)


class TestFactsSchemaMigration(unittest.TestCase):
    @patch("scripts.migrate_facts_semantic_text.generate_semantic_fact_text", return_value="My name is Yash. What's my name? Yash.")
    def test_external_migration_rebuilds_facts_without_content_column(self, _mock_semantic_text):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.executescript(
                """
                CREATE TABLE facts (
                  id TEXT PRIMARY KEY,
                  content TEXT NOT NULL,
                  tags TEXT,
                  source TEXT,
                  session_id TEXT,
                  created_at TEXT,
                  updated_at TEXT,
                  embedding BLOB
                );
                """
            )
            conn.execute(
                "INSERT INTO facts (id, content, tags, source, session_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("fact-1", "user.name = Yash", "[]", "manual", "s1", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )
            conn.commit()
            conn.close()

            migrate_fact_schema(path)

            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            columns = [r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()]
            self.assertEqual(
                columns,
                ["id", "entity", "attribute", "value", "semantic_content", "tags", "source", "session_id", "created_at", "updated_at", "embedding"],
            )
            row = conn.execute(
                "SELECT semantic_content, entity, attribute, value, embedding FROM facts WHERE id = ?",
                ("fact-1",),
            ).fetchone()
            self.assertEqual(row["semantic_content"], "My name is Yash. What's my name? Yash.")
            self.assertEqual(row["entity"], "user")
            self.assertEqual(row["attribute"], "name")
            self.assertEqual(row["value"], "Yash")
            self.assertIsNotNone(row["embedding"])
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

    def test_session_project_context_backfills_from_metadata(self):
        upsert_session(
            self.conn,
            "legacy-project-session",
            "claude",
            self.transcript,
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            metadata={
                "cwd": "/tmp/demo-repo/app",
                "repo_root": "/tmp/demo-repo",
                "git_remote": "git@github.com:example/demo.git",
                "branch": "main",
            },
        )
        session = get_session_by_id(self.conn, "legacy-project-session")
        self.assertEqual(session["project_id"], "git@github.com:example/demo.git")
        self.assertEqual(session["repo_root"], "/tmp/demo-repo")
        self.assertEqual(session["cwd"], "/tmp/demo-repo/app")
        self.assertEqual(session["git_remote"], "git@github.com:example/demo.git")
        self.assertEqual(session["git_branch"], "main")

    def test_session_fts_search_indexes_transcript_text_without_returning_snippets(self):
        upsert_session(
            self.conn,
            "s1",
            "claude",
            [
                {"role": "user", "content": "Need to resume the frostbyte rollout."},
                {"role": "assistant", "content": "We paused before the gate check."},
            ],
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
        )

        results = search_session_fts(self.conn, "frostbyte", limit=2)
        self.assertEqual([item["session_id"] for item in results], ["s1"])
        self.assertTrue(results[0]["keyword_hit"])
        self.assertNotIn("snippet", results[0])
        self.assertNotIn("transcript", results[0])

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
        insert_fact(
            self.conn,
            entity="user",
            attribute="name",
            value="Yash",
            semantic_content="My name is Yash. What's my name? Yash.",
            tags=["identity"],
            session_id="s1",
        )
        insert_fact(
            self.conn,
            entity="project",
            attribute="language",
            value="Python",
            semantic_content="The project's language is Python.",
            tags=["project"],
            session_id="s1",
        )
        results = search_facts_semantic(self.conn, embed("what is my name"), limit=2)
        self.assertEqual(results[0]["content"], "user.name = Yash")
        self.assertEqual(results[0]["semantic_content"], "My name is Yash. What's my name? Yash.")
        self.assertGreaterEqual(results[0]["similarity"], 0.38)
        self.assertEqual(results[0]["entity"], "user")
        self.assertEqual(results[0]["attribute"], "name")
        self.assertEqual(results[0]["value"], "Yash")

    def test_semantic_fact_search_handles_natural_language_queries_without_low_threshold(self):
        insert_fact(self.conn, entity="user", attribute="name", value="Yash", semantic_content="My name is Yash. What's my name? Yash.", session_id="s1")
        insert_fact(self.conn, entity="user", attribute="location", value="West Chester, OH", semantic_content="I live in West Chester, OH. Where do I live? West Chester, OH.", session_id="s1")
        insert_fact(self.conn, entity="user", attribute="company", value="Kroger", semantic_content="I work at Kroger. What company do I work at? Kroger.", session_id="s1")

        cases = [
            ("whats my name", "user.name = Yash"),
            ("where do i live", "user.location = West Chester, OH"),
            ("what company do i work at", "user.company = Kroger"),
        ]

        for query, expected_content in cases:
            with self.subTest(query=query):
                results = search_facts_semantic(self.conn, embed(query), limit=3)
                self.assertEqual(results[0]["content"], expected_content)
                self.assertGreaterEqual(results[0]["similarity"], 0.38)

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

    def test_episode_fts_search_indexes_anchor_fields(self):
        insert_episodic(
            self.conn,
            session_id="s1",
            title="Release decision",
            abstract="Captured the release decision.",
            happened_at="2026-01-01T00:00:00Z",
            details={"decisions": ["Enable frostbyte gate after smoke tests"]},
            embedding=embed("release decision"),
        )

        results = search_episodic_fts(self.conn, "frostbyte", limit=2)
        self.assertEqual(results[0]["title"], "Release decision")
        self.assertTrue(results[0]["keyword_hit"])

    def test_semantic_procedural_search(self):
        insert_procedural(
            self.conn,
            session_id="s1",
            title="Web deploy workflow",
            summary="Use this when deploying the web service.",
            updated_at="2026-01-01T00:00:00Z",
            details={"steps": ["Build the Docker image", "Run alembic upgrade"]},
            embedding=embed("deploy web service docker alembic staging production"),
        )
        insert_procedural(
            self.conn,
            session_id="s2",
            title="Invoice copy workflow",
            summary="Use this when updating billing email copy.",
            updated_at="2026-01-02T00:00:00Z",
            details={"steps": ["Open the billing templates", "Edit the copy"]},
            embedding=embed("billing email copy templates"),
        )
        results = search_procedural_semantic(self.conn, embed("how do i deploy the web service"), limit=2)
        self.assertEqual(results[0]["title"], "Web deploy workflow")

    def test_procedural_fts_search_indexes_step_anchors(self):
        insert_procedural(
            self.conn,
            session_id="s1",
            title="Ops checklist",
            summary="Operational checklist.",
            updated_at="2026-01-01T00:00:00Z",
            details={"steps": ["Run quartzsync before cutover"]},
            embedding=embed("ops checklist"),
        )

        results = search_procedural_fts(self.conn, "quartzsync", limit=2)
        self.assertEqual(results[0]["title"], "Ops checklist")
        self.assertTrue(results[0]["keyword_hit"])

    def test_working_memory_upsert_replaces_existing_session_snapshot(self):
        first_id = upsert_working_memory(
            self.conn,
            session_id="s1",
            current_goal="Finish auth middleware refactor",
            current_focus="Regression coverage",
            next_step="Write refresh-token tests",
            status="in_progress",
            updated_at="2026-01-01T00:00:00Z",
            details={"active_tasks": ["Add refresh-token tests"]},
            embedding=[1.0, 0.0],
        )
        second_id = upsert_working_memory(
            self.conn,
            session_id="s1",
            current_goal="Finish auth middleware refactor",
            current_focus="Expired-session coverage",
            next_step="Write expired-session tests",
            status="ready_to_resume",
            updated_at="2026-01-01T00:05:00Z",
            details={"active_tasks": ["Add expired-session tests"]},
            embedding=[1.0, 0.0],
        )

        self.assertEqual(first_id, second_id)
        rows = self.conn.execute("SELECT id, current_focus, status FROM working_memory WHERE session_id = ?", ("s1",)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["current_focus"], "Expired-session coverage")
        self.assertEqual(rows[0]["status"], "ready_to_resume")

    def test_semantic_session_memory_search(self):
        upsert_session_memory(
            self.conn,
            session_id="s1",
            title="Auth middleware refactor",
            summary="Moved token validation into shared middleware and fixed the login redirect loop locally.",
            left_off_at="Regression coverage is still missing for refresh-token and expired-session flows",
            updated_at="2026-01-01T00:00:00Z",
            details={"next_steps": ["Add regression coverage"]},
            embedding=embed("auth middleware refactor login redirect loop refresh token expired session"),
        )
        upsert_session_memory(
            self.conn,
            session_id="s2",
            title="Billing copy update",
            summary="Adjusted invoice wording for support.",
            left_off_at="Awaiting support review",
            updated_at="2026-01-02T00:00:00Z",
            details={"next_steps": ["Wait for support review"]},
            embedding=embed("billing copy invoice wording support review"),
        )

        results = search_session_memory_semantic(self.conn, embed("pick up auth middleware refactor where I left off"), limit=2)
        self.assertEqual(results[0]["title"], "Auth middleware refactor")

    def test_session_memory_fts_search_indexes_next_steps(self):
        upsert_session_memory(
            self.conn,
            session_id="s1",
            title="Handoff",
            summary="Paused the rollout.",
            left_off_at="Waiting for the next session.",
            updated_at="2026-01-01T00:00:00Z",
            details={"next_steps": ["Resume after lunarflag approval"]},
            embedding=embed("handoff rollout"),
        )

        results = search_session_memory_fts(self.conn, "lunarflag", limit=2)
        self.assertEqual(results[0]["title"], "Handoff")
        self.assertTrue(results[0]["keyword_hit"])


class TestShortTermMemoryPruning(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_working_memory_prune_deletes_only_rows_older_than_ttl(self):
        upsert_working_memory(
            self.conn,
            session_id="stale-session",
            current_goal="Finish auth refactor",
            current_focus="Old focus",
            next_step="Resume later",
            status="paused",
            updated_at="2026-01-01T00:00:00Z",
            details={"active_tasks": ["old-task"]},
        )
        upsert_working_memory(
            self.conn,
            session_id="fresh-session",
            current_goal="Finish auth refactor",
            current_focus="Fresh focus",
            next_step="Write tests",
            status="in_progress",
            updated_at="2026-01-20T00:00:00Z",
            details={"active_tasks": ["fresh-task"]},
        )

        deleted = prune_stale_working_memory(
            self.conn,
            days=7,
            now=datetime(2026, 1, 21, tzinfo=timezone.utc),
        )

        self.assertEqual(deleted, 1)
        remaining = self.conn.execute(
            "SELECT session_id FROM working_memory ORDER BY session_id"
        ).fetchall()
        self.assertEqual([row["session_id"] for row in remaining], ["fresh-session"])

    def test_working_memory_prune_keeps_row_at_exact_cutoff(self):
        upsert_working_memory(
            self.conn,
            session_id="boundary-session",
            current_goal="Ship feature",
            current_focus="Boundary coverage",
            next_step="Resume tomorrow",
            status="paused",
            updated_at="2026-01-14T00:00:00Z",
            details={"active_tasks": ["boundary-task"]},
        )

        deleted = prune_stale_working_memory(
            self.conn,
            days=7,
            now=datetime(2026, 1, 21, tzinfo=timezone.utc),
        )

        self.assertEqual(deleted, 0)
        remaining = self.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        self.assertEqual(remaining, 1)

    def test_session_memory_prune_deletes_rows_and_removes_fts_entries(self):
        upsert_session_memory(
            self.conn,
            session_id="stale-session",
            title="Old handoff",
            summary="Paused work long ago.",
            left_off_at="Waiting on old lunarflag approval.",
            updated_at="2026-01-01T00:00:00Z",
            details={"next_steps": ["Resume after lunarflag approval"]},
        )
        upsert_session_memory(
            self.conn,
            session_id="fresh-session",
            title="Fresh handoff",
            summary="Recent work is still active.",
            left_off_at="Continue after review.",
            updated_at="2026-01-20T00:00:00Z",
            details={"next_steps": ["Resume after starlight review"]},
        )

        deleted = prune_stale_session_memory(
            self.conn,
            days=7,
            now=datetime(2026, 1, 21, tzinfo=timezone.utc),
        )

        self.assertEqual(deleted, 1)
        remaining = self.conn.execute(
            "SELECT session_id FROM session_memory ORDER BY session_id"
        ).fetchall()
        self.assertEqual([row["session_id"] for row in remaining], ["fresh-session"])
        self.assertEqual(search_session_memory_fts(self.conn, "lunarflag", limit=5), [])
        fresh_hits = search_session_memory_fts(self.conn, "starlight", limit=5)
        self.assertEqual([row["session_id"] for row in fresh_hits], ["fresh-session"])


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
