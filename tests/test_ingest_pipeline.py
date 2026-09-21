import json
import unittest

from memory.db import init_db
from memory.servers.ingest_pipeline import ingest_session


class TestIngestPipeline(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")
        self.turns = [
            {"role": "user", "content": "How do I run the tests?"},
            {"role": "assistant", "content": "Run python3 -m pytest -q."},
        ]

    def tearDown(self):
        self.conn.close()

    def test_ingest_session_stores_session_only(self):
        outcome = ingest_session(
            self.conn,
            session_id="sess-1",
            agent="assistant",
            turns=self.turns,
            started_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
            project_context={
                "repo_root": "/tmp/demo-repo",
                "cwd": "/tmp/demo-repo/app",
                "git_remote": "git@github.com:example/demo.git",
                "git_branch": "feature/project-context",
            },
        )

        self.assertEqual(outcome.session_id, "sess-1")
        self.assertEqual(outcome.turn_count, 2)
        self.assertFalse(outcome.embedding_stored)
        self.assertEqual(outcome.chunk_count, 0)
        self.assertEqual(outcome.warnings, [])

        row = self.conn.execute(
            "SELECT transcript, turn_count, project_id, repo_root, cwd, git_remote, git_branch FROM sessions WHERE session_id = ?",
            ("sess-1",),
        ).fetchone()
        self.assertEqual(row["turn_count"], 2)
        self.assertEqual(json.loads(row["transcript"]), self.turns)
        self.assertEqual(row["project_id"], "git@github.com:example/demo.git")
        self.assertEqual(row["repo_root"], "/tmp/demo-repo")
        self.assertEqual(row["cwd"], "/tmp/demo-repo/app")
        self.assertEqual(row["git_remote"], "git@github.com:example/demo.git")
        self.assertEqual(row["git_branch"], "feature/project-context")


if __name__ == "__main__":
    unittest.main()
