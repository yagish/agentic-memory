# test_daemon.py — automated tests for the background relearning daemon.
#
# Run with:  python3 -m pytest tests/test_daemon.py -v
#
# Each test creates a fresh in-memory database so there is no state leakage
# between tests and no files written to disk.

import json
import sys
import os
import unittest
from unittest.mock import patch

# Add the project root to sys.path so we can import memory.* packages.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.db import (
    init_db,
    upsert_session,
    get_unprocessed_sessions,
    mark_session_processed,
    upsert_insight,
    list_insights,
    assign_to_cluster,
    get_cluster_sessions,
    get_clusters,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_session(conn, session_id: str, transcript=None):
    """
    Insert a minimal session row used by multiple tests.

    Args:
        conn       — open connection from init_db()
        session_id — unique string ID for the session
        transcript — optional list of turn dicts; defaults to one user turn
    """
    # Default to a single user turn if no transcript is provided.
    if transcript is None:
        transcript = [{"role": "user", "content": f"Hello from {session_id}"}]
    # Use upsert_session so all schema columns are populated correctly.
    upsert_session(
        conn,
        session_id=session_id,
        agent="claude",
        transcript=transcript,
        started_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


def _make_embedding(value: float = 0.5, dims: int = 384) -> list[float]:
    """
    Build a synthetic embedding vector of the given dimensionality.

    All components are set to `value` so that two embeddings with the same
    value have cosine distance 0 (identical), while two with very different
    values have a large cosine distance.

    Args:
        value — the scalar to fill each dimension with
        dims  — number of dimensions (must match the model: 384 for all-MiniLM-L6-v2)
    """
    # A vector where every element is the same value has magnitude sqrt(dims * value^2).
    # Two such vectors with the same value have cosine similarity 1 → distance 0.
    return [value] * dims


# ---------------------------------------------------------------------------
# Test group 1: get_unprocessed_sessions / mark_session_processed (Ticket 13-01)
# ---------------------------------------------------------------------------

class TestUnprocessedSessions(unittest.TestCase):
    """Tests for get_unprocessed_sessions and mark_session_processed."""

    def setUp(self):
        # Each test gets a fresh in-memory database — no state leaks between tests.
        self.conn = init_db(":memory:")

    def test_get_unprocessed_sessions_returns_only_unprocessed(self):
        """
        Insert two sessions, mark one processed; only the unprocessed one is returned.
        """
        # Create two sessions: session A and session B.
        _make_session(self.conn, "session-A")
        _make_session(self.conn, "session-B")

        # Mark session A as processed by the daemon.
        mark_session_processed(self.conn, "session-A")

        # Only session B should be returned.
        results = get_unprocessed_sessions(self.conn, limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "session-B")

    def test_mark_session_processed_removes_from_queue(self):
        """
        After marking a session processed, get_unprocessed_sessions must not return it.
        """
        # Create one session — it starts as unprocessed.
        _make_session(self.conn, "session-X")

        # Before marking: must be returned.
        before = get_unprocessed_sessions(self.conn, limit=10)
        self.assertEqual(len(before), 1)

        # Mark it processed.
        mark_session_processed(self.conn, "session-X")

        # After marking: must NOT be returned.
        after = get_unprocessed_sessions(self.conn, limit=10)
        self.assertEqual(len(after), 0)

    def test_get_unprocessed_sessions_respects_limit(self):
        """
        get_unprocessed_sessions must return at most `limit` sessions.
        """
        # Insert 5 sessions — all unprocessed.
        for i in range(5):
            _make_session(self.conn, f"session-{i}")

        # Ask for at most 3.
        results = get_unprocessed_sessions(self.conn, limit=3)
        self.assertLessEqual(len(results), 3)

    def test_get_unprocessed_sessions_result_has_session_id(self):
        """
        Each result dict from get_unprocessed_sessions must carry at least session_id.
        """
        _make_session(self.conn, "session-check")
        results = get_unprocessed_sessions(self.conn, limit=1)
        self.assertEqual(len(results), 1)
        # The dict must have a session_id key so the daemon can reference it.
        self.assertIn("session_id", results[0])


# ---------------------------------------------------------------------------
# Test group 2: upsert_insight / list_insights (Ticket 13-01)
# ---------------------------------------------------------------------------

class TestInsights(unittest.TestCase):
    """Tests for upsert_insight and list_insights."""

    def setUp(self):
        # Fresh in-memory database for each test.
        self.conn = init_db(":memory:")

    def test_upsert_insight_returns_string_id(self):
        """
        upsert_insight must return a non-empty string (UUID).
        """
        insight_id = upsert_insight(
            self.conn,
            insight_type="pattern",
            content="User prefers verbose explanations.",
            evidence=["session-A", "session-B"],
            confidence=0.85,
        )
        self.assertIsInstance(insight_id, str)
        self.assertGreater(len(insight_id), 0)

    def test_list_insights_returns_inserted_insight(self):
        """
        After upsert_insight, list_insights must return that insight.
        """
        upsert_insight(
            self.conn,
            insight_type="preference",
            content="User likes dark mode.",
            evidence=[],
            confidence=0.9,
        )

        results = list_insights(self.conn)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "User likes dark mode.")
        self.assertEqual(results[0]["insight_type"], "preference")

    def test_list_insights_filter_by_type(self):
        """
        list_insights with insight_type filter must return only matching rows.
        """
        # Insert two insights of different types.
        upsert_insight(self.conn, "pattern", "Pattern insight.", [], 0.7)
        upsert_insight(self.conn, "skill", "Skill insight.", [], 0.8)

        # Filter by "pattern" — should return 1.
        results = list_insights(self.conn, insight_type="pattern")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["insight_type"], "pattern")

    def test_list_insights_empty_on_fresh_db(self):
        """
        list_insights returns an empty list when no insights have been inserted.
        """
        results = list_insights(self.conn)
        self.assertEqual(results, [])

    def test_upsert_insight_accumulates_multiple(self):
        """
        Each call to upsert_insight creates a new row — insights accumulate, not replace.
        """
        upsert_insight(self.conn, "pattern", "First insight.", [], 0.7)
        upsert_insight(self.conn, "pattern", "Second insight.", [], 0.8)

        results = list_insights(self.conn)
        self.assertEqual(len(results), 2)


# ---------------------------------------------------------------------------
# Test group 3: assign_to_cluster helpers (Ticket 13-03)
# ---------------------------------------------------------------------------

class TestClusters(unittest.TestCase):
    """Tests for assign_to_cluster, get_cluster_sessions, get_clusters."""

    def setUp(self):
        # Fresh in-memory database for each test.
        self.conn = init_db(":memory:")

    def test_assign_to_cluster_creates_new(self):
        """
        The first call to assign_to_cluster creates a new cluster and returns its ID.
        """
        # Create the session so the foreign-key constraint is satisfied.
        _make_session(self.conn, "session-1")

        embedding = _make_embedding(0.5)
        cluster_id = assign_to_cluster(self.conn, "session-1", embedding, label="test-cluster")

        # A non-empty cluster ID must be returned.
        self.assertIsInstance(cluster_id, str)
        self.assertGreater(len(cluster_id), 0)

        # The cluster must appear in get_clusters().
        clusters = get_clusters(self.conn)
        self.assertEqual(len(clusters), 1)

    def test_assign_to_cluster_joins_existing(self):
        """
        Two sessions with the same embedding land in the same cluster.
        """
        # Both sessions use embedding value 0.5 — cosine distance 0 (identical).
        _make_session(self.conn, "session-same-1")
        _make_session(self.conn, "session-same-2")

        embedding = _make_embedding(0.5)

        # First session creates a cluster.
        cluster_id_1 = assign_to_cluster(self.conn, "session-same-1", embedding)
        # Second session should join the same cluster (distance = 0 < threshold 0.3).
        cluster_id_2 = assign_to_cluster(self.conn, "session-same-2", embedding)

        # Both must be in the same cluster.
        self.assertEqual(cluster_id_1, cluster_id_2)

        # Only one cluster should exist.
        clusters = get_clusters(self.conn)
        self.assertEqual(len(clusters), 1)

    def test_assign_to_cluster_creates_separate(self):
        """
        Two sessions with very different embeddings each get their own cluster.
        """
        _make_session(self.conn, "session-diff-1")
        _make_session(self.conn, "session-diff-2")

        # Embedding A: all 0.5; embedding B: all -0.5.
        # Cosine distance between [0.5, 0.5, ...] and [-0.5, -0.5, ...] is 2.0 — very far.
        embedding_a = _make_embedding(0.5)
        embedding_b = _make_embedding(-0.5)

        cluster_id_a = assign_to_cluster(self.conn, "session-diff-1", embedding_a)
        cluster_id_b = assign_to_cluster(self.conn, "session-diff-2", embedding_b)

        # They must be in different clusters.
        self.assertNotEqual(cluster_id_a, cluster_id_b)

        # Two clusters must exist.
        clusters = get_clusters(self.conn)
        self.assertEqual(len(clusters), 2)

    def test_get_cluster_sessions_returns_members(self):
        """
        get_cluster_sessions returns the session_ids assigned to that cluster.
        """
        _make_session(self.conn, "session-member")
        embedding = _make_embedding(0.5)
        cluster_id = assign_to_cluster(self.conn, "session-member", embedding)

        # The session must appear in the cluster's member list.
        members = get_cluster_sessions(self.conn, cluster_id)
        self.assertIn("session-member", members)


# ---------------------------------------------------------------------------
# Test group 4: daemon run() once mode (current processing pipeline)
# ---------------------------------------------------------------------------

class TestDaemonOnceMode(unittest.TestCase):
    """Tests for the daemon's run(once=True) entry point."""

    def test_daemon_once_mode_marks_session_processed(self):
        """
        run(once=True) with a mocked ollama processes a session and marks it done.

        We use a real temp file so the daemon can open and close connections
        normally without invalidating our verification query.
        """
        import memory.daemon as daemon_module
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        log_path = tmp_path + ".log"

        try:
            setup_conn = init_db(tmp_path)
            _make_session(setup_conn, "session-once")
            setup_conn.close()

            mock_response = json.dumps([])

            with patch.object(daemon_module, "_call_ollama", return_value=mock_response), \
                 patch.object(daemon_module, "_start_ollama_if_needed", return_value=None), \
                 patch("memory.daemon.DB_PATH", tmp_path), \
                 patch("memory.daemon._DAEMON_LOG_PATH", log_path), \
                 patch("psutil.cpu_percent", return_value=10):
                daemon_module.run(once=True)

            verify_conn = init_db(tmp_path)
            row = verify_conn.execute(
                "SELECT daemon_processed_at FROM sessions WHERE session_id = ?",
                ("session-once",),
            ).fetchone()
            verify_conn.close()

            self.assertIsNotNone(row)
            self.assertIsNotNone(row["daemon_processed_at"])
        finally:
            os.unlink(tmp_path)
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_daemon_once_mode_leaves_session_unprocessed_when_step_raises(self):
        """
        If a required per-session step raises, the daemon should not stamp the
        session as processed. This allows a later run to retry the work.
        """
        import memory.daemon as daemon_module
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name
        log_path = tmp_path + ".log"

        try:
            setup_conn = init_db(tmp_path)
            _make_session(setup_conn, "session-retry")
            setup_conn.close()

            with patch.object(daemon_module, "_start_ollama_if_needed", return_value=None), \
                 patch.object(daemon_module, "_create_episodic_entry", return_value=("Title", "Abstract")), \
                 patch.object(daemon_module, "_assign_cluster", return_value="cluster-1"), \
                 patch.object(daemon_module, "_compact_cluster_if_ready", return_value=None), \
                 patch.object(daemon_module, "_extract_procedural_patterns", side_effect=RuntimeError("boom")), \
                 patch("memory.daemon.DB_PATH", tmp_path), \
                 patch("memory.daemon._DAEMON_LOG_PATH", log_path), \
                 patch("psutil.cpu_percent", return_value=10):
                daemon_module.run(once=True)

            verify_conn = init_db(tmp_path)
            row = verify_conn.execute(
                "SELECT daemon_processed_at FROM sessions WHERE session_id = ?",
                ("session-retry",),
            ).fetchone()
            verify_conn.close()

            self.assertIsNotNone(row)
            self.assertIsNone(row["daemon_processed_at"])
        finally:
            os.unlink(tmp_path)
            if os.path.exists(log_path):
                os.unlink(log_path)


if __name__ == "__main__":
    unittest.main()
