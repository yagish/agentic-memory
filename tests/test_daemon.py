# test_daemon.py — automated tests for the facts/episodic daemon.

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.contracts import ExtractedEpisode, ExtractedProcedure
from memory.db import get_unprocessed_sessions, init_db, mark_session_processed, upsert_session


def _make_session(conn, session_id: str, transcript=None, updated_at: str = "2026-01-01T00:00:00+00:00"):
    if transcript is None:
        transcript = [{"role": "user", "content": f"Hello from {session_id}"}]
    upsert_session(
        conn,
        session_id=session_id,
        agent="claude",
        transcript=transcript,
        started_at="2026-01-01T00:00:00+00:00",
        updated_at=updated_at,
    )


class TestUnprocessedSessions(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_get_unprocessed_sessions_returns_only_unprocessed(self):
        _make_session(self.conn, "session-A")
        _make_session(self.conn, "session-B")
        mark_session_processed(self.conn, "session-A")

        results = get_unprocessed_sessions(self.conn, limit=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["session_id"], "session-B")


class TestDaemonOnceMode(unittest.TestCase):
    def test_daemon_once_mode_extracts_facts_episode_procedure_and_marks_processed(self):
        import memory.daemon as daemon_module

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            setup_conn = init_db(tmp_path)
            _make_session(setup_conn, "session-once")
            setup_conn.close()

            call_order: list[str] = []

            def fake_extract_facts(*args, **kwargs):
                call_order.append("facts")
                return ["user.name = Yash"]

            def fake_create_episode(*args, **kwargs):
                call_order.append("episodic")
                return MagicMock(title="Auth fix", abstract="Resolved the middleware bug.")

            def fake_create_procedure(*args, **kwargs):
                call_order.append("procedural")
                return MagicMock(title="Deploy workflow", summary="Use this when deploying auth.")

            with patch.object(daemon_module, "_start_ollama_if_needed", return_value=None), \
                 patch.object(daemon_module, "_create_episodic_entry", side_effect=fake_create_episode) as create_episodic, \
                 patch.object(daemon_module, "_create_procedural_entry", side_effect=fake_create_procedure) as create_procedural, \
                 patch.object(daemon_module, "_extract_facts", side_effect=fake_extract_facts) as extract_facts, \
                 patch("memory.daemon.DB_PATH", tmp_path), \
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
            create_episodic.assert_called_once()
            create_procedural.assert_called_once()
            extract_facts.assert_called_once()
            # Extractors now run in parallel (ThreadPoolExecutor), so completion
            # order is non-deterministic. We verify that all three were called —
            # not the specific order — using assertCountEqual (order-independent).
            self.assertCountEqual(call_order, ["facts", "episodic", "procedural"])
        finally:
            os.unlink(tmp_path)

    def test_daemon_once_mode_leaves_session_unprocessed_when_processing_raises(self):
        import memory.daemon as daemon_module

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            setup_conn = init_db(tmp_path)
            _make_session(setup_conn, "session-retry")
            setup_conn.close()

            with patch.object(daemon_module, "_start_ollama_if_needed", return_value=None), \
                 patch.object(daemon_module, "_create_episodic_entry", side_effect=RuntimeError("boom")), \
                 patch("memory.daemon.DB_PATH", tmp_path), \
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

    def test_daemon_once_mode_skips_cpu_gate(self):
        import memory.daemon as daemon_module

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            setup_conn = init_db(tmp_path)
            _make_session(setup_conn, "session-force")
            setup_conn.close()

            with patch.object(daemon_module, "_start_ollama_if_needed", return_value=None), \
                 patch.object(daemon_module, "_create_episodic_entry", return_value=MagicMock(title="Forced episode", abstract="Forced abstract")) as create_episodic, \
                 patch.object(daemon_module, "_create_procedural_entry", return_value=MagicMock(title="Deploy workflow", summary="Forced summary")) as create_procedural, \
                 patch.object(daemon_module, "_extract_facts", return_value=["user.name = Yash"]) as extract_facts, \
                 patch("memory.daemon.DB_PATH", tmp_path), \
                 patch("psutil.cpu_percent", side_effect=AssertionError("cpu check should be skipped for --once")):
                daemon_module.run(once=True)

            verify_conn = init_db(tmp_path)
            row = verify_conn.execute(
                "SELECT daemon_processed_at FROM sessions WHERE session_id = ?",
                ("session-force",),
            ).fetchone()
            verify_conn.close()

            self.assertIsNotNone(row)
            self.assertIsNotNone(row["daemon_processed_at"])
            create_episodic.assert_called_once()
            create_procedural.assert_called_once()
            extract_facts.assert_called_once()
        finally:
            os.unlink(tmp_path)

    def test_daemon_normal_mode_skips_cycle_on_high_cpu(self):
        import memory.daemon as daemon_module

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            setup_conn = init_db(tmp_path)
            _make_session(setup_conn, "session-high-cpu")
            setup_conn.close()

            with patch.object(daemon_module, "time") as time_module, \
                 patch.object(daemon_module, "_run_unprocessed_batch") as run_batch, \
                 patch("memory.daemon.DB_PATH", tmp_path), \
                 patch("psutil.cpu_percent", side_effect=[95, KeyboardInterrupt]):
                time_module.sleep.side_effect = KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    daemon_module.run()

            run_batch.assert_not_called()
        finally:
            os.unlink(tmp_path)


    def test_process_session_logs_extractor_and_session_timings(self):
        import memory.daemon as daemon_module

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            conn = init_db(tmp_path)
            _make_session(conn, "timed-session")
            session = get_unprocessed_sessions(conn, limit=1)[0]

            perf_values = [0.0, 0.01, 0.03, 0.04, 0.07, 0.08, 0.12, 0.13, 0.18, 0.19, 0.25, 0.3]
            with patch.object(daemon_module, "_get_or_compact_session_text", return_value="user: hi"), \
                 patch.object(daemon_module, "_extract_facts", return_value=["user.name = Yash"]), \
                 patch.object(daemon_module, "_create_working_memory_entry", return_value=None), \
                 patch.object(daemon_module, "_create_session_memory_entry", return_value=None), \
                 patch.object(daemon_module, "_create_episodic_entry", return_value=MagicMock(title="Episode")), \
                 patch.object(daemon_module, "_create_procedural_entry", return_value=None), \
                 patch.object(daemon_module.time, "perf_counter", side_effect=perf_values), \
                 patch.object(daemon_module, "activity_log") as activity_log:
                daemon_module._process_session(conn, session)

            timing_calls = [call for call in activity_log.call_args_list if call.args[1] == "extractor_timing"]
            self.assertEqual(len(timing_calls), 5)
            self.assertTrue(any(call.kwargs["extractor"] == "facts" and call.kwargs["duration_ms"] == 20.0 for call in timing_calls))
            self.assertTrue(any(call.args[1] == "session_timing" and call.kwargs["duration_ms"] == 300.0 for call in activity_log.call_args_list))
        finally:
            conn.close()
            os.unlink(tmp_path)


class TestDaemonPruning(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_prune_stale_memories_includes_short_term_memory_cleanup(self):
        import memory.daemon.pruning as pruning_module

        with patch.object(pruning_module, "prune_stale_facts", return_value=1) as prune_facts, \
             patch.object(pruning_module, "prune_stale_episodic", return_value=2) as prune_episodic, \
             patch.object(pruning_module, "prune_stale_working_memory", return_value=3) as prune_working, \
             patch.object(pruning_module, "prune_stale_session_memory", return_value=4) as prune_session, \
             patch.object(pruning_module, "_daemon_log") as daemon_log, \
             patch.object(pruning_module, "activity_log") as activity_log:
            pruning_module._prune_stale_memories(self.conn)

        prune_facts.assert_called_once_with(self.conn, days=pruning_module._FACT_TTL_DAYS)
        prune_episodic.assert_called_once_with(self.conn, days=pruning_module._EPISODIC_TTL_DAYS)
        prune_working.assert_called_once_with(self.conn, days=pruning_module._WORKING_MEMORY_TTL_DAYS)
        prune_session.assert_called_once_with(self.conn, days=pruning_module._SESSION_MEMORY_TTL_DAYS)
        daemon_log.assert_called_once_with(
            "pruned 1 stale facts, 2 stale episodes, 3 stale working-memory snapshots, 4 stale session summaries"
        )
        activity_log.assert_called_once_with(
            "daemon",
            "prune",
            deleted_facts=1,
            deleted_episodes=2,
            deleted_working_memory=3,
            deleted_session_memory=4,
        )


class TestStructuredExtractionHelpers(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_create_episodic_entry_uses_shared_extractor_and_repository(self):
        import memory.daemon as daemon_module

        session = {
            "session_id": "session-episodic",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "transcript": json.dumps([
                {"role": "user", "content": "We fixed the auth loop."},
            ]),
        }

        episode = ExtractedEpisode(
            title="Auth loop fixed",
            abstract="The auth loop was fixed and tests were left for follow-up.",
            decisions=("Move token validation into middleware",),
            outcomes=("Login loop fixed",),
            follow_ups=("Add regression coverage",),
            confidence=0.8,
        )

        with patch.object(daemon_module, "extract_episode_from_session_text", return_value=episode) as extract_episode, \
             patch.object(daemon_module, "save_extracted_episode", return_value="ep-1") as save_episode, \
             patch.object(daemon_module, "_daemon_log") as daemon_log:
            saved_episode = daemon_module._create_episodic_entry(self.conn, session)

        self.assertIsNotNone(saved_episode)
        self.assertEqual(saved_episode.title, "Auth loop fixed")
        self.assertEqual(saved_episode.abstract, "The auth loop was fixed and tests were left for follow-up.")
        extract_episode.assert_called_once()
        save_episode.assert_called_once_with(
            self.conn,
            episode,
            session_id="session-episodic",
            happened_at="2026-01-01T00:00:00+00:00",
            source="daemon",
            embed_fn=daemon_module.embed,
        )
        logged_messages = [call.args[0] for call in daemon_log.call_args_list]
        self.assertTrue(any("episode extracted for session-episodic" in message for message in logged_messages))
        self.assertTrue(any("episode decisions for session-episodic" in message for message in logged_messages))
        self.assertTrue(any("episode outcomes for session-episodic" in message for message in logged_messages))
        self.assertTrue(any("episode follow-ups for session-episodic" in message for message in logged_messages))

    def test_extract_facts_uses_shared_extractor_and_repository(self):
        import memory.daemon as daemon_module

        session = {
            "session_id": "session-facts",
            "transcript": json.dumps([
                {"role": "user", "content": "My name is Yash."},
            ]),
        }

        fake_fact = MagicMock()

        with patch.object(daemon_module, "extract_facts_from_session_text", return_value=[fake_fact]) as extract_facts, \
             patch.object(daemon_module, "normalize_extracted_facts", return_value=[fake_fact]) as normalize_facts, \
             patch.object(daemon_module, "build_fact_content", return_value="user.name = Yash") as build_content, \
             patch.object(daemon_module, "save_extracted_facts", return_value=["fact-1"]) as save_facts, \
             patch.object(daemon_module, "_daemon_log") as daemon_log:
            fact_contents = daemon_module._extract_facts(self.conn, session)

        logged_messages = [call.args[0] for call in daemon_log.call_args_list]
        self.assertTrue(any("fact extracted for session-facts: user.name = Yash" in message for message in logged_messages))

        self.assertEqual(fact_contents, ["user.name = Yash"])
        normalize_facts.assert_called_once_with([fake_fact])
        build_content.assert_called_once_with(fake_fact)

        extract_facts.assert_called_once()
        save_facts.assert_called_once_with(
            self.conn,
            [fake_fact],
            session_id="session-facts",
            source="daemon_fact_extractor",
        )

    def test_create_procedural_entry_uses_shared_extractor_and_repository(self):
        import memory.daemon as daemon_module

        session = {
            "session_id": "session-procedural",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "transcript": json.dumps([
                {"role": "user", "content": "Deploy by building the image, running alembic upgrade, then validating staging."},
            ]),
        }

        procedure = ExtractedProcedure(
            title="Web deploy workflow",
            summary="Use this when deploying the web service.",
            steps=("Build the image", "Run alembic upgrade", "Validate staging"),
            trigger_phrases=("how do i deploy the web service",),
            confidence=0.84,
        )

        with patch.object(daemon_module, "extract_procedure_from_session_text", return_value=procedure) as extract_procedure, \
             patch.object(daemon_module, "save_extracted_procedure", return_value="proc-1") as save_procedure, \
             patch.object(daemon_module, "_daemon_log") as daemon_log:
            saved_procedure = daemon_module._create_procedural_entry(self.conn, session)

        self.assertIsNotNone(saved_procedure)
        self.assertEqual(saved_procedure.title, "Web deploy workflow")
        self.assertEqual(saved_procedure.summary, "Use this when deploying the web service.")
        extract_procedure.assert_called_once()
        save_procedure.assert_called_once_with(
            self.conn,
            procedure,
            session_id="session-procedural",
            updated_at="2026-01-01T00:00:00+00:00",
            source="daemon",
            embed_fn=daemon_module.embed,
        )
        logged_messages = [call.args[0] for call in daemon_log.call_args_list]
        self.assertTrue(any("procedure extracted for session-procedural" in message for message in logged_messages))
        self.assertTrue(any("procedure steps for session-procedural" in message for message in logged_messages))
        self.assertTrue(any("procedure triggers for session-procedural" in message for message in logged_messages))


if __name__ == "__main__":
    unittest.main()
