import os
import sqlite3
import tempfile
import unittest

from memory.db import (
    ensure_schema,
    get_latest_system_stats,
    get_telemetry_table_counts,
    init_db,
    insert_latency_breakdown,
    insert_recall_event,
    insert_retrieval_lane_metric,
    insert_system_stat,
    list_recall_events,
    list_recent_latency_breakdowns,
    list_recent_system_stats,
    list_retrieval_lane_metrics,
    summarize_latency_breakdowns,
)


class TestTelemetrySchema(unittest.TestCase):
    def test_bootstrap_creates_telemetry_tables(self):
        conn = init_db(":memory:")
        try:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            self.assertTrue(
                {
                    "telemetry_recall_events",
                    "telemetry_retrieval_lane_metrics",
                    "telemetry_latency_breakdowns",
                    "telemetry_system_stats",
                }.issubset(tables)
            )
        finally:
            conn.close()

    def test_ensure_schema_adds_telemetry_tables_to_legacy_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        try:
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
            conn.commit()

            ensure_schema(conn)

            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            self.assertIn("telemetry_recall_events", tables)
            self.assertIn("telemetry_retrieval_lane_metrics", tables)
            self.assertIn("telemetry_latency_breakdowns", tables)
            self.assertIn("telemetry_system_stats", tables)
            conn.close()
        finally:
            os.unlink(path)


class TestTelemetryHelpers(unittest.TestCase):
    def setUp(self):
        self.conn = init_db(":memory:")

    def tearDown(self):
        self.conn.close()

    def test_recall_event_lane_metric_latency_and_system_stat_helpers_round_trip(self):
        insert_recall_event(
            self.conn,
            request_id="req-1",
            event_type="request",
            session_id="sess-1",
            agent="claude",
            prompt_chars=18,
            prompt_tokens_estimate=5,
            include_working_memory=True,
            project_context={
                "project_id": "demo",
                "repo_root": "/tmp/demo",
                "cwd": "/tmp/demo/app",
                "git_remote": "git@github.com:example/demo.git",
                "git_branch": "main",
            },
            details={"route": "/recall"},
        )
        insert_recall_event(
            self.conn,
            request_id="req-1",
            event_type="outcome",
            action="inject",
            facts_count=1,
            episodic_count=2,
            procedural_count=0,
            session_memory_count=1,
            working_memory_count=1,
            injection_tokens_estimate=40,
            recalled_context_tokens_estimate=120,
            compression_gain_tokens_estimate=80,
            details={"timings": {"memory_search_ms": 44.2}},
        )
        insert_retrieval_lane_metric(
            self.conn,
            request_id="req-1",
            lane="episodic",
            candidate_count=8,
            filtered_count=2,
            selected_count=2,
            hit_count=1,
            duration_ms=12.5,
            details={"reason": "semantic"},
        )
        insert_latency_breakdown(
            self.conn,
            request_id="req-1",
            component="retrieval",
            operation="recall",
            stage="prompt_embedding",
            duration_ms=12.5,
            session_id="sess-1",
            details={"metric_key": "prompt_embedding_ms"},
        )
        insert_system_stat(
            self.conn,
            component="ingest_server",
            process_id=1234,
            session_id="sess-1",
            metric_name="rss_bytes",
            metric_value=4096,
            unit="bytes",
            details={"route": "/recall"},
        )

        recall_events = list_recall_events(self.conn, request_id="req-1", limit=10)
        self.assertEqual([event["event_type"] for event in recall_events], ["outcome", "request"])
        self.assertEqual(recall_events[0]["action"], "inject")
        self.assertEqual(recall_events[-1]["project_id"], "demo")
        self.assertEqual(recall_events[-1]["include_working_memory"], True)
        self.assertEqual(recall_events[-1]["details"], {"route": "/recall"})

        lane_metrics = list_retrieval_lane_metrics(self.conn, request_id="req-1", limit=10)
        self.assertEqual(lane_metrics[0]["lane"], "episodic")
        self.assertEqual(lane_metrics[0]["candidate_count"], 8)
        self.assertEqual(lane_metrics[0]["details"], {"reason": "semantic"})

        latency_rows = list_recent_latency_breakdowns(
            self.conn,
            component="retrieval",
            operation="recall",
            stage="prompt_embedding",
            limit=10,
        )
        self.assertEqual(latency_rows[0]["duration_ms"], 12.5)
        self.assertEqual(latency_rows[0]["details"], {"metric_key": "prompt_embedding_ms"})

        system_rows = list_recent_system_stats(self.conn, component="ingest_server", limit=10)
        self.assertEqual(system_rows[0]["metric_name"], "rss_bytes")
        self.assertEqual(system_rows[0]["metric_value"], 4096.0)
        self.assertEqual(system_rows[0]["details"], {"route": "/recall"})

        latest_system = get_latest_system_stats(self.conn, component="ingest_server")
        self.assertEqual(latest_system["rss_bytes"]["process_id"], 1234)
        self.assertEqual(
            get_telemetry_table_counts(self.conn),
            {
                "recall_events": 2,
                "retrieval_lane_metrics": 1,
                "latency_breakdowns": 1,
                "system_stats": 1,
            },
        )

    def test_latency_summary_calculates_average_and_p95_by_stage(self):
        for value in (10.0, 20.0, 30.0):
            insert_latency_breakdown(
                self.conn,
                component="retrieval",
                operation="recall",
                stage="memory_search",
                duration_ms=value,
            )
        insert_latency_breakdown(
            self.conn,
            component="retrieval",
            operation="recall",
            stage="prompt_embedding",
            duration_ms=5.0,
        )

        summary = summarize_latency_breakdowns(self.conn, component="retrieval", operation="recall", limit=10)

        self.assertEqual(summary["memory_search"], {"count": 3, "avg_ms": 20.0, "p95_ms": 20.0})
        self.assertEqual(summary["prompt_embedding"], {"count": 1, "avg_ms": 5.0, "p95_ms": 5.0})


if __name__ == "__main__":
    unittest.main()
