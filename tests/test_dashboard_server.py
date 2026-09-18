import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from memory.servers import dashboard_server


class TestDashboardDaemonDetection(unittest.TestCase):
    def test_check_daemon_uses_launchctl_pid_when_available(self):
        launchctl_result = MagicMock(returncode=0, stdout='    "PID" = 4321;\n')
        with patch.object(dashboard_server.subprocess, "run", return_value=launchctl_result):
            self.assertEqual(dashboard_server._check_daemon(), (True, 4321))

    def test_check_daemon_falls_back_to_process_title_lookup(self):
        launchctl_result = MagicMock(returncode=113, stdout="")
        pgrep_result = MagicMock(returncode=0, stdout="41177\n")
        with patch.object(dashboard_server.subprocess, "run", side_effect=[launchctl_result, pgrep_result]):
            self.assertEqual(dashboard_server._check_daemon(), (True, 41177))


class TestDashboardServices(unittest.TestCase):
    def test_get_services_includes_recall_and_ollama_status_and_start_metadata(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value={"c": 12})),
            MagicMock(fetchone=MagicMock(return_value={"c": 34})),
            MagicMock(fetchone=MagicMock(return_value={"c": 5})),
            MagicMock(fetchone=MagicMock(return_value={"c": 7})),
            MagicMock(fetchone=MagicMock(return_value={"c": 3})),
            MagicMock(fetchone=MagicMock(return_value={"c": 2})),
            MagicMock(fetchone=MagicMock(return_value={"c": 4})),
        ]

        with patch.object(dashboard_server, "_check_daemon", return_value=(True, 4321)), \
             patch.object(dashboard_server, "_check_ollama", return_value=(True, "qwen2.5:3b", ["qwen2.5:3b"])), \
             patch.object(dashboard_server, "_check_recall_server", return_value={"running": True, "status": "running", "port": 7747, "embed_model_ready": True, "embed_model_name": "all-MiniLM-L6-v2", "embed_model_error": ""}), \
             patch.object(dashboard_server, "_parse_daemon_log", return_value=("2026-01-01T00:00:00Z", 17)), \
             patch.object(dashboard_server, "_parse_activity_log", return_value={"llm_calls_avoided": 9, "est_tokens_saved": 3210}), \
             patch.object(dashboard_server, "_list_logs", return_value=[]), \
             patch.object(dashboard_server, "open_db", return_value=conn), \
             patch("memory.servers.dashboard_server.os.path.exists", return_value=True), \
             patch("memory.servers.dashboard_server.os.path.getsize", return_value=2048):
            payload = dashboard_server.get_services()

        self.assertEqual(payload["daemon"], {
            "running": True,
            "pid": 4321,
            "last_run": "2026-01-01T00:00:00Z",
            "facts_extracted": 17,
            "unprocessed_sessions": 4,
            "process_one_endpoint": "/ops/daemon/process-one",
        })
        self.assertEqual(payload["memory_search_engine"]["running"], True)
        self.assertEqual(payload["memory_search_engine"]["status"], "running")
        self.assertEqual(payload["memory_search_engine"]["port"], 7747)
        self.assertEqual(payload["memory_search_engine"]["embed_model_ready"], True)
        self.assertEqual(payload["memory_search_engine"]["embed_model_name"], "all-MiniLM-L6-v2")
        self.assertEqual(payload["memory_search_engine"]["display_name"], "Memory Search Engine")
        self.assertEqual(payload["memory_search_engine"]["restart_endpoint"], "/ops/memory-search-engine/restart")
        self.assertIn("memory/servers/ingest_server.py", payload["memory_search_engine"]["restart_command"])
        self.assertEqual(payload["ollama"]["running"], True)
        self.assertEqual(payload["ollama"]["model"], "qwen2.5:3b")
        self.assertEqual(payload["ollama"]["installed_models"], ["qwen2.5:3b"])
        self.assertEqual(payload["ollama"]["default_model"], "qwen2.5:7b")
        self.assertEqual(payload["ollama"]["only_small_model_installed"], True)
        self.assertIn("qwen2.5:7b", payload["ollama"]["warning"])
        self.assertEqual(payload["ollama"]["pull_recommendation"], "ollama pull qwen2.5:7b")
        self.assertEqual(payload["ollama"]["start_command"], "ollama serve")
        self.assertEqual(payload["ollama"]["start_endpoint"], "/ops/ollama/start")
        self.assertEqual(payload["efficiency"]["llm_calls_avoided"], 9)
        self.assertEqual(payload["efficiency"]["est_tokens_saved"], 3210)
        self.assertIn("chars/4", payload["efficiency"]["method"])
        self.assertEqual(payload["memory"]["total_sessions"], 12)
        self.assertEqual(payload["memory"]["total_facts"], 34)
        self.assertEqual(payload["memory"]["total_episodes"], 5)
        self.assertEqual(payload["memory"]["total_procedures"], 7)
        self.assertEqual(payload["memory"]["total_working_memory"], 3)
        self.assertEqual(payload["memory"]["total_session_memory"], 2)
        conn.close.assert_called_once()

    def test_start_ollama_reports_running_state_after_attempt(self):
        proc = object()
        with patch.object(dashboard_server, "start_ollama_if_needed", return_value=proc), \
             patch.object(dashboard_server, "_check_ollama", return_value=(True, "qwen2.5:7b", ["qwen2.5:3b", "qwen2.5:7b"])):
            payload = dashboard_server.start_ollama()

        self.assertEqual(payload, {
            "ok": True,
            "running": True,
            "model": "qwen2.5:7b",
            "installed_models": ["qwen2.5:3b", "qwen2.5:7b"],
            "default_model": "qwen2.5:7b",
            "started_here": True,
            "start_command": "ollama serve",
        })


    def test_restart_recall_server_reports_running_state_after_attempt(self):
        proc = MagicMock(pid=9876)
        with patch.object(dashboard_server.subprocess, "run", return_value=MagicMock(returncode=0)), \
             patch.object(dashboard_server.subprocess, "Popen", return_value=proc), \
             patch("memory.servers.dashboard_server.os.path.exists", return_value=False), \
             patch.object(dashboard_server, "_check_recall_server", side_effect=[
                 {"running": True, "status": "running", "port": 7747, "embed_model_ready": True, "embed_model_name": "all-MiniLM-L6-v2", "embed_model_error": ""},
             ]):
            payload = dashboard_server.restart_recall_server()

        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["running"], True)
        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["pid"], 9876)
        self.assertEqual(payload["started_here"], True)
        self.assertEqual(payload["port"], 7747)
        self.assertEqual(payload["embed_model_ready"], True)
        self.assertEqual(payload["embed_model_name"], "all-MiniLM-L6-v2")
        self.assertIn("memory/servers/ingest_server.py", payload["restart_command"])

    def test_restart_recall_server_falls_back_to_manual_start_when_launchctl_kickstart_fails(self):
        proc = MagicMock(pid=2468)

        def fake_exists(path):
            return path.endswith("com.memory.ingest.plist")

        with patch.object(dashboard_server.subprocess, "run", return_value=MagicMock(returncode=113)), \
             patch.object(dashboard_server.subprocess, "Popen", return_value=proc), \
             patch("memory.servers.dashboard_server.os.path.exists", side_effect=fake_exists), \
             patch.object(dashboard_server, "_check_recall_server", side_effect=[
                 {"running": True, "status": "running", "port": 7747, "embed_model_ready": True, "embed_model_name": "all-MiniLM-L6-v2", "embed_model_error": ""},
             ]):
            payload = dashboard_server.restart_recall_server()

        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["pid"], 2468)
        self.assertEqual(payload["started_here"], True)
        self.assertEqual(payload["status"], "running")


    def test_process_one_daemon_session_returns_result_and_remaining_queue(self):
        fake_conn = MagicMock()
        fake_conn.execute.return_value = MagicMock(fetchone=MagicMock(return_value={"c": 3}))

        fake_process = MagicMock(return_value={"ok": True, "processed": True, "session_id": "sess-1"})
        with patch.dict("sys.modules", {"memory.daemon": MagicMock(process_one_unprocessed_session=fake_process)}), \
             patch.object(dashboard_server, "open_db", return_value=fake_conn):
            payload = dashboard_server.process_one_daemon_session()

        self.assertEqual(payload, {
            "ok": True,
            "processed": True,
            "session_id": "sess-1",
            "unprocessed_sessions": 3,
        })
        fake_conn.close.assert_called_once()


class TestDashboardActivityStats(unittest.TestCase):
    def test_parse_activity_log_sums_memory_answer_savings(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".log") as tmp:
            tmp.write(json.dumps({"action": "memory_answer", "tokens_saved_estimate": 120}) + "\n")
            tmp.write(json.dumps({"action": "memory_answer", "tokens_saved_estimate": 80}) + "\n")
            tmp.write(json.dumps({"action": "processed", "tokens_saved_estimate": 999}) + "\n")
            tmp.flush()

            old_cache = dict(dashboard_server._ACTIVITY_STATS_CACHE)
            try:
                dashboard_server._ACTIVITY_STATS_CACHE = {
                    "mtime": None,
                    "size": None,
                    "stats": {"llm_calls_avoided": 0, "est_tokens_saved": 0},
                }
                with patch.object(dashboard_server, "_ACTIVITY_LOG_PATH", tmp.name):
                    payload = dashboard_server._parse_activity_log()
            finally:
                dashboard_server._ACTIVITY_STATS_CACHE = old_cache

        self.assertEqual(payload, {"llm_calls_avoided": 2, "est_tokens_saved": 200})

    def test_parse_performance_activity_log_collects_recent_latency_samples(self):
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".log") as tmp:
            tmp.write(json.dumps({
                "timestamp": "2026-01-01T10:00:00Z",
                "component": "retrieval",
                "action": "recall_timing",
                "prompt_embedding_ms": 12.5,
                "memory_search_ms": 44.2,
            }) + "\n")
            tmp.write(json.dumps({
                "timestamp": "2026-01-01T10:00:05Z",
                "component": "daemon",
                "action": "session_timing",
                "session": "sess-1",
                "duration_ms": 912.0,
            }) + "\n")
            tmp.flush()

            old_cache = dict(dashboard_server._PERFORMANCE_STATS_CACHE)
            try:
                dashboard_server._PERFORMANCE_STATS_CACHE = {
                    "mtime": None,
                    "size": None,
                    "stats": {
                        "recall_embedding_ms_recent": [],
                        "memory_search_ms_recent": [],
                        "daemon_session_ms_recent": [],
                        "summary": {
                            "recall_embedding_avg_ms": 0.0,
                            "memory_search_avg_ms": 0.0,
                            "daemon_session_avg_ms": 0.0,
                            "recall_embedding_p95_ms": 0.0,
                            "memory_search_p95_ms": 0.0,
                            "daemon_session_p95_ms": 0.0,
                        },
                    },
                }
                with patch.object(dashboard_server, "_ACTIVITY_LOG_PATH", tmp.name):
                    payload = dashboard_server._parse_performance_activity_log()
            finally:
                dashboard_server._PERFORMANCE_STATS_CACHE = old_cache

        self.assertEqual(payload["recall_embedding_ms_recent"], [{
            "timestamp": "2026-01-01T10:00:00Z",
            "label": "10:00:00",
            "value": 12.5,
        }])
        self.assertEqual(payload["memory_search_ms_recent"][0]["value"], 44.2)
        self.assertEqual(payload["daemon_session_ms_recent"][0]["session"], "sess-1")
        self.assertEqual(payload["summary"]["recall_embedding_avg_ms"], 12.5)
        self.assertEqual(payload["summary"]["memory_search_p95_ms"], 44.2)
        self.assertEqual(payload["summary"]["daemon_session_avg_ms"], 912.0)


class TestDashboardChartStats(unittest.TestCase):
    def test_get_chart_stats_includes_performance_series(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(fetchall=MagicMock(return_value=[{"day": "2026-01-01", "count": 2}])),
            MagicMock(fetchall=MagicMock(return_value=[{"day": "2026-01-01", "count": 3}])),
            MagicMock(fetchall=MagicMock(return_value=[{"day": "2026-01-01", "count": 1}])),
        ]
        performance = {
            "recall_embedding_ms_recent": [{"label": "10:00:00", "value": 12.5}],
            "memory_search_ms_recent": [{"label": "10:00:01", "value": 44.2}],
            "daemon_session_ms_recent": [{"label": "10:00:02", "value": 912.0}],
            "summary": {"recall_embedding_avg_ms": 12.5},
        }

        with patch.object(dashboard_server, "open_db", return_value=conn), \
             patch.object(dashboard_server, "_parse_performance_activity_log", return_value=performance):
            payload = dashboard_server.get_chart_stats()

        self.assertEqual(payload["sessions_by_day"], [{"day": "2026-01-01", "count": 2}])
        self.assertEqual(payload["recall_embedding_ms_recent"], [{"label": "10:00:00", "value": 12.5}])
        self.assertEqual(payload["memory_search_ms_recent"][0]["value"], 44.2)
        self.assertEqual(payload["daemon_session_ms_recent"][0]["value"], 912.0)
        self.assertEqual(payload["performance_summary"], {"recall_embedding_avg_ms": 12.5})
        conn.close.assert_called_once()


class TestDashboardLogFormatting(unittest.TestCase):
    def test_with_log_partitions_adds_separator_after_each_non_separator_line(self):
        self.assertEqual(
            dashboard_server._with_log_partitions(["line 1", "line 2"]),
            ["line 1", "=" * 60, "line 2", "=" * 60],
        )
        self.assertEqual(
            dashboard_server._with_log_partitions(["=" * 60, "line 2"]),
            ["=" * 60, "line 2", "=" * 60],
        )


class TestDashboardHtml(unittest.TestCase):
    def test_dashboard_html_mentions_memory_search_engine_ollama_and_daemon_runtime_sections(self):
        html = dashboard_server.dashboard().body.decode("utf-8")
        self.assertIn("Restart Memory Search Engine", html)
        self.assertIn("Process One Session", html)
        self.assertIn("Serves recall requests and warms the local embedding model for semantic search.", html)
        self.assertIn("id=\"t-recall\"", html)
        self.assertIn("id=\"t-recall-port\"", html)
        self.assertIn("id=\"t-recall-embed-name\"", html)
        self.assertIn("id=\"t-recall-embed-status\"", html)
        self.assertIn("id=\"svc-memory_search_engine\"", html)
        self.assertIn("id=\"panel-procedural\"", html)
        self.assertIn("id=\"tb-procedural\"", html)
        self.assertIn("id=\"t-procedural\"", html)
        self.assertIn("switchTab('procedural')", html)
        self.assertIn("id=\"panel-working\"", html)
        self.assertIn("id=\"tb-working\"", html)
        self.assertIn("id=\"t-working\"", html)
        self.assertIn("switchTab('working')", html)
        self.assertIn("id=\"svc-working_memory\"", html)
        self.assertIn("id=\"panel-session-memory\"", html)
        self.assertIn("id=\"t-memory-answers\"", html)
        self.assertIn("id=\"t-token-saved\"", html)
        self.assertIn("id=\"tb-session-memory\"", html)
        self.assertIn("id=\"t-session-memory\"", html)
        self.assertIn("switchTab('session-memory')", html)
        self.assertIn("id=\"svc-session_memory\"", html)
        self.assertIn("id=\"t-perf-embed-avg\"", html)
        self.assertIn("id=\"t-perf-search-avg\"", html)
        self.assertIn("id=\"t-perf-daemon-avg\"", html)
        self.assertIn("id=\"ch-perf-embed\"", html)
        self.assertIn("id=\"ch-perf-search\"", html)
        self.assertIn("id=\"ch-perf-daemon\"", html)
        self.assertIn("Prompt Embedding ms", html)
        self.assertIn("Memory Search ms", html)
        self.assertIn("Daemon Session Extraction ms", html)
        self.assertIn("Start Ollama", html)
        self.assertIn("id=\"t-ollama\"", html)
        self.assertIn("id=\"t-ollama-default-model\"", html)
        self.assertIn("id=\"t-ollama-installed-models\"", html)
        self.assertIn("id=\"t-ollama-warning\"", html)
        self.assertIn("id=\"t-daemon\"", html)
        self.assertIn("id=\"t-daemon-pending\"", html)
        self.assertIn("id=\"btn-process-daemon\"", html)
        self.assertIn("aria-describedby=\"tooltip-process-daemon\"", html)
        self.assertIn("id=\"tooltip-process-daemon\"", html)
        self.assertIn("button asks the daemon pipeline to pick the next unprocessed session", html)
        self.assertNotIn("class=\"overview-copy\"", html)
        self.assertIn("id=\"svc-wake_up\"", html)
        self.assertIn("id=\"svc-save_hook\"", html)
        self.assertIn("function logLineClass(line)", html)


if __name__ == "__main__":
    unittest.main()
