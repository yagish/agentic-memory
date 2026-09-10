import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from memory.servers import dashboard_server


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
        self.assertIn("memory/ingest_server.py", payload["memory_search_engine"]["restart_command"])
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
        self.assertIn("memory/ingest_server.py", payload["restart_command"])


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


class TestDashboardHtml(unittest.TestCase):
    def test_dashboard_html_mentions_memory_search_engine_ollama_and_daemon_runtime_sections(self):
        html = dashboard_server.dashboard().body.decode("utf-8")
        self.assertIn("Restart Memory Search Engine", html)
        self.assertIn("Process One Session", html)
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
        self.assertIn("Start Ollama", html)
        self.assertIn("id=\"t-ollama\"", html)
        self.assertIn("id=\"t-ollama-default-model\"", html)
        self.assertIn("id=\"t-ollama-installed-models\"", html)
        self.assertIn("id=\"t-ollama-warning\"", html)
        self.assertIn("id=\"t-daemon\"", html)
        self.assertIn("id=\"t-daemon-pending\"", html)
        self.assertIn("id=\"btn-process-daemon\"", html)
        self.assertIn("id=\"svc-wake_up\"", html)
        self.assertIn("id=\"svc-save_hook\"", html)


if __name__ == "__main__":
    unittest.main()
