import unittest
from unittest.mock import MagicMock, patch

from memory import dashboard_server


class TestDashboardServices(unittest.TestCase):
    def test_get_services_includes_ollama_status_and_start_metadata(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value={"c": 12})),
            MagicMock(fetchone=MagicMock(return_value={"c": 34})),
            MagicMock(fetchone=MagicMock(return_value={"c": 5})),
        ]

        with patch.object(dashboard_server, "_check_daemon", return_value=(True, 4321)), \
             patch.object(dashboard_server, "_check_ollama", return_value=(True, "qwen2.5:3b", ["qwen2.5:3b"])), \
             patch.object(dashboard_server, "_parse_daemon_log", return_value=("2026-01-01T00:00:00Z", 17)), \
             patch.object(dashboard_server, "_list_logs", return_value=[]), \
             patch.object(dashboard_server, "open_db", return_value=conn), \
             patch("memory.dashboard_server.os.path.exists", return_value=True), \
             patch("memory.dashboard_server.os.path.getsize", return_value=2048):
            payload = dashboard_server.get_services()

        self.assertEqual(payload["daemon"], {
            "running": True,
            "pid": 4321,
            "last_run": "2026-01-01T00:00:00Z",
            "facts_extracted": 17,
        })
        self.assertEqual(payload["ollama"]["running"], True)
        self.assertEqual(payload["ollama"]["model"], "qwen2.5:3b")
        self.assertEqual(payload["ollama"]["installed_models"], ["qwen2.5:3b"])
        self.assertEqual(payload["ollama"]["default_model"], "qwen2.5:7b")
        self.assertEqual(payload["ollama"]["only_small_model_installed"], True)
        self.assertIn("qwen2.5:7b", payload["ollama"]["warning"])
        self.assertEqual(payload["ollama"]["pull_recommendation"], "ollama pull qwen2.5:7b")
        self.assertEqual(payload["ollama"]["start_command"], "ollama serve")
        self.assertEqual(payload["ollama"]["start_endpoint"], "/ops/ollama/start")
        self.assertEqual(payload["memory"]["total_sessions"], 12)
        self.assertEqual(payload["memory"]["total_facts"], 34)
        self.assertEqual(payload["memory"]["total_episodes"], 5)
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


class TestDashboardHtml(unittest.TestCase):
    def test_dashboard_html_mentions_ollama_status_and_start_button(self):
        html = dashboard_server.dashboard().body.decode("utf-8")
        self.assertIn("Start Ollama", html)
        self.assertIn("id=\"t-ollama\"", html)
        self.assertIn("id=\"t-ollama-default-model\"", html)
        self.assertIn("id=\"t-ollama-installed-models\"", html)
        self.assertIn("id=\"t-ollama-warning\"", html)
        self.assertIn("id=\"svc-desc-ollama\"", html)
        self.assertIn("id=\"svc-wake_up\"", html)
        self.assertIn("id=\"svc-save_hook\"", html)


if __name__ == "__main__":
    unittest.main()
