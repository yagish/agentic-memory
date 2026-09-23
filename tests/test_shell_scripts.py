from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import site


REPO_ROOT = Path(__file__).resolve().parents[1]
USER_SITE = site.getusersitepackages()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, *, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - exercised on retries only
            last_error = exc
            time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {url}: {last_error}")


def _wait_for_http_down(url: str, *, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5)
        except Exception:
            return
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {url} to stop responding")


def test_start_dashboard_script_starts_and_stops_query_server() -> None:
    port = _free_port()
    env = os.environ.copy()

    with tempfile.TemporaryDirectory() as home_dir:
        env.update(
            {
                "HOME": home_dir,
                "MEMORY_QUERY_PORT": str(port),
                "MEMORY_DISABLE_FILE_LOGS": "1",
                "PYTHONPATH": ":".join(filter(None, [str(REPO_ROOT), USER_SITE, env.get("PYTHONPATH", "")])),
            }
        )

        stop_cmd = ["bash", "scripts/stop-memory.sh", "query"]
        try:
            started = subprocess.run(
                ["bash", "scripts/start-dashboard.sh"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert started.returncode == 0, started.stderr or started.stdout
            assert f"http://localhost:{port}" in started.stdout

            _wait_for_http(f"http://127.0.0.1:{port}/")

            pid_file = Path(home_dir) / ".memory" / "query.pid"
            assert pid_file.exists()

            stopped = subprocess.run(
                stop_cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            assert stopped.returncode == 0, stopped.stderr or stopped.stdout
            _wait_for_http_down(f"http://127.0.0.1:{port}/")
        finally:
            subprocess.run(
                stop_cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )


def test_install_and_uninstall_manage_hooks_extension_and_plists() -> None:
    with tempfile.TemporaryDirectory() as home_dir, tempfile.TemporaryDirectory() as bin_dir:
        home = Path(home_dir)
        fakebin = Path(bin_dir)
        launchctl_log = home / "launchctl.log"
        (home / ".memory").mkdir(parents=True, exist_ok=True)
        (home / ".memory" / "identity.md").write_text("# Identity\n", encoding="utf-8")

        fake_python = fakebin / "python3"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"REAL_PYTHON={json.dumps(sys.executable)}\n"
            "if [[ \"${1:-}\" == \"-c\" && \"${2:-}\" == \"import sys; print(sys.executable)\" ]]; then\n"
            "  printf '%s\\n' \"$0\"\n"
            "  exit 0\n"
            "fi\n"
            "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"pip\" && \"${3:-}\" == \"install\" ]]; then\n"
            "  exit 0\n"
            "fi\n"
            "exec \"$REAL_PYTHON\" \"$@\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        fake_launchctl = fakebin / "launchctl"
        fake_launchctl.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$*\" >> \"${FAKE_LAUNCHCTL_LOG:?}\"\n",
            encoding="utf-8",
        )
        fake_launchctl.chmod(0o755)

        env = os.environ.copy()
        env.update(
            {
                "HOME": str(home),
                "PATH": f"{fakebin}:{env.get('PATH', '')}",
                "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
                "MEMORY_OLLAMA_MODEL": "qwen2.5:7b",
                "MEMORY_DISABLE_FILE_LOGS": "1",
                "PYTHONPATH": ":".join(filter(None, [str(REPO_ROOT), USER_SITE, env.get("PYTHONPATH", "")])),
            }
        )

        installed = subprocess.run(
            ["bash", "install.sh"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert installed.returncode == 0, installed.stderr or installed.stdout

        settings_path = home / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

        stop_hooks = settings["hooks"]["Stop"]
        wake_hooks = settings["hooks"]["UserPromptSubmit"]
        expected_python = str(fake_python)
        expected_save = f"{expected_python} {REPO_ROOT / 'integrations/claude/save_hook.py'}"
        expected_wake = f"{expected_python} {REPO_ROOT / 'integrations/claude/wake_up.py'}"
        assert any(h["command"] == expected_save for entry in stop_hooks for h in entry["hooks"])
        assert any(h["command"] == expected_wake for entry in wake_hooks for h in entry["hooks"])

        pi_ext = home / ".pi" / "agent" / "extensions" / "agentic-memory.ts"
        assert pi_ext.exists()
        assert str(REPO_ROOT / "integrations/pi/extension.ts") in pi_ext.read_text(encoding="utf-8")

        launch_agents = home / "Library" / "LaunchAgents"
        expected_plists = {
            "com.memory.daemon.plist",
            "com.memory.ingest.plist",
            "com.memory.query.plist",
            "com.memory.logrotate.plist",
        }
        assert expected_plists.issubset({path.name for path in launch_agents.iterdir()})

        launchctl_lines = launchctl_log.read_text(encoding="utf-8").splitlines()
        assert any("load" in line and "com.memory.logrotate.plist" in line for line in launchctl_lines)

        uninstalled = subprocess.run(
            ["bash", "install.sh", "--uninstall"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert uninstalled.returncode == 0, uninstalled.stderr or uninstalled.stdout

        remaining_plists = {path.name for path in launch_agents.iterdir()} if launch_agents.exists() else set()
        assert not (expected_plists & remaining_plists)
        assert not pi_ext.exists()

        settings_after = json.loads(settings_path.read_text(encoding="utf-8"))
        assert settings_after.get("hooks", {}).get("Stop", []) == []
        assert settings_after.get("hooks", {}).get("UserPromptSubmit", []) == []

        launchctl_lines = launchctl_log.read_text(encoding="utf-8").splitlines()
        assert any("unload" in line and "com.memory.logrotate.plist" in line for line in launchctl_lines)
