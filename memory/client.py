"""Thin Python client for the memory ingest/recall server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class MemoryClient:
    """Small stdlib-only HTTP client for /ingest, /recall, and /status."""

    def __init__(self, host: str = "localhost", port: int = 7747) -> None:
        self._base_url = f"http://{host}:{port}"

    @staticmethod
    def _error_message_from_http_error(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""

        if body:
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    detail = data.get("detail", data)
                    return f"HTTP {exc.code}: {detail}"
            except json.JSONDecodeError:
                return f"HTTP {exc.code}: {body}"

        return f"HTTP {exc.code}: {exc.reason}"

    @staticmethod
    def _server_start_hint() -> str:
        return "Memory ingest server is not running — start it with: python3 memory/ingest_server.py"

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._error_message_from_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(self._server_start_hint()) from exc

    def _get(self, path: str) -> dict:
        request = urllib.request.Request(f"{self._base_url}{path}", method="GET")
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._error_message_from_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(self._server_start_hint()) from exc

    def save_session(
        self,
        session_id: str,
        agent: str = "unknown",
        turns: list | None = None,
        started_at: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        payload: dict = {
            "session_id": session_id,
            "agent": agent,
            "turns": turns or [],
        }
        if started_at is not None:
            payload["started_at"] = started_at
        if metadata is not None:
            payload["metadata"] = metadata
        return self._post("/ingest", payload)

    def recall(self, prompt: str, *, include_working_memory: bool = False, session_id: str | None = None) -> dict:
        payload: dict = {
            "prompt": prompt,
            "include_working_memory": include_working_memory,
        }
        if session_id is not None:
            payload["session_id"] = session_id
        return self._post("/recall", payload)

    def status(self) -> dict:
        return self._get("/status")
