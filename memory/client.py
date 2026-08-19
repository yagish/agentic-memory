# client.py — thin Python client for the memory ingest server.
#
# Usage:
#   from memory.client import MemoryClient
#   client = MemoryClient()  # default port 7747
#   client.save_session(session_id="abc", agent="cursor", turns=[
#       {"role": "user", "content": "..."},
#       {"role": "assistant", "content": "..."},
#   ])

import json              # for JSON serialisation of the request body and response
import urllib.request    # stdlib HTTP client — no external dependencies needed
import urllib.error      # for catching network errors from urllib.request


class MemoryClient:
    """
    Thin HTTP client for talking to the memory ingest server.

    Uses only Python's built-in urllib package — no third-party dependencies
    like 'requests' are required.  This makes the client easy to embed inside
    any Python project without adding new packages.

    Example:
        client = MemoryClient()
        client.save_session(
            session_id="my-session-123",
            agent="cursor",
            turns=[
                {"role": "user", "content": "How does quicksort work?"},
                {"role": "assistant", "content": "Quicksort is a divide-and-conquer algorithm..."},
            ],
        )
    """

    def __init__(self, host: str = "localhost", port: int = 7747) -> None:
        """
        Create a new client pointing at the given host and port.

        Args:
            host — hostname or IP of the ingest server (default "localhost")
            port — port the ingest server listens on (default 7747)
        """
        # Build the base URL once so we don't repeat the string concatenation.
        self._base_url = f"http://{host}:{port}"

    @staticmethod
    def _error_message_from_http_error(exc: urllib.error.HTTPError) -> str:
        """Extract a readable error message from an HTTPError response body."""
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

    def _post(self, path: str, payload: dict) -> dict:
        """
        Send an HTTP POST request with a JSON body and return the parsed response.

        Args:
            path    — URL path, e.g. "/ingest"
            payload — dict that will be JSON-serialised and sent as the body

        Returns:
            Parsed JSON response as a Python dict.

        Raises:
            ConnectionError if the server is not reachable.
            RuntimeError   if the server returns a non-2xx status code.
        """
        url = f"{self._base_url}{path}"
        # Encode the payload as UTF-8 bytes — required by urllib.request.
        body = json.dumps(payload).encode("utf-8")
        # Build the request with a Content-Type header so the server knows
        # we are sending JSON, not a form.
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            # urlopen opens the connection and sends the request.
            # It raises URLError on network failure and HTTPError on bad status codes.
            with urllib.request.urlopen(request) as response:
                # Read the response bytes and decode to a string, then parse JSON.
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._error_message_from_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            # URLError means the server is not running or is unreachable.
            raise ConnectionError(
                "Memory ingest server is not running — start it with: "
                "python3 cli.py ingest-server start"
            ) from exc

    def _get(self, path: str) -> dict:
        """
        Send an HTTP GET request and return the parsed JSON response.

        Args:
            path — URL path, e.g. "/status"

        Returns:
            Parsed JSON response as a Python dict.

        Raises:
            ConnectionError if the server is not reachable.
        """
        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._error_message_from_http_error(exc)) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                "Memory ingest server is not running — start it with: "
                "python3 cli.py ingest-server start"
            ) from exc

    def save_session(
        self,
        session_id: str,
        agent: str = "unknown",
        turns: list | None = None,
        started_at: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """
        POST /ingest — send a session to the memory server for storage.

        The server will upsert the session, embed the full transcript, and
        chunk it for finer-grained semantic search — the same pipeline used
        by the Claude Code Stop hook.

        Args:
            session_id — unique identifier for the conversation
            agent      — name of the agent (e.g. "cursor", "langchain")
            turns      — list of {"role": "user"/"assistant", "content": str} dicts
            started_at — ISO 8601 timestamp of the first turn; server uses now() if absent
            metadata   — optional dict of extra key-value pairs to attach to the session

        Returns:
            {"ok": True, "session_id": "..."} on success.

        Raises:
            ConnectionError if the server is not running.
        """
        # Default turns to an empty list so callers don't have to handle None.
        # The server will reject an empty turns list with a 422 error.
        if turns is None:
            turns = []

        # Build the JSON payload that matches the IngestRequest Pydantic model.
        payload: dict = {
            "session_id": session_id,
            "agent": agent,
            "turns": turns,
        }
        # Only include optional fields if they were provided — keeps the payload clean.
        if started_at is not None:
            payload["started_at"] = started_at
        if metadata is not None:
            payload["metadata"] = metadata

        return self._post("/ingest", payload)

    def recall(self, query: str, session_id: str | None = None) -> dict:
        """
        POST /recall — build a memory wake-up digest for a prompt.

        Args:
            query      — the user's prompt text
            session_id — optional current session ID to exclude from retrieval

        Returns:
            {
              "ok": True,
              "mode": "relevance" | "recency",
              "digest": "=== MEMORY WAKE-UP === ...",
              "session_count": <int>,
              "fact_count": <int>,
              "insight_count": <int>
            }
        """
        payload: dict = {"query": query}
        if session_id is not None:
            payload["session_id"] = session_id
        return self._post("/recall", payload)

    def direct_answer(
        self,
        query: str,
        session_id: str | None = None,
        min_score: float = 0.93,
    ) -> dict:
        """
        POST /answer — try to answer a prompt directly from memory.

        This is useful for prompt interception in other agents: if the same
        question was answered before, the caller can short-circuit the LLM.
        """
        payload: dict = {"query": query, "min_score": min_score}
        if session_id is not None:
            payload["session_id"] = session_id
        return self._post("/answer", payload)

    def status(self) -> dict:
        """
        GET /status — return the server's health summary.

        Returns:
            {
              "status": "ok",
              "total_sessions": <int>,
              "newest_session": "<ISO timestamp or null>",
              "db_size_bytes": <int>
            }

        Raises:
            ConnectionError if the server is not running.
        """
        return self._get("/status")
