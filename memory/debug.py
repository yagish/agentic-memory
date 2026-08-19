import os
import sys
from datetime import datetime, timezone


_DEBUG_LOG_PATH = os.path.expanduser("~/.memory/debug.log")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} [debug] {message}\n"
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(line)
    except Exception:
        pass

    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass


def enable_debug(service: str) -> bool:
    """
    Enable an optional debugpy listener for a script.

    Per-service environment variables:
      MEMORY_DEBUG_<SERVICE>_PORT   required to enable debugging
      MEMORY_DEBUG_<SERVICE>_HOST   optional, defaults to MEMORY_DEBUG_HOST or 127.0.0.1
      MEMORY_DEBUG_<SERVICE>_WAIT   optional, defaults to MEMORY_DEBUG_WAIT or 0

    Examples:
      MEMORY_DEBUG_DAEMON_PORT=5678
      MEMORY_DEBUG_DAEMON_WAIT=1
      MEMORY_DEBUG_HOST=127.0.0.1
    """
    service_key = service.upper()
    port_raw = os.environ.get(f"MEMORY_DEBUG_{service_key}_PORT")
    if not port_raw:
        return False

    host = os.environ.get(
        f"MEMORY_DEBUG_{service_key}_HOST",
        os.environ.get("MEMORY_DEBUG_HOST", "127.0.0.1"),
    )
    wait_for_client = _truthy(
        os.environ.get(
            f"MEMORY_DEBUG_{service_key}_WAIT",
            os.environ.get("MEMORY_DEBUG_WAIT", "0"),
        )
    )

    try:
        port = int(port_raw)
    except ValueError:
        _debug_log(f"{service}: invalid debug port {port_raw!r}")
        return False

    try:
        import debugpy
    except ImportError:
        _debug_log(
            f"{service}: debug requested on {host}:{port}, but debugpy is not installed"
        )
        return False

    try:
        debugpy.listen((host, port))
        _debug_log(f"{service}: debugpy listening on {host}:{port}")
        if wait_for_client:
            _debug_log(f"{service}: waiting for debugger attach")
            debugpy.wait_for_client()
            _debug_log(f"{service}: debugger attached")
        return True
    except Exception as exc:
        _debug_log(f"{service}: failed to enable debugpy on {host}:{port}: {exc}")
        return False
