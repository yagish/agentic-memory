"""Write-only HTTP endpoint for session ingestion.

This server now supports only the retained session-ingest flow.
Read/query functionality lives in dashboard_server.py and the wake-up hook.
"""

from __future__ import annotations

import copy
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG

from integrations.common import build_recall_response, retrieve_prompt_memory
from memory.db import bootstrap_db, open_db
from memory.ingest_pipeline import ingest_session
from memory.logger import error_log, log_memory_answer
from memory.debug import enable_debug
from memory.inference import embed_text
from memory.vectors import _MODEL_NAME as EMBED_MODEL_NAME


DB_PATH = os.path.expanduser("~/.memory/memory.db")
PORT = int(os.environ.get("MEMORY_INGEST_PORT", "7747"))
_EMBED_MODEL_READY = False
_EMBED_MODEL_ERROR = ""


def _timestamped_excepthook(exc_type, exc, tb) -> None:
    import traceback

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for line in traceback.format_exception(exc_type, exc, tb):
        for part in line.rstrip("\n").splitlines():
            sys.stderr.write(f"{stamp} ERROR ingest {part}\n")


sys.excepthook = _timestamped_excepthook

_UVICORN_LOG_CONFIG = copy.deepcopy(UVICORN_LOGGING_CONFIG)
_UVICORN_LOG_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
_UVICORN_LOG_CONFIG["formatters"]["default"]["datefmt"] = "%Y-%m-%dT%H:%M:%S"
_UVICORN_LOG_CONFIG["formatters"]["access"]["fmt"] = (
    "%(asctime)s %(levelprefix)s %(client_addr)s - \"%(request_line)s\" %(status_code)s"
)
_UVICORN_LOG_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%dT%H:%M:%S"

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _EMBED_MODEL_READY, _EMBED_MODEL_ERROR

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = bootstrap_db(DB_PATH)
    conn.close()

    try:
        embed_text("memory recall warmup")
    except Exception as exc:
        _EMBED_MODEL_READY = False
        _EMBED_MODEL_ERROR = str(exc)
    else:
        _EMBED_MODEL_READY = True
        _EMBED_MODEL_ERROR = ""

    yield


app = FastAPI(title="Memory Ingest Server", version="2.0.0", lifespan=_lifespan)


def _get_conn():
    if not os.path.exists(DB_PATH) or os.path.getsize(DB_PATH) == 0:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return bootstrap_db(DB_PATH)
    return open_db(DB_PATH)


_default_origins = "https://copilot.microsoft.com,https://github.com"
_origins_raw = os.environ.get("MEMORY_INGEST_CORS_ORIGINS", _default_origins)
_allowed_origins = [origin.strip() for origin in _origins_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class Turn(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, value: str) -> str:
        if value not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{value}'")
        return value


class IngestRequest(BaseModel):
    session_id: str
    agent: str = "unknown"
    turns: list[Turn]
    started_at: str | None = None
    metadata: dict | None = None

    @field_validator("turns")
    @classmethod
    def turns_must_not_be_empty(cls, value: list) -> list:
        if not value:
            raise ValueError("turns must not be empty")
        return value


class RecallRequest(BaseModel):
    prompt: str
    include_working_memory: bool = False
    session_id: str | None = None
    agent: str | None = None


@app.get("/status")
def get_status() -> dict:
    return {
        "status": "ok",
        "db_path": DB_PATH,
        "db_exists": os.path.exists(DB_PATH),
        "embed_model_ready": _EMBED_MODEL_READY,
        "embed_model_name": EMBED_MODEL_NAME,
        "embed_model_error": _EMBED_MODEL_ERROR,
    }


@app.post("/recall")
def post_recall(request: RecallRequest) -> dict:
    prompt = request.prompt.strip()
    if not prompt:
        return {"action": "noop"}

    conn = _get_conn()
    try:
        context = retrieve_prompt_memory(
            conn,
            prompt,
            include_working_memory=request.include_working_memory,
            session_id=request.session_id,
        )
        response = build_recall_response(prompt, context)
        if response.get("action") == "answer":
            log_memory_answer(
                "recall_server",
                prompt=prompt,
                answer=str(response.get("answer", "")),
                session_id=request.session_id,
                agent=request.agent,
                response=response,
            )
        return response
    except Exception as exc:
        error_log("ingest", f"unhandled error in POST /recall: {exc}", exc=exc)
        raise
    finally:
        conn.close()


@app.post("/ingest")
def post_ingest(request: IngestRequest) -> dict:
    conn = _get_conn()
    try:
        started_at = request.started_at or datetime.now(timezone.utc).isoformat()
        updated_at = datetime.now(timezone.utc).isoformat()
        outcome = ingest_session(
            conn,
            session_id=request.session_id,
            agent=request.agent,
            turns=[turn.model_dump() for turn in request.turns],
            started_at=started_at,
            updated_at=updated_at,
            metadata=request.metadata,
        )
        return {
            "ok": True,
            "session_id": request.session_id,
            "turns_stored": outcome.turn_count,
            "warnings": [warning.__dict__ for warning in outcome.warnings],
        }
    except Exception as exc:
        error_log("ingest", f"unhandled error in POST /ingest: {exc}", exc=exc)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    enable_debug("ingest")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info",
        log_config=_UVICORN_LOG_CONFIG,
    )
