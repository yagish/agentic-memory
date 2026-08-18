# ingest_server.py — Write-only HTTP endpoint for agent session ingestion.
#
# This server's only job is to accept session writes from other agents.
# Dashboard, monitoring, and memory read queries are in dashboard_server.py (port 7748).
#
# Routes:
#   POST /ingest  — store a full session transcript from any agent
#   POST /recall  — return a memory wake-up digest for a prompt (agent-facing)
#   POST /answer  — direct-answer lookup for repeated prompts (agent-facing)
#
# Run:
#   python3 memory/ingest_server.py          # listens on localhost:7747
#   MEMORY_INGEST_PORT=8080 python3 ...      # custom port

import json       # JSON serialisation/deserialisation
import os         # file paths, environment variables
import sqlite3    # graceful FTS error handling in /recall
import sys        # module search path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import uvicorn

from memory.db import (
    init_db,
    upsert_session,
    embed,
    store_embedding,
    chunk_transcript,
    delete_chunks_for_session,
    store_chunk,
    semantic_search_chunks,
    search_facts,
    list_insights,
    find_direct_answer,
    log_retrieval,
)
from memory.logger import activity_log, error_log

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH       = os.path.expanduser("~/.memory/memory.db")
IDENTITY_PATH = os.path.expanduser("~/.memory/identity.md")

# /recall tuning — how many results to pull from each search.
RECENT_SESSION_COUNT = 5
CHUNK_SEARCH_LIMIT   = 5
FACT_SEARCH_LIMIT    = 5

PORT = int(os.environ.get("MEMORY_INGEST_PORT", "7747"))

app = FastAPI(title="Memory Ingest Server", version="1.0.0")

# ── CORS — let browser-based agents POST from known domains ──────────────────
# Override with MEMORY_INGEST_CORS_ORIGINS (comma-separated) if needed.
_default_origins = "https://copilot.microsoft.com,https://github.com"
_origins_raw = os.environ.get("MEMORY_INGEST_CORS_ORIGINS", _default_origins)
_allowed_origins = [o.strip() for o in _origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


# ── Request models ────────────────────────────────────────────────────────────

class Turn(BaseModel):
    """One turn in a conversation — a single message from user or assistant."""
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v


class IngestRequest(BaseModel):
    session_id: str
    agent: str = "unknown"
    turns: list[Turn]
    started_at: str | None = None
    metadata: dict | None = None

    @field_validator("turns")
    @classmethod
    def turns_must_not_be_empty(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("turns must not be empty")
        return v


class RecallRequest(BaseModel):
    query: str = ""
    session_id: str | None = None


class AnswerRequest(BaseModel):
    query: str
    session_id: str | None = None
    min_score: float = 0.93

    @field_validator("min_score")
    @classmethod
    def score_must_be_unit_interval(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("min_score must be between 0 and 1")
        return v


# ── Helpers for /recall ───────────────────────────────────────────────────────

def _read_identity() -> str:
    """Read ~/.memory/identity.md, returning empty string if absent."""
    try:
        with open(IDENTITY_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def _fetch_recent_sessions(conn, session_id: str | None) -> list[dict]:
    """Return the N most recent sessions, excluding the current one."""
    rows = conn.execute(
        """
        SELECT session_id, updated_at, turn_count, transcript
        FROM sessions
        WHERE (? IS NULL OR session_id != ?)
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (session_id, session_id, RECENT_SESSION_COUNT),
    ).fetchall()

    results = []
    for row in rows:
        try:
            turns = json.loads(row["transcript"] or "[]")
            first = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
            preview = first[:100].replace("\n", " ")
        except Exception:
            preview = ""
        results.append({
            "session_id": row["session_id"],
            "updated_at": row["updated_at"],
            "turn_count": row["turn_count"],
            "first_user_message": preview,
        })
    return results


def _fetch_relevant_context(conn, query: str, current_session_id: str | None) -> tuple[list[dict], list[dict]]:
    """Semantic chunk search + FTS fact search for a query."""
    chunk_results = semantic_search_chunks(conn, query, limit=CHUNK_SEARCH_LIMIT)

    best_chunks: dict[str, dict] = {}
    for chunk in chunk_results:
        sid = chunk["session_id"]
        if current_session_id and sid == current_session_id:
            continue
        if sid not in best_chunks or chunk["distance"] < best_chunks[sid]["distance"]:
            best_chunks[sid] = chunk

    sessions = []
    for sid, chunk in best_chunks.items():
        row = conn.execute(
            "SELECT updated_at, turn_count FROM sessions WHERE session_id = ?", (sid,)
        ).fetchone()
        if row is None:
            continue
        similarity = round((1 - chunk["distance"] / 2) * 100, 1)
        sessions.append({
            "session_id": sid,
            "updated_at":  row["updated_at"],
            "turn_count":  row["turn_count"],
            "snippet":     chunk["snippet"],
            "similarity":  similarity,
        })
    sessions.sort(key=lambda s: s["similarity"], reverse=True)

    facts: list[dict] = []
    try:
        facts = search_facts(conn, query, limit=FACT_SEARCH_LIMIT)
    except sqlite3.OperationalError:
        facts = []

    return sessions, facts


def _build_digest(
    identity: str,
    sessions: list[dict],
    facts: list[dict] | None,
    insights: list[dict] | None,
) -> str:
    """Format a wake-up digest that an agent can prepend to an LLM turn."""
    lines = ["=== MEMORY WAKE-UP ===", ""]

    lines.append("[L0 — Identity]")
    lines.append(identity or "(no identity.md found)")
    lines.append("")

    lines.append("[L1 — Relevant Sessions]" if facts is not None else "[L1 — Recent Sessions]")
    if sessions:
        for i, s in enumerate(sessions, 1):
            updated = s.get("updated_at") or "unknown"
            turns   = s.get("turn_count") or 0
            if "similarity" in s:
                lines.append(f"{i}. {updated} ({turns} turns, {s['similarity']}%) — {s.get('snippet','')}")
            else:
                lines.append(f"{i}. {updated} ({turns} turns) — {s.get('first_user_message','')}")
    else:
        lines.append("(no past sessions found)")
    lines.append("")

    if facts:
        lines.append("[L2 — Relevant Facts]")
        for f in facts:
            tags = f.get("tags", [])
            line = f"• {f.get('content','')}"
            if tags:
                line += f" [tags: {', '.join(tags)}]"
            lines.append(line)
        lines.append("")

    if insights:
        lines.append("[L3 — Learned Insights]")
        for ins in insights:
            lines.append(
                f"• {ins.get('content','')} ({ins.get('insight_type','pattern')}, "
                f"confidence {ins.get('confidence',0.0):.0%})"
            )
        lines.append("")

    lines.append("[System — Identity Updates]")
    lines.append(
        "If you learn something new and personal about the user during this conversation "
        "(name, role, tech stack, preferences, working style), update ~/.memory/identity.md "
        "silently using file-write tools."
    )
    lines.append("")
    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest")
def post_ingest(req: IngestRequest) -> dict:
    """Accept a session from any agent and run the full save pipeline."""
    try:
        turns_as_dicts = [{"role": t.role, "content": t.content} for t in req.turns]
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = init_db(DB_PATH)

        updated_at = datetime.now(timezone.utc).isoformat()
        started_at = req.started_at or updated_at

        upsert_session(
            conn,
            session_id=req.session_id,
            agent=req.agent,
            transcript=turns_as_dicts,
            started_at=started_at,
            updated_at=updated_at,
            metadata=req.metadata,
        )

        try:
            full_text = " ".join(t.content for t in req.turns if isinstance(t.content, str))
            if full_text.strip():
                store_embedding(conn, req.session_id, embed(full_text))
        except Exception as exc:
            error_log("ingest", f"embedding failed for {req.session_id}", exc=exc)

        try:
            chunk_texts = chunk_transcript(turns_as_dicts)
            delete_chunks_for_session(conn, req.session_id)
            for i, text in enumerate(chunk_texts):
                store_chunk(conn, req.session_id, i, text, embed(text) if text.strip() else None)
        except Exception as exc:
            error_log("ingest", f"chunking failed for {req.session_id}", exc=exc)

        conn.close()
        activity_log("ingest", "ingest", session=req.session_id, agent=req.agent, turns=len(req.turns))
        return {"ok": True, "session_id": req.session_id}

    except Exception as exc:
        error_log("ingest", "unhandled error in POST /ingest", exc=exc)
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.post("/recall")
def post_recall(req: RecallRequest) -> dict:
    """Build a memory wake-up digest for a prompt (agent-facing)."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = init_db(DB_PATH)
        query = (req.query or "").strip()

        identity = _read_identity()
        sessions: list[dict] = []
        facts: list[dict] | None = None
        mode = "relevance"

        if query:
            try:
                sessions, facts = _fetch_relevant_context(conn, query, req.session_id)
            except Exception:
                sessions, facts = [], None

        if not sessions:
            sessions = _fetch_recent_sessions(conn, req.session_id)
            facts, mode = None, "recency"

        try:
            insights = list_insights(conn, limit=3)
        except Exception:
            insights = []

        digest = _build_digest(identity, sessions, facts, insights)
        log_retrieval(conn, "memory_recall", query or None, len(digest))
        activity_log("ingest", "recall", session=req.session_id or "", mode=mode,
                     sessions=len(sessions), facts=len(facts or []))
        conn.close()

        return {
            "ok": True, "mode": mode, "digest": digest,
            "session_count": len(sessions), "fact_count": len(facts or []),
            "insight_count": len(insights or []),
        }
    except Exception as exc:
        error_log("ingest", "unhandled error in POST /recall", exc=exc)
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.post("/answer")
def post_answer(req: AnswerRequest) -> dict:
    """Try to answer a prompt directly from memory without calling the LLM."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = init_db(DB_PATH)
        match = find_direct_answer(conn, req.query, min_score=req.min_score,
                                   exclude_session_id=req.session_id)
        result = {"ok": True, "answered": match is not None, "match": match}
        log_retrieval(conn, "memory_direct_answer", req.query, len(json.dumps(result)))
        activity_log("ingest", "answer", session=req.session_id or "",
                     answered=match is not None,
                     similarity=match["similarity"] if match else 0)
        conn.close()
        return result
    except Exception as exc:
        error_log("ingest", "unhandled error in POST /answer", exc=exc)
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@app.post("/compress")
def post_compress() -> dict:
    """
    Compress all sessions into a structured memory document and delete them.

    Useful for agents and CLI tools that POST to the ingest server. The
    dashboard uses the same endpoint on dashboard_server (port 7748) instead.

    Returns the compressed content and session count on success.
    """
    from memory.compress import compress_memory  # noqa: PLC0415

    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = init_db(DB_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    try:
        result = compress_memory(conn)
        return {
            "ok": True,
            "sessions_compressed": result.sessions_compressed,
            "model": result.model,
            "created_at": result.created_at,
            "content": result.content,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    finally:
        conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
