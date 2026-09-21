"""Deep ingest module for session ingestion.

Adapters should hand parsed session turns to this module instead of re-implementing
session upsert, embedding, and chunk replacement themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.db import upsert_session
from memory.llm.inference import embed_text


@dataclass(frozen=True)
class IngestWarning:
    """A non-fatal issue encountered while ingesting a session."""

    stage: str
    message: str


@dataclass(frozen=True)
class IngestOutcome:
    """Structured result for one session ingestion run."""

    session_id: str
    turn_count: int
    chunk_count: int
    embedding_stored: bool
    warnings: list[IngestWarning] = field(default_factory=list)


def ingest_session(
    conn,
    *,
    session_id: str,
    agent: str,
    turns: list[dict],
    started_at: str,
    updated_at: str,
    metadata: dict | None = None,
    project_context: dict | None = None,
    embed_fn=embed_text,
) -> IngestOutcome:
    """Store a session only.

    The database was simplified to keep just sessions, facts, and episodic
    memory. Session ingest therefore persists the transcript and skips the old
    response-cache/chunk/session-vector side tables.
    """
    del embed_fn

    upsert_session(
        conn,
        session_id=session_id,
        agent=agent,
        transcript=turns,
        started_at=started_at,
        updated_at=updated_at,
        metadata=metadata,
        project_context=project_context,
    )

    return IngestOutcome(
        session_id=session_id,
        turn_count=len(turns),
        chunk_count=0,
        embedding_stored=False,
        warnings=[],
    )
