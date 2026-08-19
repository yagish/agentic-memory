"""Deep ingest module for session ingestion.

Adapters should hand parsed session turns to this module instead of re-implementing
session upsert, embedding, and chunk replacement themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.db import (
    chunk_transcript,
    delete_chunks_for_session,
    store_chunk,
    store_embedding,
    upsert_session,
)
from memory.inference import embed_text


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
    embed_fn=embed_text,
) -> IngestOutcome:
    """Store a session, its full-session embedding, and its chunk embeddings.

    Session storage is authoritative. Embedding and chunking failures are treated as
    non-fatal warnings so adapters can keep user-facing flows unblocked.
    """
    warnings: list[IngestWarning] = []

    upsert_session(
        conn,
        session_id=session_id,
        agent=agent,
        transcript=turns,
        started_at=started_at,
        updated_at=updated_at,
        metadata=metadata,
    )

    embedding_stored = False
    full_text = " ".join(
        turn.get("content", "")
        for turn in turns
        if isinstance(turn.get("content"), str)
    ).strip()

    if full_text:
        try:
            store_embedding(conn, session_id, embed_fn(full_text))
            embedding_stored = True
        except Exception as exc:
            warnings.append(IngestWarning("embedding", str(exc)))

    chunk_count = 0
    try:
        chunk_texts = chunk_transcript(turns)
        delete_chunks_for_session(conn, session_id)
        for chunk_index, chunk_text in enumerate(chunk_texts):
            chunk_vector = embed_fn(chunk_text) if chunk_text.strip() else None
            store_chunk(conn, session_id, chunk_index, chunk_text, chunk_vector)
        chunk_count = len(chunk_texts)
    except Exception as exc:
        warnings.append(IngestWarning("chunking", str(exc)))

    return IngestOutcome(
        session_id=session_id,
        turn_count=len(turns),
        chunk_count=chunk_count,
        embedding_stored=embedding_stored,
        warnings=warnings,
    )
