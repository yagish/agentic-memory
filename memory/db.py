# db.py — the storage layer for the memory system.
# Everything that touches the SQLite database lives here.
# Other modules call these functions; they never write SQL themselves.

import json       # used to convert Python dicts ↔ text for storage
import math       # used for cosine similarity calculation (sqrt, dot product)
import sqlite3    # Python's built-in SQLite driver — no install needed
import struct     # used to pack float32 arrays into bytes for SQLite blob storage
from datetime import datetime, timezone, timedelta   # for timestamp arithmetic in consolidation helpers
import uuid                                           # for generating UUIDs in new Phase 13 helpers

# Try to import SentenceTransformer — the library that converts text into vectors.
# If not installed, embed() will raise a clear ImportError with a helpful message.
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# _model starts as None and is loaded on the first call to embed().
# We defer loading so that importing db.py doesn't pay the ~2-second startup
# cost on every hook invocation that doesn't need embeddings.
_model = None

# The embedding model we use. all-MiniLM-L6-v2 is ~90MB, runs fully locally,
# and produces 384-dimensional vectors of good quality for semantic search.
_MODEL_NAME = "all-MiniLM-L6-v2"

# Number of dimensions in each embedding vector.
# all-MiniLM-L6-v2 always outputs exactly 384 floats.
_DIMS = 384


# _SCHEMA defines the base tables. The IF NOT EXISTS guards make it safe to
# re-run every startup — it won't wipe existing data.
_SCHEMA = """
-- Main table: one row per conversation session.
-- transcript is stored as raw JSON text (verbatim — never summarised).
CREATE TABLE IF NOT EXISTS sessions (
  session_id   TEXT PRIMARY KEY,   -- unique ID Claude gives each conversation
  agent        TEXT NOT NULL DEFAULT 'claude',
  started_at   TEXT,               -- ISO timestamp of first turn
  updated_at   TEXT,               -- ISO timestamp of most recent save
  turn_count   INTEGER,            -- total number of turns (user + assistant)
  transcript   TEXT                -- full conversation as a JSON string
);

-- FTS5 virtual table: a full-text search index that mirrors the sessions table.
-- "content='sessions'" means SQLite keeps the index in sync via triggers below.
-- "session_id UNINDEXED" means session_id is carried along but not searched.
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
  USING fts5(session_id UNINDEXED, transcript, content='sessions');

-- Trigger: when a new session row is inserted, also add it to the FTS index.
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
  INSERT INTO sessions_fts(session_id, transcript) VALUES (new.session_id, new.transcript);
END;

-- Trigger: when a session row is updated (e.g. new turns added), refresh the FTS index.
-- We first delete the old FTS entry, then insert the updated one.
CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
  INSERT INTO sessions_fts(sessions_fts, session_id, transcript) VALUES ('delete', old.session_id, old.transcript);
  INSERT INTO sessions_fts(session_id, transcript) VALUES (new.session_id, new.transcript);
END;
"""

# _VEC_SCHEMA creates the vector storage table.
# It's a plain SQLite table — no extensions needed.
# We store embeddings as BLOB (binary data) and compute similarity in Python.
_VEC_SCHEMA = """
-- session_vecs: one 384-dimensional embedding vector per session.
-- The embedding column holds packed float32 bytes (384 floats × 4 bytes = 1536 bytes).
-- We read all blobs back into Python and compute cosine similarity there.
CREATE TABLE IF NOT EXISTS session_vecs (
    session_id  TEXT PRIMARY KEY,   -- matches sessions.session_id
    embedding   BLOB NOT NULL       -- packed little-endian float32 array
);
"""


# _RETRIEVAL_SCHEMA tracks every time Claude calls an MCP memory tool.
# This lets us report "how often is memory actually being used?" in the dashboard.
_RETRIEVAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS retrievals (
    id          TEXT PRIMARY KEY,   -- unique ID (UUID) for each retrieval event
    tool        TEXT NOT NULL,      -- which MCP tool was called, e.g. 'memory_search'
    query       TEXT,               -- the search query (NULL for memory_status)
    result_size INTEGER,            -- how many bytes were returned to Claude
    called_at   TEXT NOT NULL       -- ISO timestamp of the call
);
"""

# _CHUNK_SCHEMA stores sub-session chunks for finer-grained semantic search.
# Instead of one vector per whole session, we split the transcript into
# overlapping windows and store one vector per window (chunk).
# This lets semantic search find the right session even when the matching
# content is buried deep in a long conversation.
_CHUNK_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  id          TEXT PRIMARY KEY,   -- "{session_id}:{chunk_index}" — unique per chunk
  session_id  TEXT NOT NULL REFERENCES sessions(session_id),
  chunk_index INTEGER NOT NULL,   -- 0-based position of this chunk in the session
  text        TEXT NOT NULL,      -- concatenated turn text for this window
  embedding   BLOB,               -- packed float32 embedding, or NULL if not yet embedded
  created_at  TEXT                -- ISO UTC timestamp when this chunk was stored
);
"""

# _FACTS_SCHEMA stores structured facts that Claude saves proactively mid-conversation.
# Unlike sessions (which are full transcripts saved passively), facts are small,
# targeted pieces of information the agent deliberately records — e.g. a user preference,
# a key decision, or a domain fact worth remembering across sessions.
# The FTS5 virtual table + triggers keep the full-text search index in sync automatically.
_FACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id         TEXT PRIMARY KEY,   -- UUID, generated at insert time
  content    TEXT NOT NULL,      -- the fact text (e.g. "User prefers dark mode")
  tags       TEXT,               -- JSON array of tag strings, e.g. '["python","preferences"]'
  source     TEXT,               -- who created the fact: "agent" (MCP tool) or "manual"
  session_id TEXT,               -- optional: which session this fact came from
  created_at TEXT,               -- ISO UTC timestamp when the fact was first saved
  updated_at TEXT                -- ISO UTC timestamp of the most recent edit
);

-- FTS5 virtual table for full-text search over fact content and tags.
-- "content='facts'" means SQLite reads the text from the facts table via triggers;
-- the index itself is kept in sync by the three triggers below.
-- "id UNINDEXED" means the id is carried along for JOINs but not searched.
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
  USING fts5(id UNINDEXED, content, tags, content='facts');

-- Trigger: when a new fact row is inserted, add it to the FTS index.
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(id, content, tags) VALUES (new.id, new.content, new.tags);
END;

-- Trigger: when a fact row is updated, refresh the FTS index.
-- We delete the old entry first, then insert the new one.
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, id, content, tags) VALUES ('delete', old.id, old.content, old.tags);
  INSERT INTO facts_fts(id, content, tags) VALUES (new.id, new.content, new.tags);
END;

-- Trigger: when a fact row is deleted, remove it from the FTS index.
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, id, content, tags) VALUES ('delete', old.id, old.content, old.tags);
END;
"""


# _INSIGHTS_SCHEMA stores cross-session patterns discovered by the daemon (Phase 13).
# Insights accumulate over time — each call to upsert_insight adds a new row.
_INSIGHTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS insights (
  id           TEXT PRIMARY KEY,   -- UUID generated at insert time
  insight_type TEXT NOT NULL,      -- "pattern", "preference", or "skill"
  content      TEXT NOT NULL,      -- the insight text
  evidence     TEXT,               -- JSON array of session_ids that support this insight
  confidence   REAL,               -- 0.0–1.0 confidence score from the LLM
  created_at   TEXT,               -- ISO UTC timestamp when first recorded
  updated_at   TEXT                -- ISO UTC timestamp of last update
);
"""

# _TOPIC_CLUSTERS_SCHEMA stores the centroid and metadata for each topic cluster.
# Each cluster groups sessions whose transcript embeddings are nearby in vector space.
_TOPIC_CLUSTERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS topic_clusters (
  id           TEXT PRIMARY KEY,   -- UUID generated at cluster-creation time
  label        TEXT NOT NULL,      -- short human-readable label (first 40 chars of first session)
  centroid     BLOB,               -- packed float32 centroid embedding (rolling average of members)
  member_count INTEGER DEFAULT 0,  -- number of sessions assigned to this cluster
  updated_at   TEXT                -- ISO UTC timestamp of the last member update
);
"""

# _CLUSTER_MEMBERSHIPS_SCHEMA records which sessions belong to which cluster.
# A session can only appear once per cluster (PRIMARY KEY constraint).
_CLUSTER_MEMBERSHIPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS cluster_memberships (
  session_id  TEXT NOT NULL REFERENCES sessions(session_id),
  cluster_id  TEXT NOT NULL REFERENCES topic_clusters(id),
  distance    REAL,               -- cosine distance from the session embedding to the centroid
  PRIMARY KEY (session_id, cluster_id)
);
"""

# _SUMMARIES_SCHEMA stores LLM-generated summaries of old sessions (Phase 12).
# Once a summary exists, the raw transcript can safely be pruned to save space.
_SUMMARIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    summary     TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT 'llama3.2:3b',
    created_at  TEXT NOT NULL
);
"""


def log_retrieval(conn: sqlite3.Connection, tool: str, query: str, result_size: int) -> None:
    """
    Record one MCP tool call in the retrievals table.

    Called by mcp_server.py after every tool invocation so we have a real
    count of how often memory is used — not an estimate.

    Args:
        conn        — open connection from init_db()
        tool        — name of the MCP tool (e.g. 'memory_search')
        query       — the search string, or None if not applicable
        result_size — size of the returned result in bytes
    """
    import uuid                  # uuid generates a unique ID for each row
    from datetime import datetime, timezone

    # datetime.now(timezone.utc) gives the current time in UTC as an aware datetime.
    # .isoformat() converts it to a string like "2026-08-16T10:00:00+00:00".
    conn.execute(
        """
        INSERT INTO retrievals (id, tool, query, result_size, called_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (str(uuid.uuid4()), tool, query, result_size, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _cosine_distance(a: list[float], b_blob: bytes) -> float:
    """
    Compute cosine distance between vector `a` and a packed blob `b_blob`.

    Cosine distance = 1 - cosine_similarity.
    Range: 0.0 (identical direction) to 2.0 (opposite direction).
    Lower is more similar — we sort by distance ascending.

    We do this in Python because macOS system Python doesn't support
    SQLite extension loading (enable_load_extension is unavailable),
    so we can't use sqlite-vec's vec_distance_cosine() SQL function.
    For a personal memory system with hundreds of sessions, Python-side
    math is fast enough — each dot product over 384 floats takes microseconds.
    """
    # Unpack the binary blob back into a list of 384 floats.
    n = len(b_blob) // 4          # 4 bytes per float32
    b = struct.unpack(f"<{n}f", b_blob)

    # Dot product: sum of element-wise products.
    dot = sum(x * y for x, y in zip(a, b))

    # Magnitudes (Euclidean norms).
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))

    if mag_a == 0 or mag_b == 0:
        return 1.0  # treat zero vectors as maximally distant

    # cosine_similarity = dot / (|a| * |b|)
    # cosine_distance   = 1 - cosine_similarity
    return 1.0 - dot / (mag_a * mag_b)


def init_db(path: str) -> sqlite3.Connection:
    """
    Open (or create) the SQLite database at `path` and ensure the schema exists.

    Pass ":memory:" for path to create a temporary in-memory database —
    useful for tests because nothing is written to disk.

    Returns a connection object you pass to the other functions.
    """
    conn = sqlite3.connect(path)

    # row_factory makes each result row behave like a dict (row["column_name"])
    # instead of a plain tuple (row[0]). Much easier to work with.
    conn.row_factory = sqlite3.Row

    # Run the core schema — sessions table + FTS5 index + triggers.
    conn.executescript(_SCHEMA)
    conn.commit()

    # Create the vector table. This is a plain SQLite table (no extension needed).
    # Cosine similarity is computed in Python inside semantic_search().
    conn.executescript(_VEC_SCHEMA)
    conn.commit()

    # Create the retrievals table for tracking MCP tool usage.
    conn.executescript(_RETRIEVAL_SCHEMA)
    conn.commit()

    # Create the chunks table for sub-session semantic search (Phase 7).
    # This follows the same pattern as _VEC_SCHEMA above.
    conn.executescript(_CHUNK_SCHEMA)
    conn.commit()

    # Create the facts table for proactive structured fact storage (Phase 8).
    # This follows the same pattern as _CHUNK_SCHEMA above.
    conn.executescript(_FACTS_SCHEMA)
    conn.commit()

    # Create the summaries table for LLM-generated session summaries (Phase 12).
    # Summaries are stored here once ollama has processed an old session.
    conn.executescript(_SUMMARIES_SCHEMA)
    conn.commit()

    # Create Phase 13 tables: insights, topic_clusters, cluster_memberships.
    conn.executescript(_INSIGHTS_SCHEMA)
    conn.commit()
    conn.executescript(_TOPIC_CLUSTERS_SCHEMA)
    conn.commit()
    conn.executescript(_CLUSTER_MEMBERSHIPS_SCHEMA)
    conn.commit()

    # Add daemon_processed_at column to sessions if not already present.
    # SQLite's ALTER TABLE does not support IF NOT EXISTS, so we check
    # the pragma first to avoid errors on an existing database.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "daemon_processed_at" not in existing_cols:
        # This column is stamped by mark_session_processed() once the daemon
        # has fully processed a session (extracted facts, assigned cluster).
        conn.execute("ALTER TABLE sessions ADD COLUMN daemon_processed_at TEXT")
    conn.commit()

    return conn


def upsert_session(
    conn: sqlite3.Connection,
    session_id: str,
    agent: str,
    transcript: list[dict],   # list of {"role": "user"/"assistant", "content": "..."}
    started_at: str,          # ISO 8601 timestamp, e.g. "2026-08-16T10:00:00Z"
    updated_at: str,
) -> None:
    """
    Save (or update) a conversation session in the database.

    If session_id already exists, the transcript and metadata are overwritten
    with the new values. This is called an "upsert" (update + insert).

    The full transcript list is serialised to a JSON string for storage —
    SQLite stores it as text and we decode it back to a list when reading.
    """
    # json.dumps converts the Python list of dicts into a JSON string for storage.
    transcript_json = json.dumps(transcript)

    # Each element in the transcript list is one turn, so len() gives total turns.
    turn_count = len(transcript)

    conn.execute(
        """
        INSERT INTO sessions (session_id, agent, started_at, updated_at, turn_count, transcript)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
          agent      = excluded.agent,
          updated_at = excluded.updated_at,
          turn_count = excluded.turn_count,
          transcript = excluded.transcript
        """,
        # The ?s above are placeholders; SQLite fills them in from this tuple.
        # Using placeholders (not string formatting) prevents SQL injection.
        (session_id, agent, started_at, updated_at, turn_count, transcript_json),
    )
    conn.commit()  # flush the write to disk


def search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """
    Full-text search across all saved transcripts.

    Returns up to `limit` sessions that contain the query terms, ranked by
    relevance (best match first). Each result includes a short excerpt
    (snippet) showing where the match was found.

    Example:
        results = search(conn, "quantum entanglement")
        # → [{"session_id": "abc", "agent": "claude", "updated_at": "...", "snippet": "..."}]
    """
    rows = conn.execute(
        """
        SELECT
          s.session_id,
          s.agent,
          s.updated_at,
          -- snippet() is a built-in FTS5 function that extracts the matching
          -- portion of text. The '[' and ']' wrap the matched words.
          -- 16 is the number of surrounding words to include for context.
          snippet(sessions_fts, 1, '[', ']', '...', 16) AS snippet
        FROM sessions_fts
        -- JOIN pulls the real session row so we get agent and updated_at.
        JOIN sessions s ON s.session_id = sessions_fts.session_id
        WHERE sessions_fts MATCH ?   -- MATCH is FTS5's search operator
        ORDER BY rank                -- rank is FTS5's built-in relevance score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    # Convert each sqlite3.Row object into a plain dict for easier use by callers.
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Phase 5 additions: embedding + semantic (vector) search
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """
    Convert a text string into a 384-dimensional vector of floats.

    Semantically similar texts produce similar vectors — even with different
    words. "machine learning" and "neural networks" will be close in vector
    space, so semantic_search() can find relevant sessions without exact
    keyword matches.

    The model is loaded on first call and cached for the session lifetime.
    First call takes ~2 seconds (model load); subsequent calls are fast.

    Args:
        text — any string (transcript text, search query, etc.)

    Returns:
        A list of 384 floats — the embedding vector.

    Raises:
        ImportError if sentence-transformers is not installed.
    """
    if not _ST_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip3 install sentence-transformers"
        )

    # _model is declared at module level; 'global' lets us reassign it here.
    global _model
    if _model is None:
        # SentenceTransformer downloads the model on first use (~90MB).
        # After that it's cached in ~/.cache/huggingface and loads instantly.
        _model = SentenceTransformer(_MODEL_NAME)

    # encode() returns a numpy array; .tolist() converts it to a plain Python list
    # of floats, which is easier to pass around without the numpy dependency.
    return _model.encode(text).tolist()


def _pack_vector(vector: list[float]) -> bytes:
    """
    Pack a list of floats into a compact binary blob for SQLite storage.

    SQLite has no native float-array type, so we serialize the vector ourselves.
    The format is little-endian float32 — this is the format that sqlite-vec's
    vec_distance_cosine() function expects.

    Example: [0.1, 0.2] → 8 bytes of binary data
    """
    # struct.pack format breakdown:
    #   '<'  = little-endian byte order (required by sqlite-vec)
    #   'f'  = single-precision (32-bit) float
    #   repeated len(vector) times
    return struct.pack(f"<{len(vector)}f", *vector)


def store_embedding(conn: sqlite3.Connection, session_id: str, vector: list[float]) -> None:
    """
    Save (or replace) the vector embedding for a session.

    Called by the save hook after upsert_session(), so the stored vector
    always reflects the latest transcript content.

    Does nothing if the session_vecs table doesn't exist (i.e., sqlite-vec
    failed to load) — so FTS5-only deployments aren't affected.

    Args:
        conn       — open connection from init_db()
        session_id — must match an existing row in sessions
        vector     — 384-float list from embed()
    """
    # Check whether the vector table was created (requires sqlite-vec).
    # If not, silently skip — don't break the calling code.
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='session_vecs'"
    ).fetchone()
    if not table_check:
        return

    # Pack the float list into the binary format sqlite-vec understands.
    blob = _pack_vector(vector)

    # INSERT OR REPLACE: if a vector for this session already exists, overwrite it.
    conn.execute(
        """
        INSERT OR REPLACE INTO session_vecs (session_id, embedding)
        VALUES (?, ?)
        """,
        (session_id, blob),
    )
    conn.commit()


def semantic_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """
    Find sessions whose content is semantically similar to the query.

    Unlike FTS5 keyword search, semantic search understands meaning —
    it can match "machine learning" to sessions about "neural networks",
    or "broken authentication" to sessions about "login bug".

    Results are ordered by cosine distance (lower = more similar).
    A distance of 0.0 means identical; 2.0 means maximally different.

    Cosine similarity is computed in Python (not via a SQLite extension)
    so this works on any platform, including macOS with system Python.

    Args:
        conn  — open connection from init_db()
        query — natural-language search string
        limit — max results to return (default 10)

    Returns:
        List of dicts with keys: session_id, agent, updated_at, distance
    """
    # Embed the query text into a 384-dimensional vector.
    query_vector = embed(query)

    # Fetch all stored embeddings plus their session metadata in one query.
    # For a personal memory system (hundreds of sessions), this is fast.
    # Each row: session_id, agent, updated_at, embedding blob.
    rows = conn.execute(
        """
        SELECT sv.session_id, s.agent, s.updated_at, sv.embedding
        FROM session_vecs sv
        JOIN sessions s ON s.session_id = sv.session_id
        """
    ).fetchall()

    if not rows:
        return []

    # Compute cosine distance for every stored session.
    scored = []
    for row in rows:
        dist = _cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append({
            "session_id": row["session_id"],
            "agent":      row["agent"],
            "updated_at": row["updated_at"],
            "distance":   dist,
        })

    # Sort by distance (closest first) and return up to `limit` results.
    scored.sort(key=lambda r: r["distance"])
    return scored[:limit]


# ---------------------------------------------------------------------------
# Phase 7 additions: sub-session chunking + chunk-level semantic search
# ---------------------------------------------------------------------------

def chunk_transcript(turns: list[dict], window: int = 6, overlap: int = 1) -> list[str]:
    """
    Split a transcript into overlapping text windows (chunks).

    Instead of embedding the whole session as one blob, we slide a window
    of `window` turns across the transcript, advancing by (window - overlap)
    turns each step. Each window becomes one chunk string. This gives
    semantic search a finer-grained target — a query about topic X can match
    the specific chunk where X was discussed, not just the overall session.

    Args:
        turns   — list of {"role": "user"/"assistant", "content": "..."}
        window  — how many turns to include in each chunk (default 6)
        overlap — how many turns to repeat between consecutive chunks (default 1)

    Returns:
        List of strings, one per chunk. Always returns at least [""] so that
        callers never have to handle an empty list.
    """
    # Filter out turns whose content is not a plain string.
    # Tool call turns can have list/dict content — we skip those silently
    # because they add noise and aren't human-readable text.
    text_turns = [t for t in turns if isinstance(t.get("content"), str)]

    # If there are no usable turns, return a single empty string.
    # Callers should still store this chunk so the session is represented.
    if not text_turns:
        return [""]

    # step = how far we advance the window start between chunks.
    # overlap=1 means the last 1 turn of chunk N is the first turn of chunk N+1.
    step = window - overlap

    # Build each chunk by concatenating turn text with role prefixes.
    chunks = []
    start = 0
    while start < len(text_turns):
        # Slice the window — may be smaller than `window` at the end of the transcript.
        window_turns = text_turns[start : start + window]

        # Build the chunk string: "user: ...\nassistant: ...\n" for each turn.
        lines = [f"{t['role']}: {t['content']}" for t in window_turns]
        chunk_text = "\n".join(lines)
        chunks.append(chunk_text)

        # If the current window already reaches the end of the transcript,
        # there are no new turns for the next chunk — stop.
        # Without this guard a 6-turn transcript with window=6 would produce
        # a second chunk containing only the overlap turn, which adds no value.
        if start + window >= len(text_turns):
            break

        # Advance by step. If step <= 0 the caller passed bad args; clamp to 1
        # to avoid an infinite loop.
        start += max(step, 1)

    return chunks


def store_chunk(
    conn: sqlite3.Connection,
    session_id: str,
    chunk_index: int,
    text: str,
    embedding: list[float] | None,
) -> None:
    """
    Insert or replace one chunk row in the chunks table.

    The chunk's primary key is "{session_id}:{chunk_index}" so that
    re-running the hook for the same session replaces old chunks
    rather than accumulating duplicates.

    Args:
        conn        — open connection from init_db()
        session_id  — the session this chunk belongs to
        chunk_index — 0-based position of this chunk in the session
        text        — the concatenated turn text for this window
        embedding   — 384-float list, or None if embedding was skipped
    """
    from datetime import datetime, timezone

    # Build the composite primary key — unique per (session, chunk position).
    chunk_id = f"{session_id}:{chunk_index}"

    # Convert the float list to a binary blob, or use None if no embedding.
    blob = _pack_vector(embedding) if embedding is not None else None

    # created_at records when this chunk was stored — useful for debugging.
    created_at = datetime.now(timezone.utc).isoformat()

    # INSERT OR REPLACE: if the row already exists (same id), overwrite it.
    # This is the upsert pattern used throughout db.py.
    conn.execute(
        """
        INSERT OR REPLACE INTO chunks (id, session_id, chunk_index, text, embedding, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (chunk_id, session_id, chunk_index, text, blob, created_at),
    )
    conn.commit()


def get_chunks_for_session(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """
    Return all chunks for a session, ordered by chunk_index ascending.

    Used by tests and by semantic_search_chunks() to inspect stored chunks.

    Args:
        conn       — open connection from init_db()
        session_id — the session to fetch chunks for

    Returns:
        List of dicts with keys: id, session_id, chunk_index, text, embedding, created_at.
        Returns [] if no chunks exist for this session.
    """
    rows = conn.execute(
        """
        SELECT id, session_id, chunk_index, text, embedding, created_at
        FROM chunks
        WHERE session_id = ?
        ORDER BY chunk_index ASC
        """,
        (session_id,),
    ).fetchall()

    # Convert each sqlite3.Row to a plain dict so callers can use dict syntax.
    return [dict(r) for r in rows]


def delete_chunks_for_session(conn: sqlite3.Connection, session_id: str) -> None:
    """
    Delete all chunks belonging to a session.

    Called before re-chunking a session so stale chunks from the previous
    save don't accumulate. Silently does nothing if no chunks exist yet.

    Args:
        conn       — open connection from init_db()
        session_id — whose chunks to remove
    """
    conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
    conn.commit()


def semantic_search_chunks(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """
    Find sessions whose content is semantically similar to the query,
    searching at chunk granularity rather than whole-session granularity.

    This produces more precise results than session-level search because a
    50-turn conversation is split into overlapping 6-turn windows. A query
    about topic X matches the window where X was actually discussed, not a
    blended average of the whole session.

    Results are one entry per chunk (not per session). Use the session_id
    field to group them. Returns top `limit` chunks by cosine distance.

    Args:
        conn  — open connection from init_db()
        query — natural-language search string
        limit — max results to return (default 10)

    Returns:
        List of dicts with keys: session_id, chunk_index, distance, text, snippet
        Raises ImportError if sentence-transformers is not installed.
    """
    if not _ST_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip3 install sentence-transformers"
        )

    # Embed the query into a vector so we can compare it against chunk vectors.
    query_vector = embed(query)

    # Fetch all chunk rows that have an embedding stored.
    # Chunks without embeddings (embedding IS NULL) are skipped — they can't
    # participate in cosine distance computation.
    rows = conn.execute(
        """
        SELECT session_id, chunk_index, text, embedding
        FROM chunks
        WHERE embedding IS NOT NULL
        """
    ).fetchall()

    if not rows:
        return []

    # Compute cosine distance between the query vector and each chunk embedding.
    scored = []
    for row in rows:
        dist = _cosine_distance(query_vector, bytes(row["embedding"]))
        scored.append({
            "session_id":  row["session_id"],
            "chunk_index": row["chunk_index"],
            "distance":    dist,
            "text":        row["text"],
            # snippet is a preview of the chunk — first 200 characters.
            "snippet":     row["text"][:200],
        })

    # Sort by distance ascending (closest match first), then slice to limit.
    scored.sort(key=lambda r: r["distance"])
    return scored[:limit]


# ---------------------------------------------------------------------------
# Phase 8 additions: structured fact storage (CRUD + FTS5 search)
# ---------------------------------------------------------------------------

def insert_fact(
    conn: sqlite3.Connection,
    content: str,
    tags: list[str] | None = None,
    source: str = "manual",
    session_id: str | None = None,
) -> str:
    """
    Save a new structured fact to the database and return its generated ID.

    Facts are small, deliberately saved pieces of information — a user preference,
    a key decision, or a domain fact worth recalling across sessions.

    Args:
        conn       — open connection from init_db()
        content    — the fact text (e.g. "User prefers dark mode")
        tags       — optional list of tag strings for categorisation
        source     — "manual" (default) or "agent" (saved via MCP tool)
        session_id — optional: the session this fact was observed in

    Returns:
        The UUID string assigned to the new fact row.
    """
    import uuid                  # uuid generates a unique, collision-safe ID
    from datetime import datetime, timezone

    # Generate a new UUID to serve as the primary key for this fact.
    fact_id = str(uuid.uuid4())

    # Tags are stored as a JSON array string so SQLite can hold them in one column.
    # json.dumps([]) produces '[]' for an empty list — never NULL.
    tags_json = json.dumps(tags or [])

    # Both created_at and updated_at start at the same timestamp — the moment of creation.
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO facts (id, content, tags, source, session_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (fact_id, content, tags_json, source, session_id, now, now),
    )
    conn.commit()  # flush the write to disk

    return fact_id


def update_fact(
    conn: sqlite3.Connection,
    fact_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """
    Update one or more fields of an existing fact.

    Only the fields passed as non-None are changed. updated_at is always
    refreshed to the current UTC time so callers can see when an edit occurred.

    Args:
        conn     — open connection from init_db()
        fact_id  — the UUID of the fact to update
        content  — new text for the fact, or None to leave it unchanged
        tags     — new tag list, or None to leave tags unchanged

    Returns:
        True if the fact was found and updated, False if fact_id does not exist.
    """
    from datetime import datetime, timezone

    # Always stamp the current time on update regardless of what else changed.
    now = datetime.now(timezone.utc).isoformat()

    # Build the SET clause dynamically — only include fields that were supplied.
    # Starting with updated_at means we always have at least one field to set.
    set_clauses = ["updated_at = ?"]
    values: list = [now]

    if content is not None:
        # Caller wants to change the fact text.
        set_clauses.append("content = ?")
        values.append(content)

    if tags is not None:
        # Caller wants to change the tags — re-encode to JSON for storage.
        set_clauses.append("tags = ?")
        values.append(json.dumps(tags))

    # Append fact_id last — it goes into the WHERE clause.
    values.append(fact_id)

    cursor = conn.execute(
        f"UPDATE facts SET {', '.join(set_clauses)} WHERE id = ?",
        values,
    )
    conn.commit()

    # rowcount is the number of rows the UPDATE touched.
    # 0 means no row with this id existed; anything > 0 means success.
    return cursor.rowcount > 0


def delete_fact(conn: sqlite3.Connection, fact_id: str) -> bool:
    """
    Delete one fact from the database.

    The FTS5 trigger (facts_ad) automatically removes the corresponding
    entry from the search index when the row is deleted.

    Args:
        conn    — open connection from init_db()
        fact_id — the UUID of the fact to remove

    Returns:
        True if the fact was found and deleted, False if fact_id does not exist.
    """
    cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    conn.commit()

    # rowcount = 0 → no row with this id existed; > 0 → deleted successfully.
    return cursor.rowcount > 0


def search_facts(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """
    Full-text search across all stored facts.

    Searches the content and tags columns using FTS5 (same engine as session search).
    Results are ranked by relevance and include a highlighted snippet showing
    where the match was found.

    Args:
        conn  — open connection from init_db()
        query — the words or phrase to search for
        limit — maximum results to return (default 10)

    Returns:
        List of dicts with keys: id, content, tags (as list), source, session_id,
        created_at, updated_at, snippet.
    """
    rows = conn.execute(
        """
        SELECT
          f.id,
          f.content,
          f.tags,
          f.source,
          f.session_id,
          f.created_at,
          f.updated_at,
          -- snippet() extracts the matching portion of text with highlights.
          -- Column index 1 is 'content' in the facts_fts virtual table definition.
          -- '[' and ']' wrap matched words; 16 is the surrounding-word context count.
          snippet(facts_fts, 1, '[', ']', '...', 16) AS snippet
        FROM facts_fts
        -- JOIN pulls the real fact row so we get all columns including source.
        JOIN facts f ON f.id = facts_fts.id
        WHERE facts_fts MATCH ?   -- MATCH is FTS5's search operator
        ORDER BY rank             -- rank is FTS5's built-in relevance score
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        # Convert the stored JSON string back to a Python list for callers.
        # "[]" is the default for facts with no tags, so this is always safe.
        d["tags"] = json.loads(d["tags"] or "[]")
        result.append(d)
    return result


def list_facts(
    conn: sqlite3.Connection,
    tag: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    List stored facts, optionally filtered to a specific tag.

    Unlike search_facts() (which does keyword matching), this is a
    structured filter — it returns all facts that carry the exact tag string,
    ordered by most recently updated first.

    Args:
        conn  — open connection from init_db()
        tag   — if given, only return facts whose tags list contains this string
        limit — maximum results to return (default 50)

    Returns:
        List of dicts with keys: id, content, tags (as list), source, session_id,
        created_at, updated_at. (No snippet — this is a listing, not a search.)
    """
    if tag is not None:
        # json_each() expands the JSON array in the tags column into individual rows.
        # We filter to only the rows where one of those values equals our tag.
        # This is more reliable than a LIKE query, which could partially match
        # a tag that contains another tag as a substring.
        rows = conn.execute(
            """
            SELECT f.id, f.content, f.tags, f.source, f.session_id, f.created_at, f.updated_at
            FROM facts f, json_each(f.tags) je
            WHERE je.value = ?
            ORDER BY f.updated_at DESC
            LIMIT ?
            """,
            (tag, limit),
        ).fetchall()
    else:
        # No tag filter — return all facts, newest first.
        rows = conn.execute(
            """
            SELECT id, content, tags, source, session_id, created_at, updated_at
            FROM facts
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        # Parse the JSON tag array back into a Python list for callers.
        d["tags"] = json.loads(d["tags"] or "[]")
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Phase 11: Hybrid search — Reciprocal Rank Fusion of FTS5 + semantic results
# ---------------------------------------------------------------------------

def hybrid_search(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[dict]:
    """
    Combine full-text (FTS5) and semantic (chunk-level) search using
    Reciprocal Rank Fusion (RRF) to produce a single ranked result list.

    RRF scores each session based on its rank in each individual result list.
    A session that ranks highly in BOTH lists gets a higher combined score than
    one that only appears in one list, making the results more robust.

    This is the recommended default retrieval tool — it almost always
    outperforms keyword-only or semantic-only search on its own.

    Args:
        conn  — open connection from init_db()
        query — natural-language search string
        limit — maximum results to return (default 10)

    Returns:
        List of dicts with keys: session_id, agent, updated_at, snippet, rrf_score
        Same shape as search() so callers do not need to branch on result type.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Gather FTS5 keyword results (fetch more than limit so RRF   #
    # has a broad pool to fuse from).                                      #
    # ------------------------------------------------------------------ #
    fts_results = search(conn, query, limit=limit * 3)

    # ------------------------------------------------------------------ #
    # Step 2: Gather semantic chunk results.                               #
    # If sentence-transformers is not installed, treat as empty list so   #
    # hybrid_search degrades gracefully to FTS5-only.                    #
    # ------------------------------------------------------------------ #
    try:
        # Fetch extra chunks so deduplication still leaves plenty of sessions.
        chunk_results = semantic_search_chunks(conn, query, limit=limit * 3)
    except ImportError:
        # sentence-transformers not installed — skip semantic leg entirely.
        chunk_results = []

    # ------------------------------------------------------------------ #
    # Step 3: Deduplicate semantic results by session_id.                 #
    # Keep only the best-scoring chunk (lowest distance) per session.     #
    # ------------------------------------------------------------------ #
    # best_chunk maps session_id → the chunk dict with the lowest distance.
    best_chunk: dict[str, dict] = {}
    for chunk in chunk_results:
        sid = chunk["session_id"]
        # If we haven't seen this session yet, or this chunk is closer, keep it.
        if sid not in best_chunk or chunk["distance"] < best_chunk[sid]["distance"]:
            best_chunk[sid] = chunk

    # Convert the dedup map to an ordered list (already sorted by distance
    # because semantic_search_chunks returns them sorted).
    sem_results = sorted(best_chunk.values(), key=lambda r: r["distance"])

    # ------------------------------------------------------------------ #
    # Step 4: Build rank lookup tables for RRF computation.               #
    # fts_rank[session_id]  = 0-based position in fts_results             #
    # sem_rank[session_id]  = 0-based position in sem_results             #
    # ------------------------------------------------------------------ #
    # enumerate() yields (index, item) pairs starting at 0.
    fts_rank = {r["session_id"]: i for i, r in enumerate(fts_results)}
    sem_rank  = {r["session_id"]: i for i, r in enumerate(sem_results)}

    # ------------------------------------------------------------------ #
    # Step 5: Compute RRF score for every session that appeared in        #
    # at least one result list.                                           #
    # RRF formula: score += 1 / (60 + rank + 1)                          #
    # The constant 60 dampens the effect of low ranks (standard value).  #
    # ------------------------------------------------------------------ #
    # Collect the union of all session IDs seen in either result list.
    all_session_ids = set(fts_rank.keys()) | set(sem_rank.keys())

    # rrf_scores maps session_id → accumulated RRF score.
    rrf_scores: dict[str, float] = {}
    for sid in all_session_ids:
        score = 0.0
        # Add FTS5 contribution if this session appeared in keyword results.
        if sid in fts_rank:
            score += 1.0 / (60 + fts_rank[sid] + 1)
        # Add semantic contribution if this session appeared in chunk results.
        if sid in sem_rank:
            score += 1.0 / (60 + sem_rank[sid] + 1)
        rrf_scores[sid] = score

    # ------------------------------------------------------------------ #
    # Step 6: Sort sessions by RRF score (highest first), take top limit. #
    # ------------------------------------------------------------------ #
    # sorted() returns a new list; reverse=True puts the highest score first.
    ranked_ids = sorted(rrf_scores.keys(), key=lambda sid: rrf_scores[sid], reverse=True)
    top_ids = ranked_ids[:limit]

    # ------------------------------------------------------------------ #
    # Step 7: Build index structures for fast metadata + snippet lookup.  #
    # ------------------------------------------------------------------ #
    # FTS5 results already carry agent, updated_at, and snippet.
    fts_by_id   = {r["session_id"]: r for r in fts_results}
    # Semantic results carry the chunk snippet (first 200 chars of the chunk).
    sem_by_id   = {r["session_id"]: r for r in sem_results}

    # For sessions that only appeared semantically (not in FTS5), we need to
    # fetch agent and updated_at from the sessions table directly.
    sem_only_ids = [sid for sid in top_ids if sid not in fts_by_id]
    sessions_meta: dict[str, dict] = {}
    if sem_only_ids:
        # Build a comma-separated placeholder string: "?,?,?" for len ids.
        placeholders = ",".join("?" * len(sem_only_ids))
        meta_rows = conn.execute(
            f"SELECT session_id, agent, updated_at FROM sessions WHERE session_id IN ({placeholders})",
            sem_only_ids,
        ).fetchall()
        # Store as dict for O(1) lookup below.
        for row in meta_rows:
            sessions_meta[row["session_id"]] = dict(row)

    # ------------------------------------------------------------------ #
    # Step 8: Assemble final result list.                                 #
    # ------------------------------------------------------------------ #
    output = []
    for sid in top_ids:
        # Prefer FTS5 snippet because it has highlighted keywords ([word]).
        # Fall back to the semantic chunk snippet if this session was not in FTS5.
        if sid in fts_by_id:
            fts_row = fts_by_id[sid]
            agent      = fts_row["agent"]
            updated_at = fts_row["updated_at"]
            snippet    = fts_row["snippet"]
        else:
            # Session came only from semantic search — look up metadata.
            meta = sessions_meta.get(sid, {})
            agent      = meta.get("agent", "")
            updated_at = meta.get("updated_at", "")
            snippet    = sem_by_id[sid]["snippet"]

        output.append({
            "session_id": sid,
            "agent":      agent,
            "updated_at": updated_at,
            "snippet":    snippet,
            "rrf_score":  rrf_scores[sid],   # useful for debugging / ranking transparency
        })

    return output


# ---------------------------------------------------------------------------
# Phase 12 additions: summaries table helpers for consolidation and decay
# ---------------------------------------------------------------------------

def insert_summary(conn: sqlite3.Connection, session_id: str, summary: str, model: str) -> None:
    """
    Store a generated summary for a session in the summaries table.

    Uses INSERT OR REPLACE so that if a summary already exists for this
    session_id (e.g. from a previous run), it is overwritten with the new one.

    Args:
        conn       — open connection from init_db()
        session_id — the session whose transcript was summarised
        summary    — the generated summary text (2-3 sentences)
        model      — the ollama model that generated the summary, e.g. "llama3.2:3b"
    """
    # datetime.now(timezone.utc).isoformat() gives an ISO 8601 UTC timestamp.
    conn.execute(
        "INSERT OR REPLACE INTO summaries (session_id, summary, model, created_at) VALUES (?, ?, ?, ?)",
        (session_id, summary, model, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def get_summary(conn: sqlite3.Connection, session_id: str) -> str | None:
    """
    Return the summary text for a session, or None if not yet summarised.

    Args:
        conn       — open connection from init_db()
        session_id — the session to look up

    Returns:
        The summary string if one exists, or None if no summary has been stored.
    """
    row = conn.execute(
        "SELECT summary FROM summaries WHERE session_id = ?", (session_id,)
    ).fetchone()
    # row is None if no summary exists; row["summary"] is the text if it does.
    return row["summary"] if row else None


def sessions_needing_summary(conn: sqlite3.Connection, days_threshold: int) -> list[dict]:
    """
    Return sessions older than days_threshold that have no summary yet.

    A session qualifies if:
      - Its updated_at is before the cutoff date (older than days_threshold)
      - It has no row in the summaries table yet (LEFT JOIN + NULL check)
      - Its transcript is non-null and non-empty (not '[]')

    Results are ordered oldest-first so the most urgent sessions are processed first.

    Args:
        conn           — open connection from init_db()
        days_threshold — sessions older than this many days are returned

    Returns:
        List of dicts with keys: session_id, updated_at, turn_count, transcript.
    """
    # timedelta(days=days_threshold) subtracts that many days from the current time.
    # .isoformat() converts to a string like "2026-07-17T10:00:00+00:00" for SQL comparison.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).isoformat()

    rows = conn.execute(
        """
        SELECT s.session_id, s.updated_at, s.turn_count, s.transcript
        FROM sessions s
        LEFT JOIN summaries su ON su.session_id = s.session_id
        WHERE s.updated_at < ? AND su.session_id IS NULL
          AND s.transcript IS NOT NULL AND s.transcript != '[]'
        ORDER BY s.updated_at ASC
        """,
        (cutoff,),
    ).fetchall()

    # Convert each sqlite3.Row to a plain dict for easy access by callers.
    return [dict(r) for r in rows]


def sessions_needing_prune(conn: sqlite3.Connection, days_threshold: int) -> list[dict]:
    """
    Return sessions older than days_threshold that have a summary and a non-null transcript.

    A session is ready for pruning if:
      - Its updated_at is before the cutoff date (older than days_threshold)
      - It has a row in the summaries table (INNER JOIN — summary must exist first)
      - Its transcript is non-null and non-empty (there is still data to prune)

    Args:
        conn           — open connection from init_db()
        days_threshold — sessions older than this many days are returned

    Returns:
        List of dicts with keys: session_id, updated_at, turn_count.
    """
    # Build the cutoff timestamp the same way as sessions_needing_summary.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).isoformat()

    rows = conn.execute(
        """
        SELECT s.session_id, s.updated_at, s.turn_count
        FROM sessions s
        INNER JOIN summaries su ON su.session_id = s.session_id
        WHERE s.updated_at < ?
          AND s.transcript IS NOT NULL AND s.transcript != '[]'
        ORDER BY s.updated_at ASC
        """,
        (cutoff,),
    ).fetchall()

    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Phase 13 additions: daemon processing, insights, and topic clustering
# ---------------------------------------------------------------------------

def get_unprocessed_sessions(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """
    Return up to `limit` sessions that the daemon has not yet processed.

    A session is considered unprocessed when daemon_processed_at IS NULL,
    which is its initial state after being saved by the hook.

    Args:
        conn  — open connection from init_db()
        limit — maximum number of sessions to return (default 10)

    Returns:
        List of dicts with at least session_id, transcript, updated_at.
    """
    # ORDER BY updated_at ASC processes oldest sessions first so nothing ages out.
    rows = conn.execute(
        """
        SELECT session_id, transcript, updated_at, turn_count
        FROM sessions
        WHERE daemon_processed_at IS NULL
          AND transcript IS NOT NULL AND transcript != '[]'
        ORDER BY updated_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    # Convert sqlite3.Row objects to plain dicts so callers can use dict syntax.
    return [dict(r) for r in rows]


def mark_session_processed(conn: sqlite3.Connection, session_id: str) -> None:
    """
    Stamp daemon_processed_at with the current UTC time for a session.

    Called by the daemon after it has extracted facts and assigned the session
    to a topic cluster, so the session is not re-processed on the next run.

    Args:
        conn       — open connection from init_db()
        session_id — the session to stamp
    """
    # datetime.now(timezone.utc).isoformat() gives a standard UTC timestamp string.
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE sessions SET daemon_processed_at = ? WHERE session_id = ?",
        (now, session_id),
    )
    conn.commit()


def upsert_insight(
    conn: sqlite3.Connection,
    insight_type: str,
    content: str,
    evidence: list[str],
    confidence: float,
) -> str:
    """
    Insert a new insight row and return its UUID.

    Insights always accumulate — every call creates a new row.  The caller
    is responsible for deduplication (e.g. by checking content before calling).

    Args:
        conn         — open connection from init_db()
        insight_type — "pattern", "preference", or "skill"
        content      — the insight text (e.g. "User consistently uses Python")
        evidence     — list of session_ids that support this insight
        confidence   — 0.0–1.0 confidence score (from the LLM)

    Returns:
        The UUID string assigned to the new insight row.
    """
    # Generate a unique ID for this insight.
    insight_id = str(uuid.uuid4())

    # Store the evidence list as a JSON array for portability.
    evidence_json = json.dumps(evidence or [])

    # Both timestamps start at the moment of creation.
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO insights (id, insight_type, content, evidence, confidence, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (insight_id, insight_type, content, evidence_json, confidence, now, now),
    )
    conn.commit()
    return insight_id


def list_insights(
    conn: sqlite3.Connection,
    insight_type: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """
    Return stored insights, optionally filtered by type.

    Args:
        conn         — open connection from init_db()
        insight_type — optional filter: "pattern", "preference", "skill", or "topic_cluster";
                       pass None to return all types
        limit        — maximum number of insights to return (default 20)

    Returns:
        List of dicts with keys: id, insight_type, content, evidence (as list),
        confidence, created_at, updated_at. Newest first.
    """
    if insight_type is not None:
        # Filter to only the requested type.
        rows = conn.execute(
            """
            SELECT id, insight_type, content, evidence, confidence, created_at, updated_at
            FROM insights
            WHERE insight_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (insight_type, limit),
        ).fetchall()
    else:
        # No filter — return all insight types.
        rows = conn.execute(
            """
            SELECT id, insight_type, content, evidence, confidence, created_at, updated_at
            FROM insights
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result = []
    for row in rows:
        d = dict(row)
        # Parse the JSON evidence list back into a Python list.
        d["evidence"] = json.loads(d["evidence"] or "[]")
        result.append(d)
    return result


def assign_to_cluster(
    conn: sqlite3.Connection,
    session_id: str,
    embedding: list[float],
    label: str = "",
) -> str:
    """
    Assign a session to the nearest topic cluster, or create a new one.

    Steps:
      1. Load all existing clusters (id, centroid, member_count).
      2. Compute cosine distance from embedding to each centroid.
      3. If best distance < 0.3, assign to that cluster; else create a new cluster.
      4. Update centroid as a rolling average: (old * old_count + new) / (old_count + 1).
      5. Upsert a row in cluster_memberships.
      6. Update topic_clusters.member_count and centroid.

    Args:
        conn       — open connection from init_db()
        session_id — the session being clustered
        embedding  — 384-float embedding of the session transcript
        label      — optional label for a newly-created cluster

    Returns:
        The cluster_id (UUID) that the session was assigned to.
    """
    # ------------------------------------------------------------------ #
    # Step 1: Load all existing clusters.                                 #
    # ------------------------------------------------------------------ #
    cluster_rows = conn.execute(
        "SELECT id, centroid, member_count, label FROM topic_clusters"
    ).fetchall()

    # ------------------------------------------------------------------ #
    # Step 2: Find the nearest cluster by cosine distance.                #
    # ------------------------------------------------------------------ #
    # COSINE_THRESHOLD defines the maximum distance for two embeddings to
    # be considered "the same topic".  0.3 is a sensible default.
    COSINE_THRESHOLD = 0.3

    best_cluster_id = None
    best_distance = float("inf")

    for row in cluster_rows:
        centroid_blob = bytes(row["centroid"])
        dist = _cosine_distance(embedding, centroid_blob)
        if dist < best_distance:
            best_distance = dist
            best_cluster_id = row["id"]

    # ------------------------------------------------------------------ #
    # Step 3: Assign to existing cluster or create a new one.             #
    # ------------------------------------------------------------------ #
    now = datetime.now(timezone.utc).isoformat()

    if best_cluster_id is not None and best_distance < COSINE_THRESHOLD:
        # The session is close enough to an existing cluster — join it.
        cluster_id = best_cluster_id
    else:
        # No close cluster found — create a new one with this embedding as centroid.
        cluster_id = str(uuid.uuid4())
        # Use the supplied label (fallback to first 40 chars of session_id).
        cluster_label = label[:40] if label else session_id[:40]
        initial_centroid = _pack_vector(embedding)
        conn.execute(
            """
            INSERT INTO topic_clusters (id, label, centroid, member_count, updated_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (cluster_id, cluster_label, initial_centroid, now),
        )

    # ------------------------------------------------------------------ #
    # Step 4: Update centroid as a rolling average.                       #
    # ------------------------------------------------------------------ #
    row = conn.execute(
        "SELECT centroid, member_count FROM topic_clusters WHERE id = ?",
        (cluster_id,),
    ).fetchone()

    old_count = row["member_count"]
    old_centroid_blob = bytes(row["centroid"])

    # Unpack the old centroid from its binary blob.
    n = len(old_centroid_blob) // 4         # 4 bytes per float32
    old_centroid = list(struct.unpack(f"<{n}f", old_centroid_blob))

    # Rolling average: new_centroid[i] = (old[i] * old_count + new[i]) / (old_count + 1)
    new_count = old_count + 1
    new_centroid = [
        (old_centroid[i] * old_count + embedding[i]) / new_count
        for i in range(len(embedding))
    ]
    new_centroid_blob = _pack_vector(new_centroid)

    # ------------------------------------------------------------------ #
    # Step 5: Upsert the cluster_memberships row.                         #
    # ------------------------------------------------------------------ #
    conn.execute(
        """
        INSERT OR REPLACE INTO cluster_memberships (session_id, cluster_id, distance)
        VALUES (?, ?, ?)
        """,
        (session_id, cluster_id, best_distance if best_cluster_id == cluster_id else 0.0),
    )

    # ------------------------------------------------------------------ #
    # Step 6: Update the cluster's centroid and member_count.             #
    # ------------------------------------------------------------------ #
    conn.execute(
        """
        UPDATE topic_clusters
        SET centroid = ?, member_count = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_centroid_blob, new_count, now, cluster_id),
    )
    conn.commit()

    return cluster_id


def get_cluster_sessions(conn: sqlite3.Connection, cluster_id: str) -> list[str]:
    """
    Return all session_ids assigned to a given cluster.

    Args:
        conn       — open connection from init_db()
        cluster_id — UUID of the cluster to query

    Returns:
        List of session_id strings. Empty list if the cluster has no members.
    """
    rows = conn.execute(
        "SELECT session_id FROM cluster_memberships WHERE cluster_id = ?",
        (cluster_id,),
    ).fetchall()
    # Extract the session_id string from each Row object.
    return [row["session_id"] for row in rows]


def get_clusters(conn: sqlite3.Connection) -> list[dict]:
    """
    Return all topic clusters.

    Args:
        conn — open connection from init_db()

    Returns:
        List of dicts with keys: id, label, member_count, updated_at.
        The centroid blob is excluded — use assign_to_cluster for centroid access.
    """
    rows = conn.execute(
        """
        SELECT id, label, member_count, updated_at
        FROM topic_clusters
        ORDER BY member_count DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def prune_transcript(conn: sqlite3.Connection, session_id: str) -> int:
    """
    Null out the transcript for a session by setting it to '[]'.

    Called after a summary has been generated and stored. The session row
    itself is preserved (metadata like updated_at and turn_count stay intact)
    but the raw transcript text is discarded to save space.

    Args:
        conn       — open connection from init_db()
        session_id — the session whose transcript should be cleared

    Returns:
        The old turn_count value (before pruning) so callers can log it.
    """
    # Fetch the old turn_count before we wipe the transcript.
    row = conn.execute(
        "SELECT turn_count FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    # Default to 0 if the session doesn't exist (shouldn't happen in normal use).
    turn_count = row["turn_count"] if row else 0

    # Set transcript to '[]' — an empty JSON array — rather than NULL.
    # This keeps the column type consistent and avoids NULL checks in other queries.
    conn.execute(
        "UPDATE sessions SET transcript = '[]' WHERE session_id = ?", (session_id,)
    )
    conn.commit()

    return turn_count
