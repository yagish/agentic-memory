# db.py — the storage layer for the memory system.
# Everything that touches the SQLite database lives here.
# Other modules call these functions; they never write SQL themselves.

import json       # used to convert Python dicts ↔ text for storage
import math       # used for cosine similarity calculation (sqrt, dot product)
import sqlite3    # Python's built-in SQLite driver — no install needed
import struct     # used to pack float32 arrays into bytes for SQLite blob storage

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
