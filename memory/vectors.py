"""Vector operations and embeddings.

This module handles all numerical vector operations:
- Converting text to embeddings (via sentence-transformers)
- Packing/unpacking vectors to/from binary format
- Computing cosine distance between vectors

This layer is independent of the database and can be tested/mocked
separately from storage operations.
"""

from __future__ import annotations

import math
import os
import struct

# Force Hugging Face/Transformers offline mode for this process.
# HF_HUB_OFFLINE=1: disables HTTP calls in huggingface_hub (metadata/file fetches).
# TRANSFORMERS_OFFLINE=1: prevents Transformers from attempting online resolution.
# We use setdefault() so callers can still override explicitly before import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Try to import SentenceTransformer — the library that converts text into vectors.
# If not installed, embed() will raise a clear ImportError with a helpful message.
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# _model starts as None and is loaded on the first call to embed().
# We defer loading so that importing this module doesn't pay the ~2-second startup
# cost on every hook invocation that doesn't need embeddings.
_model = None

# The embedding model we use. all-MiniLM-L6-v2 is ~90MB, runs fully locally,
# and produces 384-dimensional vectors of good quality for semantic search.
_MODEL_NAME = "all-MiniLM-L6-v2"

# Number of dimensions in each embedding vector.
# all-MiniLM-L6-v2 always outputs exactly 384 floats.
_DIMS = 384


def embed(text: str) -> list[float]:
    """
    Convert a text string into a 384-dimensional vector of floats.

    Semantically similar texts produce similar vectors — even with different
    words. "machine learning" and "neural networks" will be close in vector
    space, so semantic search can find relevant content without exact
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
        # local_files_only=True guarantees no Hugging Face network calls.
        # The model must already exist in local cache (default: ~/.cache/huggingface).
        try:
            _model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "Embedding model not available in local cache while offline mode is enabled. "
                "Preload sentence-transformers/all-MiniLM-L6-v2 into ~/.cache/huggingface "
                "before running this service."
            ) from exc

    # encode() returns a numpy array; .tolist() converts it to a plain Python list
    # of floats, which is easier to pass around without the numpy dependency.
    return _model.encode(text).tolist()


def pack_vector(vector: list[float]) -> bytes:
    """
    Pack a list of floats into a compact binary blob for SQLite storage.

    SQLite has no native float-array type, so we serialize the vector ourselves.
    The format is little-endian float32 — this is the format that sqlite-vec's
    vec_distance_cosine() function expects.

    Example: [0.1, 0.2] → 8 bytes of binary data

    Args:
        vector — list of floats to pack

    Returns:
        Binary blob suitable for BLOB column storage
    """
    # struct.pack format breakdown:
    #   '<'  = little-endian byte order (required by sqlite-vec)
    #   'f'  = single-precision (32-bit) float
    #   repeated len(vector) times
    return struct.pack(f"<{len(vector)}f", *vector)


def cosine_distance(a: list[float], b_blob: bytes) -> float:
    """
    Compute cosine distance between vector `a` and a packed blob `b_blob`.

    Cosine distance = 1 - cosine_similarity.
    Range: 0.0 (identical direction) to 2.0 (opposite direction).
    Lower is more similar — sort by distance ascending.

    We do this in Python because macOS system Python doesn't support
    SQLite extension loading (enable_load_extension is unavailable),
    so we can't use sqlite-vec's vec_distance_cosine() SQL function.
    For a personal memory system with hundreds of sessions, Python-side
    math is fast enough — each dot product over 384 floats takes microseconds.

    Args:
        a       — query vector as list of floats
        b_blob  — stored vector as binary blob (from pack_vector)

    Returns:
        Cosine distance as float (0.0 = identical, 2.0 = opposite)
    """
    # Unpack the binary blob back into a list of floats.
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
