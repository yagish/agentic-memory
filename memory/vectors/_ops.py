from __future__ import annotations

import math
import struct


def pack_vector(vector: list[float]) -> bytes:
    """Pack a list of floats into a compact binary blob for SQLite storage.

    Uses little-endian float32 — the format sqlite-vec's vec_distance_cosine() expects.
    """
    return struct.pack(f"<{len(vector)}f", *vector)


def cosine_distance(a: list[float], b_blob: bytes) -> float:
    """Compute cosine distance between vector a and a packed blob b_blob.

    Returns 0.0 for identical vectors, up to 2.0 for opposite. Lower = more similar.
    Implemented in Python because macOS system Python doesn't support sqlite extension loading.
    """
    n = len(b_blob) // 4
    b = struct.unpack(f"<{n}f", b_blob)
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)
