from __future__ import annotations

import contextlib
import io
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMS = 384


def embed(text: str) -> list[float]:
    """Convert text to a 384-dimensional embedding vector.

    Model is loaded on first call and cached for the process lifetime.
    """
    if not _ST_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip3 install sentence-transformers"
        )
    global _model
    if _model is None:
        try:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                _model = SentenceTransformer(_MODEL_NAME, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "Embedding model not available in local cache while offline mode is enabled. "
                "Preload sentence-transformers/all-MiniLM-L6-v2 into ~/.cache/huggingface "
                "before running this service."
            ) from exc
    return _model.encode(text).tolist()
