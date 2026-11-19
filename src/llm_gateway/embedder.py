"""Singleton wrapper around SentenceTransformer.

The model is loaded once at startup (via warm-up call in lifespan) so the
~2 s initialisation cost is not visible on the first real request.
Thread safety: FastAPI + Uvicorn run in a single async thread; no lock needed.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> np.ndarray:
    """Return a 1-D float32 unit-norm embedding for text."""
    return get_embedder().encode(text, normalize_embeddings=True)
