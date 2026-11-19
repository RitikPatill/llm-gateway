"""SQLite-backed semantic cache for LLM responses.

Cache key: concatenated user message contents (system prompts excluded so a
cached answer survives system-prompt tweaks without a cache miss).
Embeddings are stored as raw numpy float32 blobs via np.save/np.load.
"""
import io
import json
import sqlite3
import time

import numpy as np

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding  BLOB    NOT NULL,
    response   TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
"""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _serialize(embedding: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, embedding)
    return buf.getvalue()


def _deserialize(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


def init_cache(db_path: str, ttl_seconds: int) -> None:
    """Create table if not exists; evict rows older than ttl_seconds."""
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(
            "DELETE FROM cache WHERE created_at < ?",
            (time.time() - ttl_seconds,),
        )
        conn.commit()


def get_cached(
    db_path: str,
    embedding: np.ndarray,
    threshold: float,
) -> dict | None:
    """Return the best-matching cached response dict, or None on miss."""
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        rows = conn.execute("SELECT embedding, response FROM cache").fetchall()

    if not rows:
        return None

    best_sim = -1.0
    best_response = None
    for blob, response_json in rows:
        cached_emb = _deserialize(blob)
        sim = _cosine(embedding, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_response = response_json

    if best_sim >= threshold:
        return json.loads(best_response)
    return None


def put_cached(
    db_path: str,
    embedding: np.ndarray,
    response: dict,
) -> None:
    """Insert a new (embedding, response, timestamp) row."""
    blob = _serialize(embedding.astype(np.float32))
    with sqlite3.connect(db_path, check_same_thread=False) as conn:
        conn.execute(
            "INSERT INTO cache (embedding, response, created_at) VALUES (?, ?, ?)",
            (blob, json.dumps(response), time.time()),
        )
        conn.commit()
