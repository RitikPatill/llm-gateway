"""Unit tests for llm_gateway.cache.

All tests use a tmp_path SQLite file — no files persist between runs.
Embeddings are plain numpy float32 arrays; the sentence-transformer model
is never loaded here.
"""
import time

import numpy as np
import pytest

from llm_gateway.cache import get_cached, init_cache, put_cached

V1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal to V1
V3 = np.array([0.9950, 0.0998, 0.0], dtype=np.float32)  # ~cos 0.995 w.r.t. V1 (unnormalised)

RESPONSE = {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]}
THRESHOLD = 0.92


def _db(tmp_path) -> str:
    db = str(tmp_path / "test.db")
    init_cache(db, ttl_seconds=86400)
    return db


def test_cache_miss_returns_none(tmp_path):
    db = _db(tmp_path)
    result = get_cached(db, V1, THRESHOLD)
    assert result is None


def test_cache_put_and_hit(tmp_path):
    db = _db(tmp_path)
    put_cached(db, V1, RESPONSE)
    result = get_cached(db, V1, THRESHOLD)
    assert result is not None
    assert result["id"] == "chatcmpl-1"


def test_cache_hit_above_threshold(tmp_path):
    """Vector with cosine similarity ~0.995 against stored vector → hit."""
    db = _db(tmp_path)
    put_cached(db, V1, RESPONSE)
    # Normalise V3 so cosine sim is well-defined
    v3_norm = V3 / np.linalg.norm(V3)
    result = get_cached(db, v3_norm, THRESHOLD)
    assert result is not None


def test_cache_miss_below_threshold(tmp_path):
    """Orthogonal vector (cosine sim = 0) → miss."""
    db = _db(tmp_path)
    put_cached(db, V1, RESPONSE)
    result = get_cached(db, V2, THRESHOLD)
    assert result is None


def test_cache_ttl_eviction(tmp_path):
    """Row inserted with old timestamp is evicted by init_cache."""
    import sqlite3
    import json

    db_path = str(tmp_path / "evict.db")
    init_cache(db_path, ttl_seconds=86400)

    # Manually insert a row with an old timestamp (25 hours ago)
    old_ts = time.time() - 90_000
    from llm_gateway.cache import _serialize
    blob = _serialize(V1)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO cache (embedding, response, created_at) VALUES (?, ?, ?)",
            (blob, json.dumps(RESPONSE), old_ts),
        )
        conn.commit()

    # Verify row exists before eviction
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    assert count == 1

    # Re-run init_cache with 24 h TTL — old row should be evicted
    init_cache(db_path, ttl_seconds=86400)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
    assert count == 0
