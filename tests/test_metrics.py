"""Unit tests for MetricsStore and the /metrics HTTP endpoint."""

import pytest
from fastapi.testclient import TestClient

from llm_gateway.main import app
from llm_gateway.metrics import MetricsStore


# ---------------------------------------------------------------------------
# Unit tests — MetricsStore
# ---------------------------------------------------------------------------


def _fresh() -> MetricsStore:
    """Return a new, zeroed MetricsStore for each test."""
    return MetricsStore()


def test_initial_state():
    store = _fresh()
    snap = store.snapshot()
    assert snap["total_requests"] == 0
    assert snap["cache_hits"] == 0
    assert snap["tokens_saved"] == 0
    assert snap["hit_rate"] == 0.0
    assert snap["p50_latency_ms"] == 0.0
    assert snap["p95_latency_ms"] == 0.0
    assert store.hit_rate == 0.0


def test_record_miss():
    store = _fresh()
    store.record_request(latency_ms=120.0, cache_hit=False, model="gpt-4o-mini")
    snap = store.snapshot()
    assert snap["total_requests"] == 1
    assert snap["cache_hits"] == 0
    assert store.hit_rate == 0.0


def test_record_hit():
    store = _fresh()
    store.record_request(latency_ms=5.0, cache_hit=True, model="gpt-4o-mini", tokens_saved=100)
    assert store.hit_rate == 1.0
    snap = store.snapshot()
    assert snap["cache_hits"] == 1
    assert snap["tokens_saved"] == 100


def test_hit_rate_mixed():
    store = _fresh()
    store.record_request(latency_ms=10.0, cache_hit=True, model="m")
    store.record_request(latency_ms=10.0, cache_hit=False, model="m")
    store.record_request(latency_ms=10.0, cache_hit=False, model="m")
    store.record_request(latency_ms=10.0, cache_hit=False, model="m")
    assert store.hit_rate == pytest.approx(0.25)


def test_p50_p95():
    store = _fresh()
    for i in range(1, 101):
        store.record_request(latency_ms=float(i), cache_hit=False, model="m")
    # numpy percentile of 1..100: p50 ≈ 50.5, p95 ≈ 95.05
    assert store.p_latency(50) == pytest.approx(50.5, abs=1.0)
    assert store.p_latency(95) == pytest.approx(95.05, abs=1.0)


def test_tokens_saved_accumulate():
    store = _fresh()
    store.record_request(latency_ms=5.0, cache_hit=True, model="m", tokens_saved=200)
    store.record_request(latency_ms=5.0, cache_hit=True, model="m", tokens_saved=300)
    assert store.snapshot()["tokens_saved"] == 500


def test_recent_log_capped():
    store = _fresh()
    for i in range(60):
        store.record_request(latency_ms=float(i), cache_hit=False, model="m")
    # recent_log(20) must return at most 20 entries
    log = store.recent_log(20)
    assert len(log) == 20
    # The internal deque is capped at 50; recent_log(20) returns last 20 of those 50
    assert all("ts" in entry for entry in log)


# ---------------------------------------------------------------------------
# Integration test — /metrics HTTP endpoint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    expected_keys = {
        "total_requests",
        "cache_hits",
        "hit_rate",
        "p50_latency_ms",
        "p95_latency_ms",
        "tokens_saved",
        "recent_log",
    }
    assert expected_keys.issubset(data.keys())
