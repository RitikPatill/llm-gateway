import json

import httpx
import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.main import app

SAMPLE_ROUTES_CONFIG = {
    "tiers": {
        "fast": {"model": "gpt-3.5-turbo", "description": "Simple queries"},
        "balanced": {"model": "gpt-4o-mini", "description": "Medium complexity"},
        "powerful": {"model": "gpt-4o", "description": "Reasoning, code, long context"},
    },
    "routing": {
        "fast_threshold": 30,
        "balanced_threshold": 70,
        "default_tier": "balanced",
    },
}

UPSTREAM = "https://api.openai.com/v1"
COMPLETIONS_URL = f"{UPSTREAM}/chat/completions"

NON_STREAM_BODY = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": False}).encode()
STREAM_BODY = json.dumps({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()

FAKE_RESPONSE = {"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello!"}}]}
SSE_CHUNKS = b"data: {}\n\ndata: [DONE]\n\n"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_non_streaming_proxy(monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-key")

    with respx.mock(assert_all_called=False) as mock:
        mock.post(COMPLETIONS_URL).mock(
            return_value=httpx.Response(200, json=FAKE_RESPONSE)
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                content=NON_STREAM_BODY,
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "chatcmpl-123"


@pytest.mark.asyncio
async def test_streaming_proxy(monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-key")

    with respx.mock(assert_all_called=False) as mock:
        mock.post(COMPLETIONS_URL).mock(
            return_value=httpx.Response(200, content=SSE_CHUNKS, headers={"content-type": "text/event-stream"})
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                content=STREAM_BODY,
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert b"data:" in resp.content


@pytest.mark.asyncio
async def test_missing_upstream_key_still_forwards(monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "")

    with respx.mock(assert_all_called=False) as mock:
        mock.post(COMPLETIONS_URL).mock(
            return_value=httpx.Response(200, json=FAKE_RESPONSE)
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                content=NON_STREAM_BODY,
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_upstream_error_passthrough(monkeypatch, tmp_path):
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DB_PATH", str(tmp_path / "err.db"))

    error_body = {"error": {"message": "Internal server error"}}

    with respx.mock(assert_all_called=False) as mock:
        mock.post(COMPLETIONS_URL).mock(
            return_value=httpx.Response(500, json=error_body)
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                content=NON_STREAM_BODY,
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == "Internal server error"


@pytest.mark.asyncio
async def test_cache_hit_skips_upstream(monkeypatch, tmp_path):
    """Second identical request is served from cache; upstream is called only once."""
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-key")
    monkeypatch.setenv("CACHE_DB_PATH", str(tmp_path / "test_cache.db"))
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.92")

    fixed_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr("llm_gateway.main.embed", lambda text: fixed_vector)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(COMPLETIONS_URL).mock(
            return_value=httpx.Response(200, json=FAKE_RESPONSE)
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # First request — cache miss, upstream called
            resp1 = await ac.post(
                "/v1/chat/completions",
                content=NON_STREAM_BODY,
                headers={"content-type": "application/json"},
            )
            # Second identical request — cache hit, upstream NOT called again
            resp2 = await ac.post(
                "/v1/chat/completions",
                content=NON_STREAM_BODY,
                headers={"content-type": "application/json"},
            )

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp2.json()["id"] == FAKE_RESPONSE["id"]
    # Upstream should have been called exactly once
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_router_downgrades_model(monkeypatch, tmp_path):
    """Simple prompt with model=gpt-4o is silently downgraded to gpt-3.5-turbo."""
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-key")
    # Fresh DB so no pre-existing embeddings pollute cache similarity checks
    monkeypatch.setenv("CACHE_DB_PATH", str(tmp_path / "router.db"))

    # Patch embed to return a fixed vector so the cache always misses
    fixed_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr("llm_gateway.main.embed", lambda text: fixed_vector)
    # Use a high similarity threshold so the fixed vector won't match itself
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.9999")

    captured: dict = {}

    def capture_request(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=FAKE_RESPONSE)

    simple_body = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }).encode()

    with respx.mock(assert_all_called=False) as mock:
        mock.post(COMPLETIONS_URL).mock(side_effect=capture_request)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            # Inject the sample routes config so routing is active
            app.state.routes_config = SAMPLE_ROUTES_CONFIG
            resp = await ac.post(
                "/v1/chat/completions",
                content=simple_body,
                headers={"content-type": "application/json"},
            )

    assert resp.status_code == 200
    # "hi" is a 1-word prompt → score ≤ 30 → fast tier → gpt-3.5-turbo
    assert captured.get("model") == "gpt-3.5-turbo"
