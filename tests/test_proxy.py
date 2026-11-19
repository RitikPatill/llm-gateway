import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.main import app

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
async def test_upstream_error_passthrough(monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", UPSTREAM)
    monkeypatch.setenv("UPSTREAM_API_KEY", "test-key")

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
