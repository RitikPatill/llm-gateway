import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.config import get_settings

load_dotenv()

_HOP_BY_HOP = {"transfer-encoding", "connection", "keep-alive", "te", "trailers", "upgrade", "host"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    settings = get_settings()
    body = await request.body()

    try:
        json_body = await request.json()
    except Exception:
        json_body = {}

    stream = bool(json_body.get("stream", False))
    upstream_url = settings.upstream_base_url.rstrip("/") + "/chat/completions"

    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    if settings.upstream_api_key:
        forward_headers["authorization"] = f"Bearer {settings.upstream_api_key}"

    client: httpx.AsyncClient = request.app.state.http_client
    start = time.monotonic()

    if stream:
        async def _stream_generator():
            async with client.stream(
                "POST",
                upstream_url,
                content=body,
                headers=forward_headers,
            ) as resp:
                latency = time.monotonic() - start
                ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
                print(
                    f"[{ts}] POST /v1/chat/completions → {upstream_url}"
                    f" | status={resp.status_code} | latency={latency:.3f}s | stream=True"
                )
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(_stream_generator(), media_type="text/event-stream")

    else:
        resp = await client.post(
            upstream_url,
            content=body,
            headers=forward_headers,
        )
        latency = time.monotonic() - start
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(
            f"[{ts}] POST /v1/chat/completions → {upstream_url}"
            f" | status={resp.status_code} | latency={latency:.3f}s | stream=False"
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
