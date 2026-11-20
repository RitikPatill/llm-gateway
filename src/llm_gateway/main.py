import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_gateway.cache import get_cached, init_cache, put_cached
from llm_gateway.config import get_settings, load_routes_config
from llm_gateway.embedder import embed
from llm_gateway.metrics import metrics
from llm_gateway.router import resolve_model, score_complexity

load_dotenv()

logger = logging.getLogger(__name__)

_HOP_BY_HOP = {"transfer-encoding", "connection", "keep-alive", "te", "trailers", "upgrade", "host"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Initialise cache (create table, evict stale rows)
    init_cache(settings.cache_db_path, settings.cache_ttl_seconds)
    # Warm up embedder so the ~2 s model load happens before the first request
    embed("")
    # Load routing config (empty dict → routing disabled gracefully)
    app.state.routes_config = load_routes_config(settings.routes_config_path)
    app.state.http_client = httpx.AsyncClient(timeout=120.0)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics() -> dict:
    return metrics.snapshot()


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
        # Apply routing before forwarding to upstream
        if request.app.state.routes_config:
            messages = json_body.get("messages", [])
            score = score_complexity(messages)
            resolved = resolve_model(messages, request.app.state.routes_config)
            logger.info("router: score=%d model=%s", score, resolved)
            json_body["model"] = resolved
            body = json.dumps(json_body).encode()

        _stream_model = json_body.get("model", "unknown")

        async def _stream_generator():
            try:
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
            finally:
                latency_ms = (time.monotonic() - start) * 1000
                metrics.record_request(
                    latency_ms=latency_ms,
                    cache_hit=False,
                    model=_stream_model,
                )

        return StreamingResponse(_stream_generator(), media_type="text/event-stream")

    else:
        # Build cache key from user message contents only (system prompts excluded
        # so a cached answer survives system-prompt tweaks without a cache miss).
        cache_key = " ".join(
            m.get("content", "")
            for m in json_body.get("messages", [])
            if m.get("role") != "system"
        )
        emb = embed(cache_key)

        cached = get_cached(settings.cache_db_path, emb, settings.similarity_threshold)
        if cached is not None:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            latency = time.monotonic() - start
            latency_ms = latency * 1000
            print(
                f"[{ts}] POST /v1/chat/completions → CACHE HIT"
                f" | latency={latency:.3f}s | stream=False"
            )
            tokens_saved = len(str(cached)) // 4
            metrics.record_request(
                latency_ms=latency_ms,
                cache_hit=True,
                model=json_body.get("model", "unknown"),
                tokens_saved=tokens_saved,
            )
            return JSONResponse(content=cached, status_code=200)

        # Cache miss — apply routing before forwarding to upstream
        if request.app.state.routes_config:
            messages = json_body.get("messages", [])
            score = score_complexity(messages)
            resolved = resolve_model(messages, request.app.state.routes_config)
            logger.info("router: score=%d model=%s", score, resolved)
            json_body["model"] = resolved
            body = json.dumps(json_body).encode()

        resp = await client.post(
            upstream_url,
            content=body,
            headers=forward_headers,
        )
        latency = time.monotonic() - start
        latency_ms = latency * 1000
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(
            f"[{ts}] POST /v1/chat/completions → {upstream_url}"
            f" | status={resp.status_code} | latency={latency:.3f}s | stream=False"
        )

        resp_json = resp.json()
        if resp.status_code == 200:
            put_cached(settings.cache_db_path, emb, resp_json)

        metrics.record_request(
            latency_ms=latency_ms,
            cache_hit=False,
            model=json_body.get("model", "unknown"),
        )
        return JSONResponse(content=resp_json, status_code=resp.status_code)
