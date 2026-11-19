# LLM Gateway

> Semantic caching + smart routing proxy for OpenAI-compatible APIs.

## Why

Every team hitting LLM APIs burns money on duplicate prompts and over-powered models for trivial tasks. LLM Gateway sits transparently in front of any OpenAI-compatible endpoint: it embeds each incoming prompt, finds near-identical cached answers via cosine similarity, and routes low-complexity queries to a cheaper model — all without changing a single line of client code.

## Architecture

Target pipeline (steps 1–4 are roadmap items; step 5 is live):

```
┌──────────┐    POST /v1/chat/completions
│  Client  │──────────────────────────────►┐
└──────────┘                               │
                                   ┌───────▼────────┐
                                   │  LLM Gateway   │
                                   │  (FastAPI)     │
                                   │                │
                                   │ 1. Embed query │  (M3)
                                   │ 2. Cache hit?  │──► return cached response
                                   │    (SQLite)    │
                                   │ 3. Score cmplx │  (M4)
                                   │ 4. Route model │  (M4)
                                   │ 5. Forward SSE │  ✅ live
                                   └───────┬────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │  OpenAI / Ollama / LiteLLM API   │
                          └──────────────────────────────────┘
```

Currently the gateway forwards every request directly to the upstream (step 5). Caching and routing will be layered in front without changing the client-facing API.

## Features

Checked items are implemented and working; unchecked items are on the roadmap.

**M1 — Scaffold**

- [x] Python package scaffold (`src/llm_gateway/`) with pinned dependencies (`pyproject.toml`)
- [x] FastAPI app skeleton with `/healthz` liveness endpoint
- [x] `.env.example` documenting all configuration keys
- [x] Dev toolchain wired: pytest, ruff, mypy, hatchling build

**M2 — Proxy core**

- [x] `config.py` settings module: `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, `LOG_LEVEL` read from environment
- [x] `POST /v1/chat/completions` forwarded to configurable upstream via `httpx`
- [x] Streaming (SSE) and non-streaming response modes
- [x] Hop-by-hop header stripping; `Authorization` injected from `UPSTREAM_API_KEY`
- [x] Shared `httpx.AsyncClient` via FastAPI lifespan (connection pooling, 120 s timeout)
- [x] Upstream HTTP status codes and error bodies forwarded verbatim
- [x] Structured stdout log line per request: timestamp, upstream URL, status, latency, stream flag
- [x] Integration tests covering non-streaming, streaming, missing API key, and upstream error passthrough

**Roadmap**

- [ ] Semantic cache (cosine-sim, SQLite)
- [ ] Complexity-based model routing
- [ ] Rich terminal dashboard
- [ ] Docker Compose (gateway + Ollama)

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourhandle/llm-gateway.git
cd llm-gateway
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env — set UPSTREAM_BASE_URL and UPSTREAM_API_KEY

# 3. Run
uvicorn llm_gateway.main:app --reload

# 4. Verify liveness
curl http://localhost:8000/healthz
# {"status":"ok"}

# 5. Smoke-test the proxy (requires a valid UPSTREAM_API_KEY in .env)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"stream":false}'
```

Point any OpenAI client at `http://localhost:8000` by setting:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
```

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check src/ tests/

# Type-check
mypy src/
```

## Milestone status

| # | Name | Status |
|---|------|--------|
| M1 | Scaffold + README | ✅ |
| M2 | Proxy core | ✅ |
| M3 | Semantic cache | 🔲 |
| M4 | Complexity routing | 🔲 |
| M5 | Rich dashboard | 🔲 |
| M6 | Docker Compose | 🔲 |

## License

[MIT](LICENSE) © 2024 Ritik
