# LLM Gateway

> Semantic caching + smart routing proxy for OpenAI-compatible APIs.

## Why

Every team hitting LLM APIs burns money on duplicate prompts and over-powered models for trivial tasks. LLM Gateway sits transparently in front of any OpenAI-compatible endpoint: it embeds each incoming prompt, finds near-identical cached answers via cosine similarity, and routes low-complexity queries to a cheaper model — all without changing a single line of client code.

## Architecture

Target pipeline (steps 1–2 and 5 are live; steps 3–4 are roadmap):

```
┌──────────┐    POST /v1/chat/completions
│  Client  │──────────────────────────────►┐
└──────────┘                               │
                                   ┌───────▼────────┐
                                   │  LLM Gateway   │
                                   │  (FastAPI)     │
                                   │                │
                                   │ 1. Embed query │  ✅ live (M3)
                                   │ 2. Cache hit?  │──► return cached response
                                   │    (SQLite)    │  ✅ live (M3)
                                   │ 3. Score cmplx │  (M4)
                                   │ 4. Route model │  (M4)
                                   │ 5. Forward SSE │  ✅ live (M2)
                                   └───────┬────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │  OpenAI / Ollama / LiteLLM API   │
                          └──────────────────────────────────┘
```

Steps 1–2 (embed + cache lookup) are now live. On a cache hit the upstream is never called; on a miss the response is stored and future similar requests are served in <10 ms.

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

**M3 — Semantic cache**

- [x] `sentence-transformers/all-MiniLM-L6-v2` embeddings via singleton `embedder.py`
- [x] SQLite-backed cache (`cache.py`): `init_cache`, `get_cached`, `put_cached`
- [x] Cosine similarity lookup; configurable threshold (default `0.92`)
- [x] TTL eviction on startup (default 24 h)
- [x] Non-streaming path: cache hit returns stored response without upstream call
- [x] Streaming path: bypasses cache, passed through unchanged
- [x] Unit tests (`tests/test_cache.py`) + integration test for cache-hit behaviour

**Roadmap**

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

## Configuration

All settings are read from environment variables (copy `.env.example` to `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible API |
| `UPSTREAM_API_KEY` | — | API key forwarded to the upstream |
| `SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity cutoff for cache hits (0.0–1.0) |
| `CACHE_DB_PATH` | `cache.db` | Path to the SQLite cache file |
| `CACHE_TTL_SECONDS` | `86400` | Cache entry lifetime in seconds (default 24 h) |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

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
| M3 | Semantic cache | ✅ |
| M4 | Complexity routing | 🔲 |
| M5 | Rich dashboard | 🔲 |
| M6 | Docker Compose | 🔲 |

## License

[MIT](LICENSE) © 2024 Ritik
