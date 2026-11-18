# LLM Gateway

> Semantic caching + smart routing proxy for OpenAI-compatible APIs.

## Why

Every team hitting LLM APIs burns money on duplicate prompts and over-powered models for trivial tasks. LLM Gateway sits transparently in front of any OpenAI-compatible endpoint: it embeds each incoming prompt, finds near-identical cached answers via cosine similarity, and routes low-complexity queries to a cheaper model — all without changing a single line of client code.

## Architecture

```
┌──────────┐    POST /v1/chat/completions
│  Client  │──────────────────────────────►┐
└──────────┘                               │
                                   ┌───────▼────────┐
                                   │  LLM Gateway   │
                                   │  (FastAPI)     │
                                   │                │
                                   │ 1. Embed query │
                                   │ 2. Cache hit?  │──► return cached response
                                   │    (SQLite)    │
                                   │ 3. Score cmplx │
                                   │ 4. Route model │
                                   │ 5. Forward SSE │
                                   └───────┬────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │  OpenAI / Ollama / LiteLLM API   │
                          └──────────────────────────────────┘
```

## Features

Checked items are implemented and working; unchecked items are on the roadmap.

- [x] Python package scaffold (`src/llm_gateway/`) with pinned dependencies (`pyproject.toml`)
- [x] FastAPI app skeleton with `/healthz` liveness endpoint
- [x] `.env.example` documenting all configuration keys
- [x] Dev toolchain wired: pytest, ruff, mypy, hatchling build
- [ ] Transparent `/v1/chat/completions` proxy
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

# 4. Verify
curl http://localhost:8000/healthz
# {"status":"ok"}
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
| M2 | Proxy core | 🔲 |
| M3 | Semantic cache | 🔲 |
| M4 | Complexity routing | 🔲 |
| M5 | Rich dashboard | 🔲 |
| M6 | Docker Compose | 🔲 |

## License

[MIT](LICENSE) © 2024 Ritik
