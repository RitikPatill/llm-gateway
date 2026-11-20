# LLM Gateway — Semantic Cache & Smart Router

> A local OpenAI-compatible proxy that semantically caches responses and routes prompts to cheaper models when safe to do so.

![LLM Gateway demo](demo/demo.gif)

## What it is

LLM Gateway is a lightweight HTTP proxy that sits between your application and any OpenAI-compatible API (OpenAI, Ollama, LiteLLM). It exposes the standard `/v1/chat/completions` endpoint, so any existing client works without modification — just point `OPENAI_BASE_URL` at it.

On every request it embeds the user prompt using a local sentence-transformer model, computes cosine similarity against a SQLite cache, and returns the stored answer if the score clears a configurable threshold. On a cache miss it scores prompt complexity via a token-count + keyword heuristic and routes the request to the cheapest model tier that can handle it. A live Rich terminal dashboard shows hit rate, latency percentiles, and estimated token savings in real time.

## Quickstart

```bash
git clone https://github.com/RitikPatill/llm-gateway.git
cd llm-gateway

# Install (Python 3.10+)
pip install -e ".[dev]"

# Configure upstream
cp .env.example .env
# Edit .env — set UPSTREAM_BASE_URL and UPSTREAM_API_KEY

# Run the gateway
uvicorn llm_gateway.main:app --reload

# Verify it is live
curl http://localhost:8000/healthz
# {"status":"ok"}

# Open the terminal dashboard in a second terminal
python -m llm_gateway.dashboard
```

```bash
# OR: run the full stack with Docker Compose (gateway + Ollama, no Python install required)
docker compose up --build

# Pull a model on first run
docker compose exec ollama ollama pull llama3.2
```

## Usage

Point any OpenAI client at the gateway by setting the base URL:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=unused   # forwarded to upstream; set a real key in .env
```

Send a request exactly as you would to the OpenAI API:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": false
  }'
```

The first call hits the upstream and stores the response. A second call with a semantically equivalent prompt (e.g. "Capital city of France?") returns the cached answer in under 5 ms without touching the upstream. The model field in the request is transparently rewritten to the tier resolved by the complexity scorer; the caller sees the upstream's response unchanged.

Routing tiers and thresholds are edited in `config/routes.yaml`. Swap OpenAI model IDs for Ollama IDs (e.g. `llama3.2`, `mistral`) to run fully locally at zero cost.

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
                                   │ 6. /metrics    │
                                   └───────┬────────┘
                                           │
                          ┌────────────────▼─────────────────┐
                          │  OpenAI / Ollama / LiteLLM API   │
                          └──────────────────────────────────┘
```

On a cache hit the upstream is never called. On a miss the complexity scorer picks a tier (`fast` / `balanced` / `powerful`) and injects the resolved model ID into the forwarded request.

## Project structure

```
llm-gateway/
├── src/llm_gateway/   # FastAPI app: main, proxy, cache, embedder, router, metrics, dashboard
├── config/            # routes.yaml — tier definitions and complexity thresholds
├── demo/              # terminal GIF (demo.gif) and VHS tape script (demo.tape)
├── tests/             # pytest unit and integration tests
├── Dockerfile         # gateway container image
├── docker-compose.yml # gateway + Ollama stack with health checks and persistent volume
├── pyproject.toml     # package metadata and pinned dependencies
└── .env.example       # all configuration keys with descriptions
```

## Roadmap

- [ ] Replace the cosine-only cache with an approximate-nearest-neighbour index (e.g. `hnswlib`) for sub-millisecond lookup at scale
- [ ] ML-based complexity classifier trained on real prompt/model-tier pairs to replace the keyword heuristic
- [ ] Per-tenant API keys and usage quotas for multi-user deployments
- [ ] Persist metrics to SQLite and expose a Prometheus scrape endpoint
- [ ] Web UI for the dashboard (replace the Rich TUI with a minimal browser-based view)

## License

MIT — see LICENSE.

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
