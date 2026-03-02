# Deep Research Agent

Deep Research Agent is a full-stack project for long-running, multi-step research with streaming output.

It includes:
- A FastAPI + LangGraph backend for planning, searching, extraction, retrieval, analysis, critique, verification, and report writing.
- A React + Vite frontend with live streaming, progress timeline, source cards, and transparency details.

## Repository Structure

- `deep_research_backend/`: API, orchestration, tools, storage integration.
- `deep_research_frontend/`: chat/research UI.
- `DeepReearchAgent.txt`: deep research concept notes.
- `Implementation_Plan.txt`: phase roadmap (Phase 1 -> Phase 5).

## Current Capabilities

- Streaming chat (`/chat/stream`) with tool activity events.
- Streaming research (`/research/stream`) with stage-by-stage progress events.
- Multi-provider search:
  - Tavily
  - DuckDuckGo (`ddgs`)
  - Wikipedia API (best effort; can be rate-limited)
- Content extraction:
  - HTML text extraction
  - PDF extraction (PyMuPDF)
  - Image description via vision model
- RAG:
  - Embeddings + Qdrant index/query
- Reliability:
  - Hybrid source credibility
  - Critic + verifier steps
  - Confidence and evaluation metadata
- Persistence:
  - Postgres tables for run snapshots + reusable research memory

## Architecture (High Level)

1. Planner generates focused search queries.
2. Search providers run and aggregate results.
3. Extractor fetches HTML/PDF/image content.
4. Chunking + embedding + retrieval in Qdrant.
5. Multi-agent analysis flow:
   - Researchers (parallel notes)
   - Analyst (synthesis)
   - Critic (weaknesses)
   - Verifier (uncertainty)
   - Writer (final report, streamed)
6. Transparency payload emitted with queries, top sources, confidence, and evaluation.

## Prerequisites

- Python 3.11+ recommended (3.14 works but some dependency deprecation warnings may appear).
- Node.js 18+.
- Docker (for Postgres and Qdrant).
- API keys:
  - `OPENAI_API_KEY`
  - `TAVILY_API_KEY`

## Quick Start

1. Start infrastructure (from `deep_research_backend/`):

```bash
docker compose up -d
```

2. Start backend (from `deep_research_backend/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

3. Start frontend (from `deep_research_frontend/`):

```bash
npm install
npm run dev
```

4. Open UI:
- `http://localhost:5173`

## Environment Variables

Backend (`deep_research_backend/.env`):

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
QDRANT_URL=http://localhost:6333
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres
LOG_LEVEL=INFO
```

## API Endpoints

- `GET /` -> health check.
- `POST /chat/stream` -> SSE token stream for conversational mode.
- `POST /research` -> non-streaming research response.
- `POST /research/stream` -> SSE research progress + report stream.

## Development Notes

- If `localhost:8000` collisions happen, ensure no other process binds to 8000.
- Frontend currently targets `http://127.0.0.1:8000` to avoid IPv4/IPv6 ambiguity.
- Wikipedia may return 403 rate-limit responses. The pipeline continues with other providers.

## Verification Checklist

- Research mode shows timeline stages through `done`.
- Source cards render with domain favicons.
- Transparency panel includes confidence and evaluation.
- Postgres tables are populated:
  - `research_runs`
  - `research_memory`

## Troubleshooting

- `No module named 'ddgs'`:
  - Reinstall backend deps: `pip install -r requirements.txt`
- Qdrant payload too large:
  - Chunk limits and batched upserts are already implemented; check for custom modifications.
- Postgres JSON serialization errors:
  - Compacted state serialization is implemented; ensure latest backend code is running.

## Additional Docs

- Backend details: `deep_research_backend/README.md`
- Frontend details: `deep_research_frontend/README.md`
