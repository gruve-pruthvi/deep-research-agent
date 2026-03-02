# Deep Research Agent

Deep Research Agent is a full-stack AI-powered research platform that streams a structured, cited report from a single natural-language query. It is competitive with leading deep-research products (ChatGPT Deep Research, Claude Research, Gemini Deep Research) across planning quality, source credibility, iterative retrieval, and UX transparency.

It includes:
- A **FastAPI + LangGraph backend** for multi-stage research orchestration with tiered LLM models, semantic memory, and graceful error recovery.
- A **React + Vite frontend** with live streaming, pre-research clarification, plan preview, credibility-scored source cards, uncertainty gauge, and report export.

---

## Repository Structure

```
deep_research_backend/    API, orchestration, tools, storage
deep_research_frontend/   Chat + research UI
CLAUDE.md                 Project instructions for Claude Code
IMPLEMENTATION_STATUS.md  Feature checklist and known limitations
```

---

## Current Capabilities

### Research Pipeline
- **Pre-research clarification** — LLM evaluates the query and asks 1–2 targeted questions before research begins if intent is ambiguous; skips automatically for clear queries.
- **Research plan preview** — planned search queries are surfaced to the user before the search phase starts.
- **Iterative search loop** — 1–5 iterations (configurable); gap analysis controls continuation.
- **Multi-provider search** — Tavily (primary) + DuckDuckGo + Wikipedia in parallel per query.
- **Hybrid credibility scoring** — 60% heuristic (TLD) + 40% LLM; scores shown on every source card.
- **Content extraction** — HTML, PDF (PyMuPDF), images (GPT-4o vision); optional Playwright fallback for JS-heavy pages.
- **RAG** — chunk → embed (text-embedding-3-small) → Qdrant → semantic retrieval.
- **Tiered LLM models** — orchestrator model (default `gpt-4o`) for planning, analysis, critic, verifier, and writer; worker model (default `gpt-4o-mini`) for bulk summarization.
- **Semantic memory** — past research is embedded and stored in Qdrant; future queries retrieve related summaries semantically rather than by substring match.
- **Multi-agent analysis** — parallel Researchers → Analyst → Critic → Verifier → Writer.
- **Citation verification** — post-generation pass corrects mismatched `[N]` citation numbers.
- **Graceful error recovery** — each pipeline stage has its own try/except; failures emit `warning` events and the pipeline continues in degraded state.

### Frontend UX
- Credibility badges (color-coded %) on source cards.
- Uncertainty gauge (confidence bar) after the verify stage.
- Verifier notes in collapsible sections.
- Evaluation scores (coverage / evidence / clarity) as labeled chips.
- Report export: Download `.md` and Copy to clipboard.
- Advanced options: iterations slider (1–5).
- Clarification dialog for ambiguous queries.
- Research history panel — lists past runs by session; click to restore a report.

---

## Architecture Overview

```
User query
  │
  ▼
POST /research/clarify      ← optional: returns clarifying questions
  │
  ▼
POST /research/stream       ← main pipeline
  │
  ├─ plan          LLM generates 3–5 focused queries (orchestrator model)
  ├─ plan_preview  Emits planned queries as SSE before search starts
  ├─ search        Parallel: Tavily + DuckDuckGo + Wikipedia
  ├─ sources       Hybrid credibility scoring; top-5 emitted with scores
  ├─ extract       HTML / PDF / image / Playwright (optional)
  ├─ documents     RecursiveCharacterTextSplitter (800 chars, 120 overlap)
  ├─ retrieval     Embed → Qdrant → top-k semantic retrieval
  ├─ gaps          LLM decides whether to iterate again
  ├─ researchers   3 parallel workers summarise document buckets (worker model)
  ├─ analyst       Synthesises notes, runs Python sandbox (orchestrator model)
  ├─ critic        Reviews evidence quality (orchestrator model)
  ├─ verify        Assigns uncertainty 0–1; emits score + notes (orchestrator)
  ├─ writer        Streams final report; post-pass citation verification
  ├─ transparency  Emits confidence, evaluation, source trace
  └─ done          Saves to Postgres; saves semantic memory to Qdrant
```

### Depth Configuration

| Depth    | max_results | max_docs | top_k | Approx. words |
|----------|-------------|----------|-------|---------------|
| shallow  | 3           | 6        | 6     | 400–700       |
| standard | 5           | 10       | 8     | 700–1200      |
| deep     | 7           | 14       | 10    | 1200–1800     |

### Data Stores

| Store    | Purpose |
|----------|---------|
| Qdrant `research_<session_id>` | Per-session vector index (deleted and recreated each run) |
| Qdrant `research_memory_embeddings` | Persistent semantic memory across sessions |
| Postgres `research_runs` | Full run snapshots (compact JSONB) |
| Postgres `research_memory` | Query + report summaries (also indexed in Qdrant) |

---

## Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (Postgres + Qdrant via `docker compose`)
- API keys: `OPENAI_API_KEY`, `TAVILY_API_KEY`

---

## Quick Start

**1. Start infrastructure** (from `deep_research_backend/`):

```bash
docker compose up -d
```

**2. Start backend** (from `deep_research_backend/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**3. Start frontend** (from `deep_research_frontend/`):

```bash
npm install
npm run dev
```

**4. Open UI:** `http://localhost:5173`

---

## Environment Variables

Backend `deep_research_backend/.env`:

```bash
# Required
OPENAI_API_KEY=...
TAVILY_API_KEY=...

# Infrastructure
QDRANT_URL=http://localhost:6333
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# Model tiers
ORCHESTRATOR_MODEL=gpt-4o          # used for plan, analyst, critic, verifier, writer
WORKER_MODEL=gpt-4o-mini           # used for researchers, scoring, gap analysis

# Optional features
ENABLE_PLAYWRIGHT=false            # set true to enable JS-rendered page extraction
                                   # requires: playwright install chromium
ALLOWED_ORIGINS=http://localhost:5173   # comma-separated for multiple origins
LOG_LEVEL=INFO
```

> **Playwright setup** (if enabled):
> ```bash
> pip install playwright
> playwright install chromium
> ```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check |
| `POST` | `/chat/stream` | Conversational SSE stream |
| `POST` | `/research` | Non-streaming research |
| `POST` | `/research/stream` | Streaming research (main endpoint) |
| `POST` | `/research/clarify` | Pre-research clarification questions |
| `GET`  | `/research/history` | List past runs (`?session_id=` optional) |
| `GET`  | `/research/{run_id}` | Retrieve a specific run + report |

### `/research/stream` request body

```json
{
  "session_id": "optional",
  "query": "Your research topic",
  "depth": "standard",
  "max_iterations": 3,
  "approved_queries": ["optional override", "for the plan phase"]
}
```

### SSE event types

| Type | Description |
|------|-------------|
| `status` (stage: `plan`) | Iteration planning started |
| `status` (stage: `plan_preview`) | Planned queries ready; data includes `queries` list |
| `status` (stage: `search`) | Search in progress |
| `status` (stage: `sources`) | Sources found; data includes `top_sources` with `credibility` per source |
| `status` (stage: `extract`) | Document extraction in progress |
| `status` (stage: `documents`) | Extraction complete; data includes `documents` count |
| `status` (stage: `retrieval`) | Qdrant retrieval complete; data includes `chunks` count |
| `status` (stage: `gaps`) | Gap assessment; data includes `continue` bool + `gaps` list |
| `status` (stage: `researchers`) | Parallel summarisation running |
| `status` (stage: `analyst`) | Analyst synthesis running |
| `status` (stage: `critic`) | Critic review running |
| `status` (stage: `verify`) | Verification complete; data includes `uncertainty` (0–1) + `verifier_notes` |
| `status` (stage: `writer`) | Writing in progress |
| `status` (stage: `transparency`) | Final metadata; data includes `confidence`, `uncertainty`, `verifier_notes`, `evaluation`, `transparency` |
| `status` (stage: `done`) | Pipeline complete |
| `status` (stage: `warning`) | Stage failed gracefully; pipeline continued |
| `delta` | Streamed report token |
| `error` | Fatal error |
| `[DONE]` | Stream end sentinel |

---

## Verification Checklist

- `POST /research/stream` with a complex query — SSE timeline shows all stages through `done`.
- `POST /research/clarify` with an ambiguous query returns `{questions: [...], proceed: false}`.
- Source cards show colored credibility badges (green ≥80%, amber ≥60%, red <60%).
- Plan preview card renders before the search phase begins.
- After report completes, Download `.md` and Copy to clipboard buttons appear.
- Uncertainty gauge shows correct confidence level; verifier notes are collapsible.
- Evaluation scores render as three labeled chips.
- History panel shows past runs; clicking one restores the report.
- `GET /research/history?session_id=...` returns JSON with `runs` array.
- Killing Qdrant mid-run produces a `warning` event but report still completes.
- Memory search finds semantically related past research even with different phrasing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'ddgs'` | `pip install -r requirements.txt` in active venv |
| `No module named 'playwright'` | `pip install playwright && playwright install chromium` |
| Qdrant payload too large | Chunk limits and batched upserts (100/batch) already handle this |
| Postgres JSON errors | Compact state serialization is in place; ensure latest code is running |
| Wikipedia 403 | Non-fatal; pipeline continues with other providers |
| Port 8000 conflict | `lsof -i :8000` to identify and stop conflicting process |
| CORS errors | Set `ALLOWED_ORIGINS=http://your-frontend-url` in `.env` |
| Empty Playwright extraction | Ensure `playwright install chromium` was run after `pip install playwright` |

---

## Additional Docs

- Backend details: `deep_research_backend/README.md`
- Frontend details: `deep_research_frontend/README.md`
- Feature status: `IMPLEMENTATION_STATUS.md`
- Project instructions: `CLAUDE.md`
