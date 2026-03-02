# Backend - Deep Research Agent

FastAPI backend for chat + deep research orchestration.

## Stack

- FastAPI
- LangChain / LangGraph
- OpenAI models (chat + vision + embeddings)
- Qdrant (vector retrieval)
- Postgres (run snapshots + memory)
- `ddgs`, Tavily, Wikipedia search

## Key Files

- `main.py`: app routes, orchestration, tools, extraction, storage.
- `requirements.txt`: backend dependencies.
- `docker-compose.yml`: local Postgres and Qdrant services.

## Setup

1. Create environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure env (`.env`):

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

3. Start dependencies:

```bash
docker compose up -d
```

4. Run API:

```bash
uvicorn main:app --reload --port 8000
```

## Endpoints

### `GET /`
Health response.

### `POST /chat/stream`
SSE endpoint for token streaming chat.

Request:
```json
{
  "session_id": "optional-session",
  "message": "What time is it in UTC?"
}
```

### `POST /research`
Non-streaming research endpoint.

Request:
```json
{
  "session_id": "optional-session",
  "query": "Compare OpenAI deep research vs Grok deep research",
  "depth": "standard"
}
```

### `POST /research/stream`
Streaming research endpoint.

Request:
```json
{
  "session_id": "optional-session",
  "query": "Compare OpenAI deep research vs Grok deep research",
  "depth": "deep"
}
```

SSE event types emitted during research:
- `status` (`plan`, `queries`, `search`, `sources`, `extract`, `documents`, `retrieval`, `gaps`, `researchers`, `analyst`, `critic`, `verify`, `writer`, `transparency`, `done`)
- `delta` (streamed report tokens)
- `error`

## Research Pipeline

1. Planner creates search queries.
2. Parallel provider search.
3. Content extraction (HTML/PDF/image).
4. Chunk + embed + retrieve via Qdrant.
5. Multi-agent sequence:
   - Researchers
   - Analyst
   - Critic
   - Verifier
   - Writer
6. Save run snapshots + memory to Postgres.

## Tools

- `get_current_utc_time`
- `run_python` (restricted safe execution; no imports/network/file access)

## Data Stores

### Postgres
Tables:
- `research_runs`: compact state snapshots
- `research_memory`: reusable summary memory

### Qdrant
Collection pattern:
- `research_<session_id>`

## Notes on Limits and Safety

- Chunk count and chunk size are capped before Qdrant upsert.
- Upserts are batched to avoid payload-size failures.
- Serialized state is compacted and sanitized before Postgres JSONB write.

## Common Issues

- `ddgs not available`: install requirements in active venv.
- Wikipedia `403`: provider rate limits; pipeline continues.
- Port conflicts on `8000`: check `lsof -i :8000`.

## Minimal Smoke Test

```bash
curl -N -X POST http://127.0.0.1:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"n8n cloud vs self-hosted","depth":"shallow"}'
```
