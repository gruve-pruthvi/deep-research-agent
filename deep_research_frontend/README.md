# Frontend — Deep Research Agent UI

React + Vite single-page application for streaming chat and deep research workflows.

---

## Features

### Chat mode
- Streaming assistant responses via `/chat/stream`.
- Tool activity events displayed inside the assistant message (expandable).

### Research mode
- **Pre-research clarification dialog** — if the backend detects an ambiguous query, a modal shows 1–2 clarifying questions before research starts. User can answer or skip.
- **Depth selector** — `shallow`, `standard`, or `deep`.
- **Advanced options panel** — expandable toggle reveals an iterations slider (1–5).
- **Research plan preview card** — planned search queries are displayed before the search phase, so the user can see exactly what will be searched.
- **Animated research timeline** — 16 stages tracked in a side panel.
- **Source cards with credibility badges** — color-coded percentage badges per source (green ≥80%, amber ≥60%, red <60%).
- **Uncertainty gauge** — visual confidence bar shown after the verify stage.
- **Verifier notes** — collapsible section showing the verifier's textual assessment.
- **Evaluation score chips** — coverage / evidence / clarity scored /5.
- **Warning events** — non-fatal stage failures surfaced inline in the progress list.
- **Transparency panel** — queries, sources with credibility, verifier notes.
- **Report export** — Download `.md` and Copy to clipboard buttons appear after the report completes.
- **Research history** — History button opens a panel of past runs for the current session; click any run to restore the report.
- Markdown rendering via `react-markdown`.

---

## Tech Stack

- React 19 + TypeScript
- Vite 7
- CSS (custom serif theme — no external component library)

---

## Setup

```bash
npm install
npm run dev     # dev server at http://localhost:5173
npm run build   # production build (tsc + vite)
npm run lint    # ESLint
npm run preview # preview production build
```

---

## Backend Dependency

The UI talks to the backend at:

```
http://127.0.0.1:8000
```

If the backend runs elsewhere, update `API_URL` at the top of `src/App.tsx`.

---

## UX Flows

### Chat mode

```
User types message
  → POST /chat/stream
  → Streams delta events → renders in assistant bubble
  → Tool events shown in expandable "Tool activity" section
```

### Research mode

```
User types query
  → POST /research/clarify
      if questions returned → show clarification dialog
      user answers/skips
  → POST /research/stream (with query + depth + max_iterations)
      status/plan_preview  → shows "Research Plan" card with planned queries
      status/sources       → renders source cards with credibility badges
      status/verify        → shows uncertainty gauge + verifier notes
      delta                → streams tokens into assistant bubble
      status/transparency  → shows transparency panel with evaluation chips
      [DONE]               → shows export bar (Download .md + Copy to clipboard)
```

### History

```
Click "History" button in header
  → GET /research/history?session_id=...
  → Panel lists past runs
  → Click a run
      → GET /research/{run_id}
      → Restores report in message bubble
```

---

## SSE Event Contract (Consumed)

| Event | Payload fields | Rendered as |
|-------|---------------|-------------|
| `status` (any stage) | `stage`, `message`, `data` | Progress item in list; active stage highlighted in timeline |
| `status` (`plan_preview`) | `data.queries` | Plan preview card above source cards |
| `status` (`sources`) | `data.top_sources[].credibility` | Source cards with credibility badge |
| `status` (`verify`) | `data.uncertainty`, `data.verifier_notes` | Uncertainty gauge + collapsible notes |
| `status` (`transparency`) | `data.confidence`, `data.evaluation`, `data.verifier_notes` | Transparency panel, eval chips |
| `status` (`warning`) | `message` | Amber warning strip in progress list |
| `delta` | `delta` | Appended to assistant message content |
| `error` | `error` | Red error bar |
| `[DONE]` | — | Shows export bar; marks streaming complete |

---

## Component Overview

All logic lives in `src/App.tsx` (single-component architecture):

| State | Purpose |
|-------|---------|
| `messages` | Chat/research message history |
| `progressEvents` | All SSE status events for progress panel |
| `activeStage` | Currently active pipeline stage |
| `planPreview` | Planned queries from `plan_preview` event |
| `clarifyQuestions` / `clarifyAnswer` / `clarifyPending` | Clarification dialog state |
| `maxIterations` / `showAdvanced` | Advanced options |
| `reportText` / `reportDone` | Accumulated report for export |
| `historyRuns` / `historyOpen` | History panel state |

---

## File Overview

| File | Purpose |
|------|---------|
| `src/App.tsx` | All component logic, SSE parsing, UX flows |
| `src/App.css` | Theme, layout, all component styles |
| `src/index.css` | Global resets and font imports |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Blank page | Check browser console; ensure no runtime errors in `App.tsx` |
| 404 on API calls | Confirm backend is running on port 8000 and all endpoints exist |
| CORS errors | Set `ALLOWED_ORIGINS=http://localhost:5173` in backend `.env` |
| Clarification dialog never shows | Backend `ORCHESTRATOR_MODEL` must be accessible; check OPENAI_API_KEY |
| History panel empty | Backend must be running and Postgres accessible; at least one completed run needed |
| Export bar never appears | Verify `[DONE]` SSE event is being received in browser Network tab |
