# Frontend - Deep Research Agent UI

React + Vite UI for streaming chat and research workflows.

## Features

- Chat mode with streaming assistant responses.
- Research mode with:
  - Depth selector (`shallow`, `standard`, `deep`)
  - Animated research timeline
  - Live stage events
  - Source cards with favicons/domains
  - Transparency panel (confidence/evaluation/query/source trace)
- Markdown rendering via `react-markdown`.

## Tech Stack

- React 19
- TypeScript
- Vite
- CSS (custom theme)

## Setup

```bash
npm install
npm run dev
```

Open:
- `http://localhost:5173`

## Backend Dependency

UI expects backend at:
- `http://127.0.0.1:8000`

If backend runs elsewhere, update `API_URL` in `src/App.tsx`.

## Scripts

- `npm run dev` -> development server
- `npm run build` -> production build
- `npm run preview` -> preview build
- `npm run lint` -> linting

## UX Flows

### Chat Mode
- Send message -> `/chat/stream`
- Renders streamed `delta` events
- Displays tool events in assistant message details

### Research Mode
- Send query -> `/research/stream`
- Consumes `status` and `delta` SSE events
- Updates:
  - timeline active stage
  - progress details
  - source cards
  - transparency metadata

## SSE Event Contract (Consumed)

- `{"type":"status","stage":"...","message":"...","data":{...}}`
- `{"delta":"..."}`
- `{"error":"..."}`
- `[DONE]`

## File Overview

- `src/App.tsx`: mode logic, streaming parser, timeline/source/transparency rendering
- `src/App.css`: theme and layout
- `src/index.css`: global resets/fonts

## Troubleshooting

- Blank page after code changes:
  - Check browser console for runtime errors.
  - Ensure `App.tsx` has no variables declared outside component scope.
- 404 on API calls:
  - Ensure backend is running and endpoint paths exist.
  - Confirm frontend targets `127.0.0.1:8000` if local port conflicts exist.
- CORS issues:
  - Backend must allow `http://localhost:5173`.
