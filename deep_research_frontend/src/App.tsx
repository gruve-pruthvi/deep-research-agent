import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import './App.css'

type Role = 'user' | 'assistant'

type ChatMessage = {
  id: string
  role: Role
  content: string
  toolEvents?: ToolEvent[]
}

type ToolEvent = {
  id: string
  name: string
  kind: 'start' | 'end'
  payload: string
}

type ProgressEvent = {
  id: string
  stage: string
  message: string
  data?: Record<string, unknown> | null
}

type HistoryRun = {
  id: string
  session_id: string
  query: string
  depth: string
  status: string
  updated_at: string | null
}

const API_URL = 'http://127.0.0.1:8000'

type Mode = 'chat' | 'research'
type Depth = 'shallow' | 'standard' | 'deep'

function credibilityColor(score: number): string {
  if (score >= 0.8) return '#2e7d32'
  if (score >= 0.6) return '#f57f17'
  return '#c62828'
}

function UncertaintyGauge({ value }: { value: number }) {
  const confidence = Math.round((1 - value) * 100)
  const color = value <= 0.3 ? '#2e7d32' : value <= 0.6 ? '#f57f17' : '#c62828'
  return (
    <div className="uncertainty-gauge">
      <div className="uncertainty-gauge__bar">
        <div
          className="uncertainty-gauge__fill"
          style={{ width: `${confidence}%`, background: color }}
        />
      </div>
      <span className="uncertainty-gauge__label" style={{ color }}>
        {confidence}% confidence
      </span>
    </div>
  )
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('chat')
  const [depth, setDepth] = useState<Depth>('standard')
  const [progressEvents, setProgressEvents] = useState<ProgressEvent[]>([])
  const [activeStage, setActiveStage] = useState<string | null>(null)
  const [progressOpen, setProgressOpen] = useState(true)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [maxIterations, setMaxIterations] = useState(3)
  const [reportText, setReportText] = useState('')
  const [reportDone, setReportDone] = useState(false)
  // Clarification
  const [clarifyQuestions, setClarifyQuestions] = useState<string[]>([])
  const [clarifyAnswer, setClarifyAnswer] = useState('')
  const [clarifyPending, setClarifyPending] = useState(false)
  const pendingQueryRef = useRef<string>('')
  // Plan preview
  const [planPreview, setPlanPreview] = useState<string[] | null>(null)
  // History
  const [historyOpen, setHistoryOpen] = useState(false)
  const [historyRuns, setHistoryRuns] = useState<HistoryRun[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  const scrollRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef(crypto.randomUUID())

  const stages = [
    'plan',
    'queries',
    'plan_preview',
    'search',
    'sources',
    'extract',
    'documents',
    'retrieval',
    'gaps',
    'researchers',
    'analyst',
    'critic',
    'verify',
    'writer',
    'transparency',
    'done',
  ]

  const latestSources = (() => {
    const sourceEvent = [...progressEvents].reverse().find((event) => event.data?.top_sources)
    if (!sourceEvent) return []
    return sourceEvent.data?.top_sources as Array<{
      title: string
      url: string
      provider?: string
      credibility?: number
    }>
  })()

  const lastSearchEvent = [...progressEvents].reverse().find((event) => event.stage === 'search')
  const transparencyEvent = [...progressEvents]
    .reverse()
    .find((event) => event.stage === 'transparency')
  const verifyEvent = [...progressEvents]
    .reverse()
    .find((event) => event.stage === 'verify' && event.data?.uncertainty !== undefined)

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      })
    })
  }

  const loadHistory = async () => {
    setHistoryLoading(true)
    try {
      const res = await fetch(
        `${API_URL}/research/history?session_id=${sessionIdRef.current}`
      )
      if (res.ok) {
        const data = await res.json()
        setHistoryRuns(data.runs || [])
      }
    } catch {
      // non-fatal
    } finally {
      setHistoryLoading(false)
    }
  }

  const restoreRun = async (runId: string) => {
    try {
      const res = await fetch(`${API_URL}/research/${runId}`)
      if (!res.ok) return
      const data = await res.json()
      if (!data.report) return
      const restoredMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.report,
      }
      const queryMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: data.query,
      }
      setMessages([queryMsg, restoredMsg])
      setReportText(data.report)
      setReportDone(true)
      setHistoryOpen(false)
    } catch {
      // non-fatal
    }
  }

  const downloadReport = () => {
    if (!reportText) return
    const blob = new Blob([reportText], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'research-report.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  const copyReport = () => {
    if (!reportText) return
    navigator.clipboard.writeText(reportText)
  }

  // Start research after clarification is resolved
  const startResearch = async (query: string, clarificationContext: string) => {
    const finalQuery = clarificationContext
      ? `${query}\n\nClarification: ${clarificationContext}`
      : query

    const nextUserMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
    }
    const nextAssistantMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      content: '',
    }

    setMessages((prev) => [...prev, nextUserMessage, nextAssistantMessage])
    setInput('')
    setError(null)
    setProgressEvents([])
    setPlanPreview(null)
    setReportText('')
    setReportDone(false)
    setIsStreaming(true)
    scrollToBottom()

    try {
      const controller = new AbortController()
      abortRef.current = controller
      const payload = {
        session_id: sessionIdRef.current,
        query: finalQuery,
        depth,
        max_iterations: maxIterations,
      }
      const response = await fetch(`${API_URL}/research/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status})`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let done = false
      const reportAccum: string[] = []

      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          const lines = part.split('\n')
          for (const line of lines) {
            if (!line.startsWith('data:')) continue
            const data = line.replace('data:', '').trim()
            if (!data) continue
            if (data === '[DONE]') {
              done = true
              break
            }
            try {
              const parsed = JSON.parse(data) as {
                delta?: string
                error?: string
                type?: string
                stage?: string
                message?: string
                data?: Record<string, unknown>
              }
              if (parsed.error) {
                setError(parsed.error)
                done = true
                break
              }
              if (parsed.type === 'status') {
                const event: ProgressEvent = {
                  id: crypto.randomUUID(),
                  stage: parsed.stage || 'update',
                  message: parsed.message || '',
                  data: (parsed.data as Record<string, unknown>) ?? null,
                }
                setProgressEvents((prev) => [...prev, event])
                setActiveStage(event.stage)
                // Show plan preview
                if (event.stage === 'plan_preview' && Array.isArray(event.data?.queries)) {
                  setPlanPreview(event.data.queries as string[])
                }
                continue
              }
              if (parsed.delta) {
                reportAccum.push(parsed.delta)
                setMessages((prev) => {
                  const updated = [...prev]
                  const lastIndex = updated.findIndex(
                    (msg) => msg.id === nextAssistantMessage.id
                  )
                  if (lastIndex === -1) return prev
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    content: updated[lastIndex].content + parsed.delta,
                  }
                  return updated
                })
                scrollToBottom()
              }
            } catch (err) {
              console.error(err)
            }
          }
        }
      }

      const fullReport = reportAccum.join('')
      setReportText(fullReport)
      setReportDone(true)
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setError(null)
      } else {
        console.error(err)
        setError('Unable to connect to the backend.')
      }
    } finally {
      setIsStreaming(false)
      scrollToBottom()
    }
  }

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return

    if (mode === 'chat') {
      // Chat mode — direct SSE stream
      const nextUserMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: input.trim(),
      }
      const nextAssistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
      }

      const nextMessages = [...messages, nextUserMessage, nextAssistantMessage]
      setMessages(nextMessages)
      setInput('')
      setError(null)
      setIsStreaming(true)
      scrollToBottom()

      try {
        const controller = new AbortController()
        abortRef.current = controller
        const payload = {
          session_id: sessionIdRef.current,
          message: nextUserMessage.content,
        }
        const response = await fetch(`${API_URL}/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        if (!response.ok || !response.body) {
          throw new Error(`Request failed (${response.status})`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let done = false

        while (!done) {
          const { value, done: readerDone } = await reader.read()
          done = readerDone
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done })

          const parts = buffer.split('\n\n')
          buffer = parts.pop() || ''

          for (const part of parts) {
            const lines = part.split('\n')
            for (const line of lines) {
              if (!line.startsWith('data:')) continue
              const data = line.replace('data:', '').trim()
              if (!data) continue
              if (data === '[DONE]') {
                done = true
                break
              }
              try {
                const parsed = JSON.parse(data) as {
                  delta?: string
                  error?: string
                  type?: string
                  name?: string
                  input?: unknown
                  output?: unknown
                }
                if (parsed.error) {
                  setError(parsed.error)
                  done = true
                  break
                }
                if (parsed.type === 'tool_start' || parsed.type === 'tool_end') {
                  const toolEvent: ToolEvent = {
                    id: crypto.randomUUID(),
                    name: parsed.name || 'tool',
                    kind: parsed.type === 'tool_start' ? 'start' : 'end',
                    payload:
                      parsed.type === 'tool_start'
                        ? parsed.input
                          ? JSON.stringify(parsed.input, null, 2)
                          : '…'
                        : parsed.output
                        ? JSON.stringify(parsed.output, null, 2)
                        : 'done',
                  }
                  setMessages((prev) => {
                    const updated = [...prev]
                    const lastAssistantIndex = [...updated]
                      .reverse()
                      .findIndex((msg) => msg.role === 'assistant')
                    if (lastAssistantIndex === -1) return prev
                    const index = updated.length - 1 - lastAssistantIndex
                    const current = updated[index]
                    updated[index] = {
                      ...current,
                      toolEvents: [...(current.toolEvents ?? []), toolEvent],
                    }
                    return updated
                  })
                  scrollToBottom()
                  continue
                }
                if (parsed.delta) {
                  setMessages((prev) => {
                    const updated = [...prev]
                    const lastIndex = updated.findIndex(
                      (msg) => msg.id === nextAssistantMessage.id
                    )
                    if (lastIndex === -1) return prev
                    updated[lastIndex] = {
                      ...updated[lastIndex],
                      content: updated[lastIndex].content + parsed.delta,
                    }
                    return updated
                  })
                  scrollToBottom()
                }
              } catch (err) {
                console.error(err)
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          setError(null)
        } else {
          setError('Unable to connect to the backend.')
        }
      } finally {
        setIsStreaming(false)
        scrollToBottom()
      }
      return
    }

    // Research mode — check for clarification first
    const query = input.trim()
    pendingQueryRef.current = query

    try {
      const res = await fetch(`${API_URL}/research/clarify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      if (res.ok) {
        const data = await res.json() as { questions?: string[]; proceed?: boolean }
        if (!data.proceed && data.questions && data.questions.length > 0) {
          setClarifyQuestions(data.questions)
          setClarifyAnswer('')
          setClarifyPending(true)
          return
        }
      }
    } catch {
      // If clarify endpoint fails, proceed anyway
    }

    await startResearch(query, '')
  }

  const handleClarifySubmit = async () => {
    const query = pendingQueryRef.current
    const answer = clarifyAnswer.trim()
    setClarifyPending(false)
    setClarifyQuestions([])
    setClarifyAnswer('')
    await startResearch(query, answer)
  }

  const handleClarifySkip = async () => {
    const query = pendingQueryRef.current
    setClarifyPending(false)
    setClarifyQuestions([])
    setClarifyAnswer('')
    await startResearch(query, '')
  }

  const handleStop = () => {
    if (!isStreaming) return
    abortRef.current?.abort()
    abortRef.current = null
    setIsStreaming(false)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  useEffect(() => {
    if (historyOpen) loadHistory()
  }, [historyOpen])

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <p className="eyebrow">LangGraph Streaming Lab</p>
          <h1>{mode === 'chat' ? 'Streaming Chatbot' : 'Deep Research'}</h1>
          <p className="subhead">
            {mode === 'chat'
              ? 'Ask a question and watch the response arrive token by token.'
              : 'Submit a research topic and stream the report as it is written.'}
          </p>
        </div>
        <div className="header-right">
          <div className="status">
            <span className={isStreaming ? 'dot dot--live' : 'dot'} />
            {isStreaming ? 'Streaming' : 'Idle'}
          </div>
          {mode === 'research' && (
            <button
              className="button-ghost"
              onClick={() => setHistoryOpen((prev) => !prev)}
              disabled={isStreaming}
            >
              History
            </button>
          )}
        </div>
      </header>

      {/* History Panel */}
      {historyOpen && mode === 'research' && (
        <div className="history-panel">
          <div className="history-panel__header">
            <h3>Research History</h3>
            <button className="button-ghost" onClick={() => setHistoryOpen(false)}>
              Close
            </button>
          </div>
          {historyLoading ? (
            <p className="history-empty">Loading…</p>
          ) : historyRuns.length === 0 ? (
            <p className="history-empty">No past research runs found.</p>
          ) : (
            <div className="history-list">
              {historyRuns.map((run) => (
                <div key={run.id} className="history-item" onClick={() => restoreRun(run.id)}>
                  <div className="history-item__query">{run.query}</div>
                  <div className="history-item__meta">
                    <span className={`history-status history-status--${run.status}`}>
                      {run.status}
                    </span>
                    <span>{run.depth}</span>
                    <span>{run.updated_at ? new Date(run.updated_at).toLocaleString() : ''}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Clarification Dialog */}
      {clarifyPending && (
        <div className="clarify-overlay">
          <div className="clarify-dialog">
            <p className="clarify-eyebrow">Before we begin</p>
            <h3>A few clarifying questions</h3>
            <ul className="clarify-questions">
              {clarifyQuestions.map((q, i) => (
                <li key={i}>{q}</li>
              ))}
            </ul>
            <textarea
              className="clarify-textarea"
              placeholder="Your answers (optional)…"
              value={clarifyAnswer}
              onChange={(e) => setClarifyAnswer(e.target.value)}
              rows={3}
            />
            <div className="clarify-actions">
              <button onClick={handleClarifySubmit}>Start Research</button>
              <button className="button-ghost" onClick={handleClarifySkip}>
                Skip
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="chat">
        <div className="mode-toggle">
          <button
            className={mode === 'chat' ? 'mode-button mode-button--active' : 'mode-button'}
            onClick={() => setMode('chat')}
            disabled={isStreaming}
          >
            Chat
          </button>
          <button
            className={
              mode === 'research' ? 'mode-button mode-button--active' : 'mode-button'
            }
            onClick={() => setMode('research')}
            disabled={isStreaming}
          >
            Research
          </button>
          {mode === 'research' && (
            <>
              <div className="depth-toggle">
                <label>Depth</label>
                <select
                  value={depth}
                  onChange={(event) => setDepth(event.target.value as Depth)}
                  disabled={isStreaming}
                >
                  <option value="shallow">Shallow</option>
                  <option value="standard">Standard</option>
                  <option value="deep">Deep</option>
                </select>
              </div>
              <button
                className="button-ghost advanced-toggle"
                onClick={() => setShowAdvanced((v) => !v)}
                disabled={isStreaming}
              >
                {showAdvanced ? 'Hide Advanced' : 'Advanced'}
              </button>
            </>
          )}
        </div>

        {/* Advanced options */}
        {mode === 'research' && showAdvanced && (
          <div className="advanced-panel">
            <label className="advanced-label">
              Iterations: <strong>{maxIterations}</strong>
              <input
                type="range"
                min={1}
                max={5}
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                disabled={isStreaming}
                className="iterations-slider"
              />
              <span className="slider-range">1 — 5</span>
            </label>
          </div>
        )}

        {mode === 'research' && progressEvents.length > 0 && (
          <div className="progress-panel">
            <div className="progress-header">
              <div>
                <p className="progress-eyebrow">DeepSearch</p>
                <h3>Research Progress</h3>
              </div>
              <button
                className="progress-toggle"
                onClick={() => setProgressOpen((prev) => !prev)}
              >
                {progressOpen ? 'Hide' : 'Show'}
              </button>
            </div>
            <div className="progress-state">
              <div className="pulse-dot" />
              <span>{activeStage ? `Active: ${activeStage}` : 'Preparing'}</span>
            </div>
            {progressOpen && (
              <div className="progress-body">
                <aside className="progress-timeline">
                  {stages.map((stage) => (
                    <div
                      key={stage}
                      className={`timeline-item ${
                        stage === activeStage ? 'timeline-item--active' : ''
                      }`}
                    >
                      <div className="timeline-dot" />
                      <div className="timeline-label">{stage}</div>
                    </div>
                  ))}
                </aside>
                <div className="progress-stream">
                  {/* Plan Preview Card */}
                  {planPreview && planPreview.length > 0 && (
                    <div className="plan-preview-card">
                      <div className="plan-preview-card__header">Research Plan</div>
                      <ul className="plan-preview-card__queries">
                        {planPreview.map((q, i) => (
                          <li key={i}>{q}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {lastSearchEvent && (
                    <div className="search-summary">
                      <span>Searching</span>
                      <strong>{lastSearchEvent.message}</strong>
                    </div>
                  )}
                  {latestSources.length > 0 && (
                    <div className="source-cards">
                      <div className="source-cards__header">
                        <span>Sources</span>
                        <strong>{latestSources.length} highlighted</strong>
                      </div>
                      {latestSources.map((source) => {
                        const domain = new URL(source.url).hostname
                        const cred = source.credibility ?? 0
                        return (
                          <a
                            key={source.url}
                            className="source-card"
                            href={source.url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <div className="source-card__header">
                              <img
                                src={`https://www.google.com/s2/favicons?domain=${domain}&sz=64`}
                                alt=""
                              />
                              <span>{domain}</span>
                              <span
                                className="credibility-badge"
                                style={{ background: credibilityColor(cred) }}
                              >
                                {Math.round(cred * 100)}%
                              </span>
                            </div>
                            <div className="source-card__title">{source.title}</div>
                          </a>
                        )
                      })}
                    </div>
                  )}

                  {/* Uncertainty gauge (shown when verify result arrives) */}
                  {!!verifyEvent?.data && (
                    <div className="verify-card">
                      <div className="verify-card__title">Evidence Confidence</div>
                      <UncertaintyGauge value={Number(verifyEvent.data.uncertainty)} />
                      {!!verifyEvent.data.verifier_notes && (
                        <details className="verifier-notes">
                          <summary>Verifier notes</summary>
                          <p>{String(verifyEvent.data.verifier_notes)}</p>
                        </details>
                      )}
                    </div>
                  )}

                  <div className="progress-list">
                    {progressEvents.map((event) => (
                      <div
                        key={event.id}
                        className={`progress-item ${
                          event.stage === activeStage ? 'progress-item--active' : ''
                        }`}
                      >
                        <div className="progress-item__stage">{event.stage}</div>
                        <div className="progress-item__message">{event.message}</div>
                        {!!event.data?.queries && (
                          <div className="progress-item__meta">
                            <strong>Queries:</strong>{' '}
                            {(event.data.queries as string[]).join('; ')}
                          </div>
                        )}
                        {!!event.data?.providers && (
                          <div className="progress-item__meta">
                            <strong>Providers:</strong>{' '}
                            {Object.entries(event.data.providers as Record<string, number>)
                              .map(([provider, count]) => `${provider}(${count})`)
                              .join(' · ')}
                          </div>
                        )}
                        {!!event.data?.documents && (
                          <div className="progress-item__meta">
                            <strong>Documents:</strong> {Number(event.data.documents)}
                          </div>
                        )}
                        {!!event.data?.chunks && (
                          <div className="progress-item__meta">
                            <strong>Chunks:</strong> {Number(event.data.chunks)}
                          </div>
                        )}
                        {!!event.data?.gaps && (
                          <div className="progress-item__meta">
                            <strong>Gaps:</strong>{' '}
                            {(event.data.gaps as string[]).join('; ')}
                          </div>
                        )}
                        {event.data?.confidence !== undefined && (
                          <div className="progress-item__meta">
                            <strong>Confidence:</strong>{' '}
                            {Math.round(Number(event.data.confidence) * 100)}%
                          </div>
                        )}
                        {!!event.data?.evaluation && (
                          <div className="eval-scores">
                            {(['coverage', 'evidence', 'clarity'] as const).map((key) => {
                              const evalData = event.data!.evaluation as Record<string, number>
                              const val = evalData[key]
                              return val !== undefined ? (
                                <div key={key} className="eval-score">
                                  <span className="eval-score__label">{key}</span>
                                  <span className="eval-score__value">{val}/5</span>
                                </div>
                              ) : null
                            })}
                          </div>
                        )}
                        {/* Warning events */}
                        {event.stage === 'warning' && (
                          <div className="progress-item__warning">{event.message}</div>
                        )}
                      </div>
                    ))}
                  </div>
                  {!!transparencyEvent?.data?.transparency && (() => {
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    const trl = transparencyEvent.data!.transparency as any
                    return (
                      <div className="transparency-panel">
                        <h4>Transparency</h4>
                        <div>
                          <strong>Queries:</strong>{' '}
                          {(trl.queries as string[]).join('; ')}
                        </div>
                        <div>
                          <strong>Sources:</strong>{' '}
                          {(trl.sources as Array<{ title: string; credibility: number }>)
                            .map((src) => `${src.title} (${Math.round(src.credibility * 100)}%)`)
                            .join('; ')}
                        </div>
                        {!!transparencyEvent.data!.verifier_notes && (
                          <details className="verifier-notes" style={{ marginTop: 8 }}>
                            <summary>Verifier notes</summary>
                            <p>{String(transparencyEvent.data!.verifier_notes)}</p>
                          </details>
                        )}
                      </div>
                    )
                  })()}
                </div>
              </div>
            )}
          </div>
        )}
        <div className="chat__window" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="chat__empty">
              <p>Send your first message to start the stream.</p>
              <p className="hint">Tip: try "Summarize LangGraph streaming modes."</p>
            </div>
          )}
          {messages.map((message) => (
            <div key={message.id} className={`bubble bubble--${message.role}`}>
              <div className="bubble__role">
                {message.role === 'user' ? 'You' : 'Assistant'}
              </div>
              <div className="bubble__content">
                {message.content ? (
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                ) : (
                  '...'
                )}
              </div>
              {message.role === 'assistant' && message.toolEvents?.length ? (
                <details className="tool-panel">
                  <summary>Tool activity</summary>
                  <div className="tool-list">
                    {message.toolEvents.map((event) => (
                      <div key={event.id} className="tool-entry">
                        <div className="tool-entry__title">
                          {event.kind === 'start' ? 'Running' : 'Result'}: {event.name}
                        </div>
                        <pre className="tool-entry__payload">{event.payload}</pre>
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}
            </div>
          ))}
        </div>

        {/* Export buttons — shown after report completes */}
        {mode === 'research' && reportDone && reportText && (
          <div className="export-bar">
            <span className="export-bar__label">Report ready</span>
            <button className="button-export" onClick={downloadReport}>
              Download .md
            </button>
            <button className="button-export" onClick={copyReport}>
              Copy to clipboard
            </button>
          </div>
        )}

        <div className="composer">
          <textarea
            placeholder={mode === 'chat' ? 'Ask anything...' : 'Enter a research topic...'}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
          />
          <button onClick={handleSend} disabled={!input.trim() || isStreaming}>
            Send
          </button>
          <button onClick={handleStop} disabled={!isStreaming} className="button-stop">
            Stop
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </main>
    </div>
  )
}

export default App
