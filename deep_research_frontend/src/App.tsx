import { useRef, useState } from 'react'
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

const API_URL = 'http://127.0.0.1:8000'

type Mode = 'chat' | 'research'
type Depth = 'shallow' | 'standard' | 'deep'

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
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef(crypto.randomUUID())

  const stages = [
    'plan',
    'queries',
    'search',
    'sources',
    'extract',
    'documents',
    'retrieval',
    'gaps',
    'researchers',
    'analyst',
    'critic',
    'writer',
    'transparency',
    'done',
  ]

  const latestSources = (() => {
    const sourceEvent = [...progressEvents].reverse().find((event) => event.data?.top_sources)
    if (!sourceEvent) return []
    return sourceEvent.data?.top_sources as Array<{ title: string; url: string; provider?: string }>
  })()

  const lastSearchEvent = [...progressEvents].reverse().find((event) => event.stage === 'search')
  const transparencyEvent = [...progressEvents]
    .reverse()
    .find((event) => event.stage === 'transparency')

  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      })
    })
  }

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return

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
    if (mode === 'research') {
      setProgressEvents([])
    }
    setIsStreaming(true)
    scrollToBottom()

    try {
      const controller = new AbortController()
      abortRef.current = controller
      const endpoint =
        mode === 'chat' ? `${API_URL}/chat/stream` : `${API_URL}/research/stream`
      const payload =
        mode === 'chat'
          ? { session_id: sessionIdRef.current, message: nextUserMessage.content }
          : {
              session_id: sessionIdRef.current,
              query: nextUserMessage.content,
              depth,
            }
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      if (!response.ok || !response.body) {
        let details = ''
        try {
          details = await response.text()
        } catch {
          details = ''
        }
        console.error('Streaming request failed', {
          status: response.status,
          statusText: response.statusText,
          details,
          endpoint,
          mode,
        })
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
              const payload = JSON.parse(data) as {
                delta?: string
                error?: string
                type?: string
                name?: string
                input?: unknown
                output?: unknown
                stage?: string
                message?: string
                data?: Record<string, unknown>
              }
              if (payload.error) {
                console.error('Streaming payload error', payload)
                setError(payload.error)
                done = true
                break
              }
              if (payload.type === 'status') {
                const event: ProgressEvent = {
                  id: crypto.randomUUID(),
                  stage: payload.stage || 'update',
                  message: payload.message || '',
                  data: (payload.data as Record<string, unknown>) ?? null,
                }
                setProgressEvents((prev) => [...prev, event])
                setActiveStage(event.stage)
                continue
              }
              if (payload.type === 'tool_start' || payload.type === 'tool_end') {
                const toolEvent: ToolEvent = {
                  id: crypto.randomUUID(),
                  name: payload.name || 'tool',
                  kind: payload.type === 'tool_start' ? 'start' : 'end',
                  payload:
                    payload.type === 'tool_start'
                      ? payload.input
                        ? JSON.stringify(payload.input, null, 2)
                        : '…'
                      : payload.output
                      ? JSON.stringify(payload.output, null, 2)
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
              if (payload.delta) {
                setMessages((prev) => {
                  const updated = [...prev]
                  const lastIndex = updated.findIndex(
                    (msg) => msg.id === nextAssistantMessage.id
                  )
                  if (lastIndex === -1) return prev
                  updated[lastIndex] = {
                    ...updated[lastIndex],
                    content: updated[lastIndex].content + payload.delta,
                  }
                  return updated
                })
                scrollToBottom()
              }
            } catch (err) {
              console.error(err)
              setError('Streaming failed. Please retry.')
              done = true
              break
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setError(null)
      } else {
        console.error('Streaming request exception', err)
        setError('Unable to connect to the backend.')
      }
    } finally {
      setIsStreaming(false)
      scrollToBottom()
    }
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
        <div className="status">
          <span className={isStreaming ? 'dot dot--live' : 'dot'} />
          {isStreaming ? 'Streaming' : 'Idle'}
        </div>
      </header>

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
          )}
        </div>
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
                            </div>
                            <div className="source-card__title">{source.title}</div>
                          </a>
                        )
                      })}
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
                        {event.data?.queries && (
                          <div className="progress-item__meta">
                            <strong>Queries:</strong>{' '}
                            {(event.data.queries as string[]).join('; ')}
                          </div>
                        )}
                        {event.data?.providers && (
                          <div className="progress-item__meta">
                            <strong>Providers:</strong>{' '}
                            {Object.entries(event.data.providers as Record<string, number>)
                              .map(([provider, count]) => `${provider}(${count})`)
                              .join(' · ')}
                          </div>
                        )}
                        {event.data?.documents && (
                          <div className="progress-item__meta">
                            <strong>Documents:</strong> {event.data.documents as number}
                          </div>
                        )}
                        {event.data?.chunks && (
                          <div className="progress-item__meta">
                            <strong>Chunks:</strong> {event.data.chunks as number}
                          </div>
                        )}
                        {event.data?.gaps && (
                          <div className="progress-item__meta">
                            <strong>Gaps:</strong>{' '}
                            {(event.data.gaps as string[]).join('; ')}
                          </div>
                        )}
                        {event.data?.confidence && (
                          <div className="progress-item__meta">
                            <strong>Confidence:</strong> {event.data.confidence as number}
                          </div>
                        )}
                        {event.data?.evaluation && (
                          <div className="progress-item__meta">
                            <strong>Evaluation:</strong>{' '}
                            {`coverage ${event.data.evaluation.coverage}, evidence ${event.data.evaluation.evidence}, clarity ${event.data.evaluation.clarity}`}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                  {transparencyEvent?.data?.transparency && (
                    <div className="transparency-panel">
                      <h4>Transparency</h4>
                      <div>
                        <strong>Queries:</strong>{' '}
                        {(transparencyEvent.data.transparency.queries as string[]).join('; ')}
                      </div>
                      <div>
                        <strong>Sources:</strong>{' '}
                        {(transparencyEvent.data.transparency.sources as Array<{ title: string }>).map(
                          (src) => src.title
                        ).join('; ')}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        <div className="chat__window" ref={scrollRef}>
          {messages.length === 0 && (
            <div className="chat__empty">
              <p>Send your first message to start the stream.</p>
              <p className="hint">Tip: try “Summarize LangGraph streaming modes.”</p>
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
