import { useEffect, useRef, useState } from 'react'

type ApiStatus = 'checking' | 'connected' | 'disconnected'
type AnalysisState = 'idle' | 'analyzing' | 'error'

type AudioInfo = {
  path: string
  name: string
  sample_rate: number
  duration_sec: number
  channels: number
  truncated: boolean
}

type SpectrogramPreview = {
  width: number
  height: number
  data: number[]
}

type OpenAudioResult =
  | { status: 'ok'; audio: AudioInfo; preview: SpectrogramPreview }
  | { status: 'cancelled' }
  | { status: 'error'; message?: string }

const tools = [
  { id: 'select', label: 'Select', key: 'V' },
  { id: 'trace', label: 'Trace', key: 'P' },
  { id: 'erase', label: 'Erase', key: 'E' },
  { id: 'connect', label: 'Connect', key: 'C' },
]

const menuSections = [
  {
    label: 'Project',
    items: [
      { label: 'New Project' },
      { label: 'Open Audio...' },
      { label: 'Save Project' },
      { label: 'Save As...' },
    ],
  },
  {
    label: 'Analysis',
    items: [{ label: 'Analysis Settings...' }, { label: 'Plugin Manager...' }],
  },
  {
    label: 'View',
    items: [{ label: 'Zoom In' }, { label: 'Zoom Out' }, { label: 'Reset View' }],
  },
  {
    label: 'System',
    items: [{ label: 'About SOMA' }, { label: 'Quit' }],
  },
]

const formatDuration = (seconds: number) => {
  const total = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(total / 60)
  const secs = total % 60
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

function App() {
  const [ready, setReady] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeTool, setActiveTool] = useState('trace')
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLooping, setIsLooping] = useState(false)
  const [mixValue, setMixValue] = useState(55)
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [analysisState, setAnalysisState] = useState<AnalysisState>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [audioInfo, setAudioInfo] = useState<AudioInfo | null>(null)
  const [preview, setPreview] = useState<SpectrogramPreview | null>(null)
  const [statusNote, setStatusNote] = useState<string | null>(null)

  const menuRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setReady(true))
    return () => window.cancelAnimationFrame(frame)
  }, [])

  useEffect(() => {
    let alive = true

    const checkApi = async () => {
      const api = window.pywebview?.api
      if (!api?.health) {
        if (alive) setApiStatus('disconnected')
        return
      }
      try {
        const result = await api.health()
        if (alive) {
          setApiStatus(result?.status === 'ok' ? 'connected' : 'disconnected')
        }
      } catch {
        if (alive) setApiStatus('disconnected')
      }
    }

    void checkApi()
    const handleReady = () => {
      void checkApi()
    }
    window.addEventListener('pywebviewready', handleReady)
    return () => {
      alive = false
      window.removeEventListener('pywebviewready', handleReady)
    }
  }, [])

  useEffect(() => {
    if (!menuOpen) return
    const closeMenu = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', closeMenu)
    return () => {
      document.removeEventListener('mousedown', closeMenu)
    }
  }, [menuOpen])

  useEffect(() => {
    if (!preview || !canvasRef.current) return
    const canvas = canvasRef.current
    canvas.width = preview.width
    canvas.height = preview.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const image = ctx.createImageData(preview.width, preview.height)
    for (let i = 0; i < preview.data.length; i += 1) {
      const value = preview.data[i]
      const offset = i * 4
      image.data[offset] = value
      image.data[offset + 1] = value
      image.data[offset + 2] = value
      image.data[offset + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [preview])

  const openAudio = async () => {
    setMenuOpen(false)
    setStatusNote(null)
    const api = window.pywebview?.api
    if (!api?.open_audio) {
      setAnalysisState('error')
      setAnalysisError('Pywebview API is not available.')
      return
    }
    setAnalysisState('analyzing')
    setAnalysisError(null)
    try {
      const result = (await api.open_audio()) as OpenAudioResult
      if (result.status === 'ok') {
        setAudioInfo(result.audio)
        setPreview(result.preview)
        setAnalysisState('idle')
      } else if (result.status === 'cancelled') {
        setAnalysisState('idle')
      } else {
        setAnalysisState('error')
        setAnalysisError(result.message ?? 'Failed to load audio.')
      }
    } catch (error) {
      setAnalysisState('error')
      setAnalysisError(error instanceof Error ? error.message : 'Failed to load audio.')
    }
  }

  const handleMenuAction = (label: string) => {
    if (label === 'Open Audio...') {
      void openAudio()
      return
    }
    setMenuOpen(false)
    setStatusNote(`${label} is not implemented yet.`)
  }

  const statusLabel =
    analysisState === 'analyzing'
      ? 'Analyzing'
      : analysisState === 'error'
        ? 'Error'
        : isPlaying
          ? 'Playing'
          : 'Ready'
  const apiBadge =
    apiStatus === 'connected' ? 'API Connected' : apiStatus === 'checking' ? 'API Checking' : 'API Offline'

  return (
    <div className={`page ${ready ? 'is-ready' : ''}`}>
      <div className="mx-auto flex min-h-screen w-full max-w-[1400px] flex-col gap-4 px-4 pb-6 pt-6 sm:px-6">
        <header className="panel flex flex-col gap-4 rounded-2xl px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            <div className="relative" ref={menuRef}>
              <button
                className="rounded-full border border-[var(--panel-border)] bg-white px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]"
                onClick={() => setMenuOpen((prev) => !prev)}
                aria-expanded={menuOpen}
              >
                Menu
              </button>
              {menuOpen ? (
                <div className="absolute left-0 top-[calc(100%+8px)] z-20 w-56 rounded-2xl border border-[var(--panel-border)] bg-white p-3 text-xs shadow-lg">
                  {menuSections.map((section) => (
                    <div key={section.label} className="border-b border-[var(--panel-border)] py-2 last:border-b-0">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">
                        {section.label}
                      </div>
                      <div className="mt-2 flex flex-col gap-1">
                        {section.items.map((item) => (
                          <button
                            key={item.label}
                            className="rounded-lg px-2 py-1 text-left text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--ink)] hover:bg-[var(--panel-strong)]"
                            onClick={() => handleMenuAction(item.label)}
                          >
                            {item.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
            <div>
              <div className="text-xl font-semibold tracking-[0.2em] text-[var(--ink)]">SOMA</div>
              <div className="text-[10px] uppercase tracking-[0.3em] text-[var(--muted)]">
                Sonic Observation
              </div>
            </div>
            <span className="rounded-full border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
              {apiBadge}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2 rounded-full border border-[var(--panel-border)] bg-white px-3 py-2 text-sm font-semibold text-[var(--ink)]">
              <button
                onClick={() => setIsPlaying(false)}
                className="rounded-full border border-transparent px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)] hover:border-[var(--panel-border)]"
              >
                Stop
              </button>
              <button
                onClick={() => setIsPlaying((prev) => !prev)}
                className="rounded-full bg-[var(--accent)] px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-white"
              >
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button
                onClick={() => setIsLooping((prev) => !prev)}
                className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${
                  isLooping
                    ? 'bg-[var(--accent-warm)] text-white'
                    : 'border border-[var(--panel-border)] text-[var(--muted)]'
                }`}
              >
                Loop
              </button>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
                Original {100 - mixValue}% / Resynth {mixValue}%
              </span>
              <input
                aria-label="Original and resynth mix"
                className="h-1 w-40 accent-[var(--accent)]"
                type="range"
                min={0}
                max={100}
                value={mixValue}
                onChange={(event) => setMixValue(Number(event.target.value))}
              />
            </div>
            <div className="rounded-full border border-[var(--panel-border)] bg-[var(--panel-strong)] px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]">
              00:00:00.000
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="grid grid-cols-2 gap-2 rounded-2xl border border-[var(--panel-border)] bg-white p-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)] lg:grid-cols-4">
              {tools.map((tool) => (
                <button
                  key={tool.id}
                  onClick={() => setActiveTool(tool.id)}
                  className={`rounded-xl px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] ${
                    activeTool === tool.id
                      ? 'bg-[var(--accent)] text-white'
                      : 'text-[var(--muted)] hover:bg-[var(--panel-strong)]'
                  }`}
                >
                  {tool.label}
                  <span className="ml-2 font-mono text-[9px] text-white/70">{tool.key}</span>
                </button>
              ))}
            </div>
            <button className="rounded-full bg-[var(--accent-warm)] px-5 py-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-white">
              Export
            </button>
          </div>
        </header>

        <main className="flex flex-1 flex-col gap-4">
          <section className="panel rounded-2xl px-4 py-4">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
              <span>Workspace</span>
              <span className="font-mono text-[10px]">Zoom 100%</span>
            </div>
            <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_72px]">
              <div className="canvas-surface relative rounded-2xl p-4">
                <canvas
                  ref={canvasRef}
                  className="h-full min-h-[320px] w-full rounded-xl border border-white/10"
                  style={{ imageRendering: 'pixelated' }}
                />
                {!preview ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center text-white/70">
                    <div className="text-xs uppercase tracking-[0.3em]">No Audio Loaded</div>
                    <button
                      className="rounded-full bg-white/90 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]"
                      onClick={() => void openAudio()}
                    >
                      Open Audio
                    </button>
                  </div>
                ) : null}
                {analysisState === 'analyzing' ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-xs uppercase tracking-[0.24em] text-white">
                    Analyzing...
                  </div>
                ) : null}
              </div>
              <div className="panel flex flex-col items-center justify-between rounded-2xl px-3 py-4 text-[10px] uppercase tracking-[0.24em] text-[var(--muted)]">
                <span>Freq</span>
                <div className="flex flex-col items-center gap-3 font-mono text-[10px] text-[var(--accent-strong)]">
                  <span>20k</span>
                  <span>5k</span>
                  <span>1k</span>
                  <span>200</span>
                  <span>50</span>
                </div>
                <span>Hz</span>
              </div>
            </div>
          </section>

          <section className="grid gap-3 lg:grid-cols-[1.2fr_1fr]">
            <div className="panel rounded-2xl px-4 py-3 text-sm text-[var(--muted)]">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
                <span>Selection HUD</span>
                <span className="font-mono text-[10px]">Partial 00045</span>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <button className="rounded-full border border-[var(--panel-border)] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-strong)]">
                  Mute
                </button>
                <button className="rounded-full border border-[var(--panel-border)] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-warm)]">
                  Delete
                </button>
                <span className="text-[10px] uppercase tracking-[0.2em] text-[var(--muted)]">
                  T: 12.3s | F: 440Hz | A: -6dB
                </span>
              </div>
            </div>

            <div className="panel rounded-2xl px-4 py-3 text-sm text-[var(--muted)]">
              <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">Audio Info</div>
              <div className="mt-3 flex flex-col gap-2 text-[11px]">
                <span className="font-semibold text-[var(--ink)]">
                  {audioInfo ? audioInfo.name : 'No file'}
                </span>
                <span className="font-mono text-[10px] text-[var(--muted)]">
                  {audioInfo
                    ? `${audioInfo.sample_rate} Hz | ${audioInfo.channels} ch | ${formatDuration(audioInfo.duration_sec)}`
                    : 'Load a WAV file to analyze.'}
                </span>
                {audioInfo?.truncated ? (
                  <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent-warm)]">
                    Preview uses first 30 seconds.
                  </span>
                ) : null}
                {analysisError ? (
                  <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent-warm)]">
                    {analysisError}
                  </span>
                ) : null}
                {statusNote ? (
                  <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--muted)]">{statusNote}</span>
                ) : null}
              </div>
            </div>
          </section>
        </main>

        <footer className="panel flex flex-col gap-2 rounded-2xl px-4 py-3 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between">
          <span>Status: {statusLabel}</span>
          <span className="font-mono text-[10px]">T: 12.3s | F: 440Hz (A4) | A: -6dB</span>
          <span className="font-mono text-[10px]">Mem: 42MB</span>
        </footer>
      </div>
    </div>
  )
}

export default App
