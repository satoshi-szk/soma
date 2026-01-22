import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnalysisSettingsModal } from './components/modals/AnalysisSettingsModal'
import { ExportModal } from './components/modals/ExportModal'
import { HeaderToolbar } from './components/HeaderToolbar'
import { Workspace } from './components/Workspace'
import { SelectionHud } from './components/SelectionHud'
import { AudioInfoPanel } from './components/AudioInfoPanel'
import { StatusBar } from './components/StatusBar'
import { DEFAULT_SETTINGS } from './app/constants'
import type { AnalysisSettings, AudioInfo, Partial, PartialPoint, SpectrogramPreview, ToolId } from './app/types'
import { formatDuration, toPartial } from './app/utils'
import { useApiStatus } from './hooks/useApiStatus'

function App() {
  const [ready, setReady] = useState(false)
  const apiStatus = useApiStatus()
  const [analysisState, setAnalysisState] = useState<'idle' | 'analyzing' | 'error'>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [audioInfo, setAudioInfo] = useState<AudioInfo | null>(null)
  const [preview, setPreview] = useState<SpectrogramPreview | null>(null)
  const [partials, setPartials] = useState<Partial[]>([])
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULT_SETTINGS)
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeTool, setActiveTool] = useState<ToolId>('trace')
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLooping, setIsLooping] = useState(false)
  const [mixValue, setMixValue] = useState(55)
  const [statusNote, setStatusNote] = useState<string | null>(null)
  const [playbackPosition, setPlaybackPosition] = useState(0)
  const [selection, setSelection] = useState<string[]>([])
  const [showAnalysisModal, setShowAnalysisModal] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)
  const [exportTab, setExportTab] = useState<'mpe' | 'audio'>('mpe')
  const [pitchBendRange, setPitchBendRange] = useState(48)
  const [ampMapping, setAmpMapping] = useState('velocity')
  const [exportSampleRate, setExportSampleRate] = useState(44100)
  const [exportBitDepth, setExportBitDepth] = useState(16)
  const [exportType, setExportType] = useState<'sine' | 'cv'>('sine')
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const connectQueueRef = useRef<string[]>([])

  const menuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => setReady(true))
    return () => window.cancelAnimationFrame(frame)
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
    if (!isPlaying) return
    const interval = window.setInterval(async () => {
      const api = window.pywebview?.api
      if (!api?.status) return
      const result = await api.status()
      if (result?.status === 'ok') {
        if (typeof result.position === 'number') {
          setPlaybackPosition(result.position)
        }
        if (!result.is_playing) {
          setIsPlaying(false)
        }
      }
    }, 300)
    return () => window.clearInterval(interval)
  }, [isPlaying])

  useEffect(() => {
    if (analysisState !== 'analyzing') return
    let alive = true
    const poll = async () => {
      const api = window.pywebview?.api
      if (!api?.analysis_status) return
      const result = await api.analysis_status()
      if (!alive || result?.status !== 'ok') return
      if (result.state === 'ready' && result.preview) {
        setPreview(result.preview)
        setAnalysisState('idle')
        setAnalysisError(null)
      } else if (result.state === 'error') {
        setAnalysisState('error')
        setAnalysisError(result.message ?? 'Analysis failed.')
      }
    }
    void poll()
    const interval = window.setInterval(poll, 500)
    return () => {
      alive = false
      window.clearInterval(interval)
    }
  }, [analysisState])

  const updateSelection = (ids: string[]) => {
    setSelection(ids)
    if (ids.length === 1) {
      const partial = partials.find((item) => item.id === ids[0])
      if (!partial) return
      setStatusNote(`Selected ${partial.id}`)
    }
  }

  const handleMenuAction = async (label: string) => {
    setMenuOpen(false)
    setStatusNote(null)
    const api = window.pywebview?.api
    if (!api) return
    if (label === 'New Project') {
      await api.new_project()
      setAudioInfo(null)
      setPreview(null)
      setPartials([])
      setSelection([])
      return
    }
    if (label === 'Open Project...') {
      const result = await api.open_project()
      if (result?.status === 'ok' || result?.status === 'processing') {
        setAudioInfo(result.audio)
        setPreview(result.preview ?? null)
        setSettings(result.settings)
        setPartials(result.partials.map(toPartial))
        if (result.status === 'processing') {
          setAnalysisState('analyzing')
        }
      } else if (result?.status === 'error') {
        setStatusNote(result.message ?? 'Failed to open project.')
      }
      return
    }
    if (label === 'Open Audio...') {
      await openAudio()
      return
    }
    if (label === 'Save Project') {
      const result = await api.save_project()
      if (result?.status !== 'ok') {
        setStatusNote(result.message ?? 'Failed to save project.')
      }
      return
    }
    if (label === 'Save As...') {
      const result = await api.save_project_as()
      if (result?.status !== 'ok') {
        setStatusNote(result.message ?? 'Failed to save project.')
      }
      return
    }
    if (label === 'Analysis Settings...') {
      setShowAnalysisModal(true)
      return
    }
    if (label === 'Zoom In') {
      setZoom((value) => Math.min(4, value + 0.2))
      return
    }
    if (label === 'Zoom Out') {
      setZoom((value) => Math.max(0.5, value - 0.2))
      return
    }
    if (label === 'Reset View') {
      setZoom(1)
      setPan({ x: 0, y: 0 })
      return
    }
    if (label === 'Quit') {
      window.close()
      return
    }
    if (label === 'About SOMA') {
      setStatusNote('SOMA - Sonic Observation, Musical Abstraction')
      return
    }
    setStatusNote(`${label} is not implemented yet.`)
  }

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
      const result = await api.open_audio()
      if (result.status === 'ok' || result.status === 'processing') {
        setAudioInfo(result.audio)
        setPreview(result.preview ?? null)
        setSettings(result.settings)
        setPartials(result.partials.map(toPartial))
        setAnalysisState(result.status === 'processing' ? 'analyzing' : 'idle')
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

  const applySettings = async () => {
    const api = window.pywebview?.api
    if (!api?.update_settings) return
    setAnalysisState('analyzing')
    const result = await api.update_settings(settings)
    if (result.status === 'ok' || result.status === 'processing') {
      setSettings(result.settings)
      setPreview(result.preview ?? null)
      setAnalysisState(result.status === 'processing' ? 'analyzing' : 'idle')
    }
    setShowAnalysisModal(false)
  }

  const handleTraceCommit = async (trace: Array<[number, number]>) => {
    const api = window.pywebview?.api
    if (!api?.trace_partial) return
    const result = await api.trace_partial({ trace })
    if (result.status === 'ok') {
      setPartials((prev) => [...prev, toPartial(result.partial)])
    }
  }

  const handleEraseCommit = async (trace: Array<[number, number]>) => {
    const api = window.pywebview?.api
    if (!api?.erase_partial) return
    const result = await api.erase_partial({ trace, radius_hz: 60 })
    if (result.status === 'ok') {
      setPartials(result.partials.map(toPartial))
      setSelection([])
    }
  }

  const handleSelectBoxCommit = async (box: {
    time_start: number
    time_end: number
    freq_start: number
    freq_end: number
  }) => {
    const api = window.pywebview?.api
    if (!api?.select_in_box) return
    const result = await api.select_in_box(box)
    if (result.status === 'ok') {
      updateSelection(result.ids)
    }
  }

  const handleHitTestCommit = async (point: { time: number; freq: number }) => {
    const api = window.pywebview?.api
    if (!api?.hit_test) return
    const result = await api.hit_test({ ...point, tolerance: 0.08 })
    if (result.status === 'ok') {
      updateSelection(result.id ? [result.id] : [])
    }
  }

  const handleUpdatePartial = async (id: string, points: PartialPoint[]) => {
    const api = window.pywebview?.api
    if (!api?.update_partial) return
    const payload = { id, points: points.map((point) => [point.time, point.freq, point.amp]) }
    const result = await api.update_partial(payload)
    if (result.status === 'ok') {
      setPartials((prev) => prev.map((item) => (item.id === result.partial.id ? toPartial(result.partial) : item)))
    }
  }

  const handleConnectPick = async (point: { time: number; freq: number }) => {
    const api = window.pywebview?.api
    if (!api?.hit_test) return
    const result = await api.hit_test({ ...point, tolerance: 0.08 })
    if (result.status === 'ok' && result.id) {
      const current = connectQueueRef.current
      const next = current.includes(result.id) ? current : [...current, result.id]
      if (next.length === 2 && api?.merge_partials) {
        const mergeResult = await api.merge_partials({ first: next[0], second: next[1] })
        if (mergeResult.status === 'ok') {
          setPartials((items) => [...items.filter((p) => !next.includes(p.id)), toPartial(mergeResult.partial)])
          setSelection([mergeResult.partial.id])
        }
        connectQueueRef.current = []
      } else {
        connectQueueRef.current = next
      }
    }
  }

  const handlePartialMute = async () => {
    if (selection.length !== 1) return
    const api = window.pywebview?.api
    if (!api?.toggle_mute) return
    const result = await api.toggle_mute({ id: selection[0] })
    if (result.status === 'ok') {
      setPartials((prev) => prev.map((item) => (item.id === result.partial.id ? toPartial(result.partial) : item)))
    }
  }

  const handlePartialDelete = async () => {
    if (selection.length === 0) return
    const api = window.pywebview?.api
    if (!api?.delete_partials) return
    await api.delete_partials({ ids: selection })
    setPartials((prev) => prev.filter((item) => !selection.includes(item.id)))
    setSelection([])
  }

  const undo = useCallback(async () => {
    const api = window.pywebview?.api
    if (!api?.undo) return
    const result = await api.undo()
    if (result.status === 'ok') {
      setPartials(result.partials.map(toPartial))
    }
  }, [])

  const redo = useCallback(async () => {
    const api = window.pywebview?.api
    if (!api?.redo) return
    const result = await api.redo()
    if (result.status === 'ok') {
      setPartials(result.partials.map(toPartial))
    }
  }, [])

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    const key = event.key.toLowerCase()
    if (key === 'v') setActiveTool('select')
    if (key === 'p') setActiveTool('trace')
    if (key === 'e') setActiveTool('erase')
    if (key === 'c') setActiveTool('connect')
    if (key === 'z' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      void undo()
    }
    if (key === 'y' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      void redo()
    }
  }, [undo, redo])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const handlePlayToggle = async () => {
    const api = window.pywebview?.api
    if (!api?.play || !api?.pause) return
    if (isPlaying) {
      await api.pause()
      setIsPlaying(false)
    } else {
      await api.play({ mix_ratio: mixValue / 100, loop: isLooping })
      setIsPlaying(true)
    }
  }

  const handleStop = async () => {
    const api = window.pywebview?.api
    if (!api?.stop) return
    await api.stop()
    setIsPlaying(false)
    setPlaybackPosition(0)
  }

  const handleExport = async () => {
    const api = window.pywebview?.api
    if (!api) return
    if (exportTab === 'mpe') {
      const result = await api.export_mpe({ pitch_bend_range: pitchBendRange, amplitude_mapping: ampMapping })
      if (result.status === 'ok') {
        setStatusNote(`Exported ${result.paths.length} MIDI file(s).`)
      }
    } else {
      const result = await api.export_audio({
        sample_rate: exportSampleRate,
        bit_depth: exportBitDepth,
        output_type: exportType,
      })
      if (result.status === 'ok') {
        setStatusNote(`Exported ${result.path}`)
      }
    }
    setShowExportModal(false)
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

  const selectedInfo = useMemo(() => {
    if (selection.length !== 1) return null
    return partials.find((partial) => partial.id === selection[0]) ?? null
  }, [selection, partials])

  return (
    <div className={`page ${ready ? 'is-ready' : ''}`}>
      <div className="mx-auto flex min-h-screen w-full max-w-[1400px] flex-col gap-4 px-4 pb-6 pt-6 sm:px-6">
        <HeaderToolbar
          apiBadge={apiBadge}
          menuOpen={menuOpen}
          activeTool={activeTool}
          isPlaying={isPlaying}
          isLooping={isLooping}
          mixValue={mixValue}
          playbackTimeLabel={formatDuration(playbackPosition)}
          onMenuToggle={() => setMenuOpen((prev) => !prev)}
          onMenuAction={handleMenuAction}
          onToolChange={setActiveTool}
          onStop={handleStop}
          onPlayToggle={handlePlayToggle}
          onLoopToggle={() => setIsLooping((prev) => !prev)}
          onMixChange={setMixValue}
          onExport={() => setShowExportModal(true)}
          menuRef={menuRef}
        />

        <main className="flex flex-1 flex-col gap-4">
          <section className="panel rounded-2xl px-4 py-4">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
              <span>Workspace</span>
              <span className="font-mono text-[10px]">Zoom {Math.round(zoom * 100)}%</span>
            </div>
            <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_72px]">
              <Workspace
                preview={preview}
                settings={settings}
                partials={partials}
                selectedIds={selection}
                activeTool={activeTool}
                analysisState={analysisState}
                zoom={zoom}
                pan={pan}
                onZoomChange={setZoom}
                onPanChange={setPan}
                onTraceCommit={handleTraceCommit}
                onEraseCommit={handleEraseCommit}
                onSelectBoxCommit={handleSelectBoxCommit}
                onHitTestCommit={handleHitTestCommit}
                onUpdatePartial={handleUpdatePartial}
                onConnectPick={handleConnectPick}
                onOpenAudio={openAudio}
              />
              <div className="panel flex flex-col items-center justify-between rounded-2xl px-3 py-4 text-[10px] uppercase tracking-[0.24em] text-[var(--muted)]">
                <span>Freq</span>
                <div className="flex flex-col items-center gap-3 font-mono text-[10px] text-[var(--accent-strong)]">
                  <span>{Math.round((preview?.freq_max ?? settings.freq_max) / 1000)}k</span>
                  <span>{Math.round((preview?.freq_max ?? settings.freq_max) / 5000)}k</span>
                  <span>1k</span>
                  <span>200</span>
                  <span>50</span>
                </div>
                <span>Hz</span>
              </div>
            </div>
          </section>

          <section className="grid gap-3 lg:grid-cols-[1.2fr_1fr]">
            <SelectionHud
              selected={selectedInfo}
              canMute={selection.length === 1}
              canDelete={selection.length > 0}
              onMute={handlePartialMute}
              onDelete={handlePartialDelete}
            />
            <AudioInfoPanel audioInfo={audioInfo} analysisError={analysisError} statusNote={statusNote} />
          </section>
        </main>

        <StatusBar statusLabel={statusLabel} />
      </div>

      {showAnalysisModal ? (
        <AnalysisSettingsModal
          settings={settings}
          onChange={setSettings}
          onCancel={() => setShowAnalysisModal(false)}
          onApply={applySettings}
        />
      ) : null}

      {showExportModal ? (
        <ExportModal
          tab={exportTab}
          pitchBendRange={pitchBendRange}
          amplitudeMapping={ampMapping}
          exportSampleRate={exportSampleRate}
          exportBitDepth={exportBitDepth}
          exportType={exportType}
          onTabChange={setExportTab}
          onPitchBendChange={setPitchBendRange}
          onAmplitudeMappingChange={setAmpMapping}
          onSampleRateChange={setExportSampleRate}
          onBitDepthChange={setExportBitDepth}
          onOutputTypeChange={setExportType}
          onCancel={() => setShowExportModal(false)}
          onExport={handleExport}
        />
      ) : null}
    </div>
  )
}

export default App
