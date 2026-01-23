import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnalysisSettingsModal } from './components/modals/AnalysisSettingsModal'
import { ExportModal } from './components/modals/ExportModal'
import { HeaderToolbar } from './components/HeaderToolbar'
import { Workspace } from './components/Workspace'
import { StatusBar } from './components/StatusBar'
import type { ToolId } from './app/types'
import { formatDuration, formatNote } from './app/utils'
import { isPywebviewApiAvailable, pywebviewApi } from './app/pywebviewApi'
import { useApiStatus } from './hooks/useApiStatus'
import { useAnalysis } from './hooks/useAnalysis'
import { usePartials } from './hooks/usePartials'
import { usePlayback } from './hooks/usePlayback'
import { useViewport } from './hooks/useViewport'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'

function App() {
  const [ready, setReady] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [activeTool, setActiveTool] = useState<ToolId>('trace')
  const [statusNote, setStatusNote] = useState<string | null>(null)
  const [cursorInfo, setCursorInfo] = useState<{ time: number; freq: number; amp: number | null }>({
    time: 0,
    freq: 440,
    amp: null,
  })
  const [showAnalysisModal, setShowAnalysisModal] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)
  const [exportTab, setExportTab] = useState<'mpe' | 'audio'>('mpe')
  const [pitchBendRange, setPitchBendRange] = useState(48)
  const [ampMapping, setAmpMapping] = useState('cc74')
  const [exportSampleRate, setExportSampleRate] = useState(44100)
  const [exportBitDepth, setExportBitDepth] = useState(16)
  const [exportType, setExportType] = useState<'sine' | 'cv'>('sine')

  const menuRef = useRef<HTMLDivElement | null>(null)

  const reportError = useCallback((context: string, message: string) => {
    const detail = `${context}: ${message}`
    setStatusNote(detail)
    console.error(detail)
  }, [])

  const apiStatus = useApiStatus()
  const analysis = useAnalysis(reportError)
  const partialsHook = usePartials(reportError)
  const playback = usePlayback(reportError, analysis.analysisState)
  const viewport = useViewport(analysis.preview)

  useKeyboardShortcuts({
    onToolChange: setActiveTool,
    onUndo: partialsHook.undo,
    onRedo: partialsHook.redo,
    onPlayToggle: () => {
      if (analysis.analysisState !== 'analyzing') playback.togglePlay()
    },
  })

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

  const handleMenuAction = async (label: string) => {
    setMenuOpen(false)
    setStatusNote(null)

    if (label === 'New Project') {
      const success = await analysis.newProject()
      if (success) {
        partialsHook.setPartials([])
        partialsHook.setSelection([])
        setStatusNote('New project ready')
      }
      return
    }
    if (label === 'Open Project...') {
      const result = await analysis.openProject()
      if (result) {
        partialsHook.setPartials(result.partials)
        setStatusNote('Project opened')
      }
      return
    }
    if (label === 'Open Audio...') {
      const result = await analysis.openAudio()
      if (result) {
        partialsHook.setPartials(result.partials)
        setStatusNote('Audio loaded')
      }
      return
    }
    if (label === 'Save Project') {
      const success = await analysis.saveProject()
      if (success) setStatusNote('Project saved')
      return
    }
    if (label === 'Save As...') {
      const success = await analysis.saveProjectAs()
      if (success) setStatusNote('Project saved')
      return
    }
    if (label === 'Analysis Settings...') {
      setShowAnalysisModal(true)
      return
    }
    if (label === 'Export...') {
      setShowExportModal(true)
      return
    }
    if (label === 'Zoom In') {
      viewport.zoomInX()
      return
    }
    if (label === 'Zoom Out') {
      viewport.zoomOutX()
      return
    }
    if (label === 'Reset View') {
      viewport.resetView()
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

  const handleOpenAudio = async () => {
    setMenuOpen(false)
    setStatusNote(null)
    const result = await analysis.openAudio()
    if (result) {
      partialsHook.setPartials(result.partials)
      setStatusNote('Audio loaded')
    }
  }

  const handleApplySettings = async () => {
    const success = await analysis.applySettings(analysis.settings)
    if (success) setStatusNote('Settings applied')
    setShowAnalysisModal(false)
  }

  const handleTraceCommit = async (trace: Array<[number, number]>) => {
    const success = await partialsHook.trace(trace)
    if (success) setStatusNote('Trace snapped')
    return success
  }

  const handleEraseCommit = async (trace: Array<[number, number]>) => {
    await partialsHook.erase(trace)
    setStatusNote('Erase applied')
  }

  const handlePartialMute = async () => {
    await partialsHook.toggleMute()
    const partial = partialsHook.partials.find((p) => p.id === partialsHook.selection[0])
    if (partial) {
      setStatusNote(partial.is_muted ? 'Partial unmuted' : 'Partial muted')
    }
  }

  const handlePartialDelete = async () => {
    await partialsHook.deleteSelected()
    setStatusNote('Partial deleted')
  }

  const handleExport = async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api) {
      reportError('Export', 'API not available')
      return
    }
    try {
      if (exportTab === 'mpe') {
        const result = await api.export_mpe({ pitch_bend_range: pitchBendRange, amplitude_mapping: ampMapping })
        if (result.status === 'ok') {
          setStatusNote(`Exported ${result.paths.length} MIDI file(s).`)
        } else if (result.status === 'error') {
          reportError('Export', result.message ?? 'Failed to export MIDI')
        }
      } else {
        const result = await api.export_audio({
          sample_rate: exportSampleRate,
          bit_depth: exportBitDepth,
          output_type: exportType,
        })
        if (result.status === 'ok') {
          setStatusNote(`Exported ${result.path}`)
        } else if (result.status === 'error') {
          reportError('Export', result.message ?? 'Failed to export audio')
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Export', message)
    }
    setShowExportModal(false)
  }

  const handlePlayToggle = () => {
    if (analysis.analysisState !== 'analyzing') playback.togglePlay()
  }

  const statusLabel =
    analysis.analysisState === 'analyzing'
      ? 'Analyzing'
      : analysis.analysisState === 'error'
        ? 'Error'
        : playback.isPlaying
          ? 'Playing'
          : 'Ready'

  const apiBadge =
    apiStatus === 'connected' ? 'API Connected' : apiStatus === 'checking' ? 'API Checking' : 'API Offline'

  const selectedInfo = useMemo(() => {
    if (partialsHook.selection.length !== 1) return null
    return partialsHook.partials.find((partial) => partial.id === partialsHook.selection[0]) ?? null
  }, [partialsHook.selection, partialsHook.partials])

  const cursorLabel = useMemo(() => {
    if (!analysis.preview) return 'T: -- | F: -- | A: --dB'
    const note = formatNote(cursorInfo.freq)
    const ampLabel = cursorInfo.amp === null ? '--dB' : `${cursorInfo.amp.toFixed(1)}dB`
    return `T: ${cursorInfo.time.toFixed(2)}s | F: ${cursorInfo.freq.toFixed(1)}Hz (${note}) | A: ${ampLabel}`
  }, [cursorInfo, analysis.preview])

  const timeScaleLabel = useMemo(() => {
    if (!analysis.preview) return 'Time -- px/ms'
    const value = viewport.zoomX / 1000
    if (value >= 1) return `Time ${value.toFixed(2)} px/ms`
    if (value >= 0.1) return `Time ${value.toFixed(3)} px/ms`
    return `Time ${value.toFixed(4)} px/ms`
  }, [analysis.preview, viewport.zoomX])

  const audioInfoLabel = useMemo(() => {
    if (!analysis.audioInfo) return 'No audio loaded'
    const base = `${analysis.audioInfo.name} — ${analysis.audioInfo.sample_rate} Hz | ${analysis.audioInfo.channels} ch | ${formatDuration(
      analysis.audioInfo.duration_sec
    )}`
    const extras: string[] = []
    if (analysis.audioInfo.truncated) extras.push('Preview 30s')
    if (analysis.analysisError) extras.push(analysis.analysisError)
    if (extras.length === 0) return base
    return `${base} | ${extras.join(' | ')}`
  }, [analysis.audioInfo, analysis.analysisError])

  return (
    <div className={`page ${ready ? 'is-ready' : ''} h-screen`}>
      <div className="mx-auto flex h-full w-full max-w-none flex-col gap-4 px-4 pb-6 pt-6 sm:px-6">
        <HeaderToolbar
          menuOpen={menuOpen}
          activeTool={activeTool}
          isPlaying={playback.isPlaying}
          isLooping={playback.isLooping}
          mixValue={playback.mixValue}
          playbackTimeLabel={formatDuration(playback.playbackPosition)}
          onMenuToggle={() => setMenuOpen((prev) => !prev)}
          onMenuAction={handleMenuAction}
          onToolChange={setActiveTool}
          onStop={playback.stop}
          onPlayToggle={handlePlayToggle}
          onLoopToggle={playback.toggleLoop}
          onMixChange={playback.setMixValue}
          menuRef={menuRef}
          playDisabled={analysis.analysisState === 'analyzing'}
        />

        <main className="flex h-full flex-1 min-h-0 flex-col gap-4">
          <section className="panel flex flex-1 min-h-0 flex-col rounded-none px-4 py-4">
            <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
              <span className="flex min-w-0 items-center gap-3">
                <span>Workspace</span>
                <span className="min-w-0 truncate font-mono text-[10px] normal-case tracking-normal text-[var(--muted)]">
                  {audioInfoLabel}
                </span>
              </span>
              <span className="font-mono text-[10px]">{timeScaleLabel}</span>
            </div>
            <div className="mt-3 flex-1 min-h-0 h-full">
              <Workspace
                preview={analysis.preview}
                viewportPreviews={viewport.viewportPreviews}
                settings={analysis.settings}
                partials={partialsHook.partials}
                selectedIds={partialsHook.selection}
                selectedInfo={selectedInfo}
                activeTool={activeTool}
                analysisState={analysis.analysisState}
                isSnapping={partialsHook.isSnapping}
                zoomX={viewport.zoomX}
                zoomY={viewport.zoomY}
                pan={viewport.pan}
                playbackPosition={playback.playbackPosition}
                onZoomXChange={viewport.setZoomX}
                onPanChange={viewport.setPan}
                onStageSizeChange={viewport.setStageSize}
                onTraceCommit={handleTraceCommit}
                onEraseCommit={handleEraseCommit}
                onSelectBoxCommit={partialsHook.selectInBox}
                onHitTestCommit={partialsHook.hitTest}
                onUpdatePartial={partialsHook.updatePartial}
                onConnectPick={partialsHook.connectPick}
                onOpenAudio={handleOpenAudio}
                onCursorMove={setCursorInfo}
                onPartialMute={handlePartialMute}
                onPartialDelete={handlePartialDelete}
                onZoomInY={viewport.zoomInY}
                onZoomOutY={viewport.zoomOutY}
              />
            </div>
          </section>
        </main>

        <StatusBar statusLabel={statusLabel} cursorLabel={cursorLabel} statusNote={statusNote} apiBadge={apiBadge} />
      </div>

      {showAnalysisModal ? (
        <AnalysisSettingsModal
          settings={analysis.settings}
          onChange={analysis.setSettings}
          onCancel={() => setShowAnalysisModal(false)}
          onApply={handleApplySettings}
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
