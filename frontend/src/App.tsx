import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnalysisSettingsModal } from './components/modals/AnalysisSettingsModal'
import { ExportModal } from './components/modals/ExportModal'
import { PlaybackSettingsSidebar } from './components/modals/PlaybackSettingsSidebar'
import { HeaderToolbar } from './components/HeaderToolbar'
import { Workspace } from './components/Workspace'
import { StatusBar } from './components/StatusBar'
import type { AnalysisSettings, ToolId } from './app/types'
import { formatDuration, formatNoteWithCents } from './app/utils'
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
  const [spectrogramDim, setSpectrogramDim] = useState(0)
  const [cursorInfo, setCursorInfo] = useState<{ time: number; freq: number; amp: number | null }>({
    time: 0,
    freq: 440,
    amp: null,
  })
  const [showAnalysisModal, setShowAnalysisModal] = useState(false)
  const [showExportModal, setShowExportModal] = useState(false)
  const [showPlaybackSettings, setShowPlaybackSettings] = useState(false)
  const [exportTab, setExportTab] = useState<'mpe' | 'multitrack' | 'mono' | 'audio' | 'cv'>('mpe')
  const [mpePitchBendRange, setMpePitchBendRange] = useState('48')
  const [multitrackPitchBendRange, setMultitrackPitchBendRange] = useState('48')
  const [monoPitchBendRange, setMonoPitchBendRange] = useState('48')
  const [mpeAmpMapping, setMpeAmpMapping] = useState('cc74')
  const [multitrackAmpMapping, setMultitrackAmpMapping] = useState('cc74')
  const [monoAmpMapping, setMonoAmpMapping] = useState('cc74')
  const [mpeAmpCurve, setMpeAmpCurve] = useState('linear')
  const [multitrackAmpCurve, setMultitrackAmpCurve] = useState('linear')
  const [monoAmpCurve, setMonoAmpCurve] = useState('linear')
  const [cvAmpCurve, setCvAmpCurve] = useState('linear')
  const [exportSampleRate, setExportSampleRate] = useState('44100')
  const [exportBitDepth, setExportBitDepth] = useState('16')
  const [exportCvBaseFreq, setExportCvBaseFreq] = useState('440')
  const [exportCvFullScaleVolts, setExportCvFullScaleVolts] = useState('10')
  const [exportCvMode, setExportCvMode] = useState<'mono' | 'poly'>('mono')

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
  const viewport = useViewport(analysis.preview, reportError)

  useKeyboardShortcuts({
    onToolChange: setActiveTool,
    onUndo: partialsHook.undo,
    onRedo: partialsHook.redo,
    onPlayToggle: () => {
      if (analysis.analysisState !== 'analyzing' && !playback.isProbePlaying) playback.togglePlayStop()
    },
    onProbeToggle: playback.toggleHarmonicProbe,
    onSave: async () => {
      const success = await analysis.saveProject()
      if (success) setStatusNote('Project saved')
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
        await playback.syncStatus()
        setStatusNote('New project ready')
      }
      return
    }
    if (label === 'Open Project...') {
      const result = await analysis.openProject()
      if (result) {
        partialsHook.setPartials(result.partials)
        playback.applyPlaybackSettings(result.playbackSettings.master_volume)
        setStatusNote('Project opened')
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
      playback.applyPlaybackSettings(result.playbackSettings.master_volume)
      setStatusNote('Audio loaded')
    }
  }

  const handleOpenAudioPath = async (path: string) => {
    setMenuOpen(false)
    setStatusNote(null)
    const result = await analysis.openAudioPath(path)
    if (result) {
      partialsHook.setPartials(result.partials)
      playback.applyPlaybackSettings(result.playbackSettings.master_volume)
      setStatusNote('Audio loaded')
    }
  }

  const handleOpenAudioFile = async (file: File) => {
    setMenuOpen(false)
    setStatusNote(null)
    const result = await analysis.openAudioFile(file)
    if (result) {
      partialsHook.setPartials(result.partials)
      playback.applyPlaybackSettings(result.playbackSettings.master_volume)
      setStatusNote('Audio loaded')
    }
  }

  const handleApplySettings = async (nextSettings?: AnalysisSettings) => {
    const settingsToApply = nextSettings ?? analysis.settings
    const success = await analysis.applySettings(settingsToApply)
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
    const parseOptionalNumber = (value: string): { value: number | undefined; valid: boolean } => {
      const trimmed = value.trim()
      if (trimmed === '') return { value: undefined, valid: true }
      const parsed = Number(trimmed)
      if (!Number.isFinite(parsed)) {
        return { value: undefined, valid: false }
      }
      return { value: parsed, valid: true }
    }
    const currentPitchBendRange =
      exportTab === 'mpe' ? mpePitchBendRange : exportTab === 'multitrack' ? multitrackPitchBendRange : monoPitchBendRange
    const currentAmpMapping =
      exportTab === 'mpe' ? mpeAmpMapping : exportTab === 'multitrack' ? multitrackAmpMapping : monoAmpMapping
    const currentAmpCurve =
      exportTab === 'mpe'
        ? mpeAmpCurve
        : exportTab === 'multitrack'
          ? multitrackAmpCurve
          : exportTab === 'mono'
            ? monoAmpCurve
            : cvAmpCurve
    const pitchBendRange = parseOptionalNumber(currentPitchBendRange)
    const sampleRate = parseOptionalNumber(exportSampleRate)
    const bitDepth = parseOptionalNumber(exportBitDepth)
    const cvBaseFreq = parseOptionalNumber(exportCvBaseFreq)
    const cvFullScaleVolts = parseOptionalNumber(exportCvFullScaleVolts)
    if (exportTab !== 'audio' && !pitchBendRange.valid) {
      reportError('Export', 'Pitch Bend Range is invalid')
      return
    }
    if ((exportTab === 'audio' || exportTab === 'cv') && !sampleRate.valid) {
      reportError('Export', 'Sample Rate is invalid')
      return
    }
    if ((exportTab === 'audio' || exportTab === 'cv') && !bitDepth.valid) {
      reportError('Export', 'Bit Depth is invalid')
      return
    }
    if (exportTab === 'cv') {
      if (!cvBaseFreq.valid || (cvBaseFreq.value !== undefined && cvBaseFreq.value <= 0)) {
        reportError('Export', 'CV Base Frequency is invalid')
        return
      }
      if (!cvFullScaleVolts.valid || (cvFullScaleVolts.value !== undefined && cvFullScaleVolts.value <= 0)) {
        reportError('Export', 'CV Full Scale is invalid')
        return
      }
    }
    try {
      if (exportTab === 'mpe') {
        const result = await api.export_mpe({
          pitch_bend_range: pitchBendRange.value,
          amplitude_mapping: currentAmpMapping,
          amplitude_curve: currentAmpCurve,
        })
        if (result.status === 'ok') {
          setStatusNote(`Exported ${result.paths.length} MIDI file(s).`)
        } else if (result.status === 'error') {
          reportError('Export', result.message ?? 'Failed to export MIDI')
        }
      } else if (exportTab === 'multitrack') {
        const result = await api.export_multitrack_midi({
          pitch_bend_range: pitchBendRange.value,
          amplitude_mapping: currentAmpMapping,
          amplitude_curve: currentAmpCurve,
        })
        if (result.status === 'ok') {
          setStatusNote(`Exported ${result.paths.length} MIDI file(s).`)
        } else if (result.status === 'error') {
          reportError('Export', result.message ?? 'Failed to export MIDI')
        }
      } else if (exportTab === 'mono') {
        const result = await api.export_monophonic_midi({
          pitch_bend_range: pitchBendRange.value,
          amplitude_mapping: currentAmpMapping,
          amplitude_curve: currentAmpCurve,
        })
        if (result.status === 'ok') {
          setStatusNote(`Exported ${result.paths.length} MIDI file(s).`)
        } else if (result.status === 'error') {
          reportError('Export', result.message ?? 'Failed to export MIDI')
        }
      } else {
        const result = await api.export_audio({
          sample_rate: sampleRate.value,
          bit_depth: bitDepth.value,
          output_type: exportTab === 'cv' ? 'cv' : 'sine',
          cv_base_freq: exportTab === 'cv' ? cvBaseFreq.value : undefined,
          cv_full_scale_volts: exportTab === 'cv' ? cvFullScaleVolts.value : undefined,
          cv_mode: exportTab === 'cv' ? exportCvMode : undefined,
          amplitude_curve: exportTab === 'cv' ? currentAmpCurve : undefined,
        })
        if (result.status === 'ok') {
          setStatusNote(
            result.paths && result.paths.length > 1 ? `Exported ${result.paths.length} CV file(s).` : `Exported ${result.path}`
          )
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

  const handleRewind = () => {
    if (playback.isPlaying) return
    void playback.setPlayheadPosition(0)
  }

  const handlePlayStop = () => {
    if (playback.isProbePlaying) return
    if (!playback.isPlaying && analysis.analysisState === 'analyzing') return
    void playback.togglePlayStop()
  }

  const handlePlaybackSettingsToggle = () => {
    setShowPlaybackSettings((prev) => !prev)
  }

  const statusLabel =
    analysis.analysisState === 'analyzing'
      ? 'Analyzing'
      : analysis.analysisState === 'error'
        ? 'Error'
        : playback.isProbePlaying
          ? 'Harmonic Probe'
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
    if (!analysis.preview) return 'T: -- | F: --'
    const note = formatNoteWithCents(cursorInfo.freq)
    return `T: ${cursorInfo.time.toFixed(2)}s | F: ${cursorInfo.freq.toFixed(1)}Hz (${note})`
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
      <div className="relative mx-auto flex h-full w-full max-w-none flex-col gap-2 px-2 pb-2 pt-2">
        <HeaderToolbar
          menuOpen={menuOpen}
          activeTool={activeTool}
          isPlaying={playback.isPlaying}
          isPreparingPlayback={playback.isPreparingPlayback}
          masterVolume={playback.masterVolume}
          playbackTimeLabel={formatDuration(playback.playbackPosition)}
          isProbePlaying={playback.isProbePlaying}
          onMenuToggle={() => setMenuOpen((prev) => !prev)}
          onMenuAction={handleMenuAction}
          onToolChange={setActiveTool}
          onPlayStop={handlePlayStop}
          onProbeToggle={() => void playback.toggleHarmonicProbe()}
          onRewind={handleRewind}
          onMasterVolumeChange={(value) => void playback.setMasterVolume(value)}
          onPlaybackSettingsOpen={handlePlaybackSettingsToggle}
          menuRef={menuRef}
          playDisabled={analysis.analysisState === 'analyzing' || playback.isProbePlaying || playback.isPreparingPlayback}
        />

        <div
          className={`grid h-full flex-1 min-h-0 gap-2 ${
            showPlaybackSettings ? 'grid-cols-[minmax(0,1fr)_320px]' : 'grid-cols-[minmax(0,1fr)]'
          }`}
        >
          <main className="flex flex-1 min-h-0 flex-col gap-2">
            <section className="panel flex flex-1 min-h-0 flex-col rounded-none px-2 py-2">
            <div className="flex items-center justify-between gap-3 text-[11px] tracking-normal text-[var(--ink)]">
              <span className="flex min-w-0 items-center gap-3">
                <span className="font-semibold">Workspace</span>
                <span className="min-w-0 truncate font-mono text-[11px] normal-case tracking-normal text-[var(--muted)]">
                  {audioInfoLabel}
                </span>
              </span>
              <span className="font-mono text-[11px] text-[var(--ink)]">{timeScaleLabel}</span>
            </div>
            <div className="mt-1 flex-1 min-h-0 h-full">
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
                canEditPlayhead={!playback.isPlaying && !playback.isPreparingPlayback}
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
                onOpenAudioPath={handleOpenAudioPath}
                onOpenAudioFile={handleOpenAudioFile}
                onPlayheadChange={playback.setPlayheadPosition}
                allowDrop={!analysis.audioInfo}
                onCursorMove={setCursorInfo}
                onPartialMute={handlePartialMute}
                onPartialDelete={handlePartialDelete}
                onZoomInY={viewport.zoomInY}
                onZoomOutY={viewport.zoomOutY}
                spectrogramDim={spectrogramDim}
              />
            </div>
            </section>
          </main>
          {showPlaybackSettings ? (
            <PlaybackSettingsSidebar
              mixValue={playback.mixValue}
              speedPresetIndex={playback.speedPresetIndex}
              speedValue={playback.speedValue}
              timeStretchMode={playback.timeStretchMode}
              controlsDisabled={playback.isPlaying || playback.isPreparingPlayback}
              onClose={() => setShowPlaybackSettings(false)}
              onMixChange={(value) => void playback.setMixValue(value)}
              onSpeedChange={playback.setSpeedPresetIndex}
              onTimeStretchModeChange={playback.setTimeStretchMode}
            />
          ) : null}
        </div>

        <StatusBar
          statusLabel={statusLabel}
          cursorLabel={cursorLabel}
          statusNote={statusNote}
          apiBadge={apiBadge}
          spectrogramDim={spectrogramDim}
          spectrogramDimEnabled={!!analysis.preview}
          onSpectrogramDimChange={setSpectrogramDim}
        />
      </div>

      {showAnalysisModal ? (
        <AnalysisSettingsModal
          settings={analysis.settings}
          onCancel={() => setShowAnalysisModal(false)}
          onApply={handleApplySettings}
        />
      ) : null}

      {showExportModal ? (
        <ExportModal
          tab={exportTab}
          pitchBendRange={
            exportTab === 'mpe' ? mpePitchBendRange : exportTab === 'multitrack' ? multitrackPitchBendRange : monoPitchBendRange
          }
          amplitudeMapping={exportTab === 'mpe' ? mpeAmpMapping : exportTab === 'multitrack' ? multitrackAmpMapping : monoAmpMapping}
          amplitudeCurve={
            exportTab === 'mpe' ? mpeAmpCurve : exportTab === 'multitrack' ? multitrackAmpCurve : exportTab === 'mono' ? monoAmpCurve : cvAmpCurve
          }
          exportSampleRate={exportSampleRate}
          exportBitDepth={exportBitDepth}
          exportCvBaseFreq={exportCvBaseFreq}
          exportCvFullScaleVolts={exportCvFullScaleVolts}
          exportCvMode={exportCvMode}
          onTabChange={setExportTab}
          onPitchBendChange={(value) => {
            if (exportTab === 'mpe') {
              setMpePitchBendRange(value)
            } else if (exportTab === 'multitrack') {
              setMultitrackPitchBendRange(value)
            } else {
              setMonoPitchBendRange(value)
            }
          }}
          onAmplitudeMappingChange={(value) => {
            if (exportTab === 'mpe') {
              setMpeAmpMapping(value)
            } else if (exportTab === 'multitrack') {
              setMultitrackAmpMapping(value)
            } else {
              setMonoAmpMapping(value)
            }
          }}
          onAmplitudeCurveChange={(value) => {
            if (exportTab === 'mpe') {
              setMpeAmpCurve(value)
            } else if (exportTab === 'multitrack') {
              setMultitrackAmpCurve(value)
            } else if (exportTab === 'mono') {
              setMonoAmpCurve(value)
            } else {
              setCvAmpCurve(value)
            }
          }}
          onSampleRateChange={setExportSampleRate}
          onBitDepthChange={setExportBitDepth}
          onCvBaseFreqChange={setExportCvBaseFreq}
          onCvFullScaleVoltsChange={setExportCvFullScaleVolts}
          onCvModeChange={setExportCvMode}
          onCancel={() => setShowExportModal(false)}
          onExport={handleExport}
        />
      ) : null}
    </div>
  )
}

export default App
