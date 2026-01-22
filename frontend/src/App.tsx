import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnalysisSettingsModal } from './components/modals/AnalysisSettingsModal'
import { ExportModal } from './components/modals/ExportModal'
import { HeaderToolbar } from './components/HeaderToolbar'
import { Workspace } from './components/Workspace'
import { AudioInfoPanel } from './components/AudioInfoPanel'
import { StatusBar } from './components/StatusBar'
import { DEFAULT_SETTINGS } from './app/constants'
import type { AnalysisSettings, AudioInfo, Partial, PartialPoint, SpectrogramPreview, ToolId, ViewportPreview } from './app/types'
import { formatDuration, formatNote, toPartial } from './app/utils'
import { isPywebviewApiAvailable, pywebviewApi } from './app/pywebviewApi'
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
  const [isSnapping, setIsSnapping] = useState(false)
  const [mixValue, setMixValue] = useState(55)
  const [statusNote, setStatusNote] = useState<string | null>(null)
  const [playbackPosition, setPlaybackPosition] = useState(0)
  const [cursorInfo, setCursorInfo] = useState<{ time: number; freq: number; amp: number | null }>({
    time: 0,
    freq: 440,
    amp: null,
  })
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
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [viewportPreview, setViewportPreview] = useState<ViewportPreview | null>(null)
  const [viewportRequestId, setViewportRequestId] = useState<string | null>(null)
  const viewportDebounceRef = useRef<number | null>(null)
  const connectQueueRef = useRef<string[]>([])

  const menuRef = useRef<HTMLDivElement | null>(null)

  const reportError = useCallback((context: string, message: string) => {
    const detail = `${context}: ${message}`
    setStatusNote(detail)
    console.error(detail)
  }, [])

  const reportException = useCallback((context: string, error: unknown) => {
    const message = error instanceof Error ? error.message : 'Unexpected error'
    const detail = `${context}: ${message}`
    setStatusNote(detail)
    console.error(detail, error)
  }, [])

  const flushUi = useCallback(() => new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve())), [])

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
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api) return
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
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api) return
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

  // Viewport preview: debounced request on zoom/pan change
  useEffect(() => {
    if (!preview || zoom <= 1) {
      setViewportPreview(null)
      return
    }

    if (viewportDebounceRef.current) {
      window.clearTimeout(viewportDebounceRef.current)
    }

    viewportDebounceRef.current = window.setTimeout(async () => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api) return

      const duration = preview.duration_sec
      const freqMin = preview.freq_min
      const freqMax = preview.freq_max

      // Calculate visible viewport
      const visibleTimeStart = Math.max(0, (-pan.x / (stageSize.width * zoom)) * duration)
      const visibleTimeEnd = Math.min(duration, visibleTimeStart + (duration / zoom))
      const logMin = Math.log(freqMin)
      const logMax = Math.log(freqMax)
      const visibleFreqMax = Math.exp(logMax - Math.max(0, -pan.y / (stageSize.height * zoom)) * (logMax - logMin))
      const visibleFreqMin = Math.exp(logMax - Math.min(1, (stageSize.height - pan.y) / (stageSize.height * zoom)) * (logMax - logMin))

      const result = await api.request_viewport_preview({
        time_start: visibleTimeStart,
        time_end: visibleTimeEnd,
        freq_min: Math.max(freqMin, visibleFreqMin),
        freq_max: Math.min(freqMax, visibleFreqMax),
        width: Math.round(stageSize.width),
        height: Math.round(stageSize.height),
      })

      if (result.status === 'processing') {
        setViewportRequestId(result.request_id)
      }
    }, 500)

    return () => {
      if (viewportDebounceRef.current) {
        window.clearTimeout(viewportDebounceRef.current)
      }
    }
  }, [zoom, pan, stageSize, preview])

  // Viewport preview: polling
  useEffect(() => {
    if (!viewportRequestId) return
    let alive = true

    const poll = async () => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api || !alive) return
      const result = await api.viewport_preview_status()
      if (!alive || result?.status !== 'ok') return
      if (result.request_id !== viewportRequestId) return

      if (result.state === 'ready' && result.preview) {
        setViewportPreview(result.preview)
        setViewportRequestId(null)
      } else if (result.state === 'error' || result.state === 'cancelled') {
        setViewportRequestId(null)
      }
    }

    void poll()
    const interval = window.setInterval(poll, 200)
    return () => {
      alive = false
      window.clearInterval(interval)
    }
  }, [viewportRequestId])

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
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api) {
      reportError(label, 'API not available')
      return
    }
    if (label === 'New Project') {
      try {
        const result = await api.new_project()
        if (result?.status === 'ok') {
          setAudioInfo(null)
          setPreview(null)
          setPartials([])
          setSelection([])
          setStatusNote('New project ready')
        } else if (result?.status === 'error') {
          reportError('New Project', result.message ?? 'Failed to create project')
        }
      } catch (error) {
        reportException('New Project', error)
      }
      return
    }
    if (label === 'Open Project...') {
      try {
        setAnalysisState('analyzing')
        setAnalysisError(null)
        await flushUi()
        const result = await api.open_project()
        if (result?.status === 'ok' || result?.status === 'processing') {
          setAudioInfo(result.audio)
          setPreview(result.preview ?? null)
          setSettings(result.settings)
          setPartials(result.partials.map(toPartial))
          if (result.status === 'processing') {
            setAnalysisState('analyzing')
          }
          setStatusNote('Project opened')
        } else if (result?.status === 'cancelled') {
          setAnalysisState('idle')
        } else if (result?.status === 'error') {
          setAnalysisState('error')
          setAnalysisError(result.message ?? 'Failed to open project.')
          reportError('Open Project', result.message ?? 'Failed to open project.')
        }
      } catch (error) {
        setAnalysisState('error')
        const message = error instanceof Error ? error.message : 'Failed to open project.'
        setAnalysisError(message)
        reportException('Open Project', error)
      }
      return
    }
    if (label === 'Open Audio...') {
      await openAudio()
      return
    }
    if (label === 'Save Project') {
      try {
        const result = await api.save_project()
        if (result?.status === 'ok') {
          setStatusNote('Project saved')
        } else if (result?.status === 'error') {
          reportError('Save Project', result.message ?? 'Failed to save project.')
        }
      } catch (error) {
        reportException('Save Project', error)
      }
      return
    }
    if (label === 'Save As...') {
      try {
        const result = await api.save_project_as()
        if (result?.status === 'ok') {
          setStatusNote('Project saved')
        } else if (result?.status === 'error') {
          reportError('Save Project', result.message ?? 'Failed to save project.')
        }
      } catch (error) {
        reportException('Save Project', error)
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
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.open_audio) {
      setAnalysisState('error')
      setAnalysisError('Pywebview API is not available.')
      reportError('Open Audio', 'API not available')
      return
    }
    setAnalysisState('analyzing')
    setAnalysisError(null)
    await flushUi()
    try {
      const result = await api.open_audio()
      if (result.status === 'ok' || result.status === 'processing') {
        setAudioInfo(result.audio)
        setPreview(result.preview ?? null)
        setSettings(result.settings)
        setPartials(result.partials.map(toPartial))
        setAnalysisState(result.status === 'processing' ? 'analyzing' : 'idle')
        setStatusNote('Audio loaded')
      } else if (result.status === 'cancelled') {
        setAnalysisState('idle')
      } else if (result.status === 'error') {
        setAnalysisState('error')
        setAnalysisError(result.message ?? 'Failed to load audio.')
        reportError('Open Audio', result.message ?? 'Failed to load audio.')
      } else {
        setAnalysisState('error')
        setAnalysisError('Unexpected response from API.')
        reportError('Open Audio', 'Unexpected response from API.')
      }
    } catch (error) {
      setAnalysisState('error')
      const message = error instanceof Error ? error.message : 'Failed to load audio.'
      setAnalysisError(message)
      reportException('Open Audio', error)
    }
  }

  const applySettings = async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.update_settings) {
      reportError('Settings', 'API not available')
      return
    }
    setAnalysisState('analyzing')
    try {
      const result = await api.update_settings(settings)
      if (result.status === 'ok' || result.status === 'processing') {
        setSettings(result.settings)
        setPreview(result.preview ?? null)
        setAnalysisState(result.status === 'processing' ? 'analyzing' : 'idle')
        setStatusNote('Settings applied')
      } else if (result.status === 'error') {
        reportError('Settings', result.message ?? 'Failed to apply settings')
      }
    } catch (error) {
      reportException('Settings', error)
    }
    setShowAnalysisModal(false)
  }

  const handleTraceCommit = async (trace: Array<[number, number]>) => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.trace_partial) {
      reportError('Trace', 'API not available')
      return false
    }
    setIsSnapping(true)
    try {
      const result = await api.trace_partial({ trace })
      if (result.status === 'ok') {
        setPartials((prev) => [...prev, toPartial(result.partial)])
        setStatusNote('Trace snapped')
        return true
      }
      if (result.status === 'error') {
        reportError('Trace', result.message ?? 'Failed to create partial')
      }
    } catch (error) {
      reportException('Trace', error)
    } finally {
      setIsSnapping(false)
    }
    return false
  }

  const handleEraseCommit = async (trace: Array<[number, number]>) => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.erase_partial) {
      reportError('Erase', 'API not available')
      return
    }
    try {
      const result = await api.erase_partial({ trace, radius_hz: 60 })
      if (result.status === 'ok') {
        setPartials(result.partials.map(toPartial))
        setSelection([])
        setStatusNote('Erase applied')
      } else if (result.status === 'error') {
        reportError('Erase', result.message ?? 'Failed to erase')
      }
    } catch (error) {
      reportException('Erase', error)
    }
  }

  const handleSelectBoxCommit = async (box: {
    time_start: number
    time_end: number
    freq_start: number
    freq_end: number
  }) => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.select_in_box) {
      reportError('Select', 'API not available')
      return
    }
    try {
      const result = await api.select_in_box(box)
      if (result.status === 'ok') {
        updateSelection(result.ids)
      } else if (result.status === 'error') {
        reportError('Select', result.message ?? 'Selection failed')
      }
    } catch (error) {
      reportException('Select', error)
    }
  }

  const handleHitTestCommit = async (point: { time: number; freq: number }) => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.hit_test) {
      reportError('HitTest', 'API not available')
      return
    }
    try {
      const result = await api.hit_test({ ...point, tolerance: 0.08 })
      if (result.status === 'ok') {
        updateSelection(result.id ? [result.id] : [])
      } else if (result.status === 'error') {
        reportError('HitTest', result.message ?? 'Hit test failed')
      }
    } catch (error) {
      reportException('HitTest', error)
    }
  }

  const handleUpdatePartial = async (id: string, points: PartialPoint[]) => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.update_partial) {
      reportError('Update', 'API not available')
      return
    }
    try {
      const payload = {
        id,
        points: points.map((point) => [point.time, point.freq, point.amp] as [number, number, number]),
      }
      const result = await api.update_partial(payload)
      if (result.status === 'ok') {
        setPartials((prev) => prev.map((item) => (item.id === result.partial.id ? toPartial(result.partial) : item)))
        setStatusNote('Partial updated')
      } else if (result.status === 'error') {
        reportError('Update', result.message ?? 'Update failed')
      }
    } catch (error) {
      reportException('Update', error)
    }
  }

  const handleConnectPick = async (point: { time: number; freq: number }) => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.hit_test) {
      reportError('Connect', 'API not available')
      return
    }
    try {
      const result = await api.hit_test({ ...point, tolerance: 0.08 })
      if (result.status === 'ok' && result.id) {
        const current = connectQueueRef.current
        const next = current.includes(result.id) ? current : [...current, result.id]
        if (next.length === 2 && api?.merge_partials) {
          const mergeResult = await api.merge_partials({ first: next[0], second: next[1] })
          if (mergeResult.status === 'ok') {
            setPartials((items) => [...items.filter((p) => !next.includes(p.id)), toPartial(mergeResult.partial)])
            setSelection([mergeResult.partial.id])
            setStatusNote('Partials merged')
          } else if (mergeResult.status === 'error') {
            reportError('Connect', mergeResult.message ?? 'Merge failed')
          }
          connectQueueRef.current = []
        } else {
          connectQueueRef.current = next
        }
      } else if (result.status === 'error') {
        reportError('Connect', result.message ?? 'Hit test failed')
      }
    } catch (error) {
      reportException('Connect', error)
    }
  }

  const handlePartialMute = async () => {
    if (selection.length !== 1) return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.toggle_mute) {
      reportError('Mute', 'API not available')
      return
    }
    try {
      const result = await api.toggle_mute({ id: selection[0] })
      if (result.status === 'ok') {
        setPartials((prev) => prev.map((item) => (item.id === result.partial.id ? toPartial(result.partial) : item)))
        setStatusNote(result.partial.is_muted ? 'Partial muted' : 'Partial unmuted')
      } else if (result.status === 'error') {
        reportError('Mute', result.message ?? 'Mute failed')
      }
    } catch (error) {
      reportException('Mute', error)
    }
  }

  const handlePartialDelete = async () => {
    if (selection.length === 0) return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.delete_partials) {
      reportError('Delete', 'API not available')
      return
    }
    try {
      const result = await api.delete_partials({ ids: selection })
      if (result.status === 'ok') {
        setPartials((prev) => prev.filter((item) => !selection.includes(item.id)))
        setSelection([])
        setStatusNote('Partial deleted')
      } else if (result.status === 'error') {
        reportError('Delete', result.message ?? 'Delete failed')
      }
    } catch (error) {
      reportException('Delete', error)
    }
  }

  const undo = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.undo) {
      reportError('Undo', 'API not available')
      return
    }
    try {
      const result = await api.undo()
      if (result.status === 'ok') {
        setPartials(result.partials.map(toPartial))
        setStatusNote('Undo')
      } else if (result.status === 'error') {
        reportError('Undo', result.message ?? 'Undo failed')
      }
    } catch (error) {
      reportException('Undo', error)
    }
  }, [reportError, reportException])

  const redo = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.redo) {
      reportError('Redo', 'API not available')
      return
    }
    try {
      const result = await api.redo()
      if (result.status === 'ok') {
        setPartials(result.partials.map(toPartial))
        setStatusNote('Redo')
      } else if (result.status === 'error') {
        reportError('Redo', result.message ?? 'Redo failed')
      }
    } catch (error) {
      reportException('Redo', error)
    }
  }, [reportError, reportException])

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    const key = event.key.toLowerCase()
    if (key === 'v') setActiveTool('select')
    if (key === 'p') setActiveTool('trace')
    if (key === 'e') setActiveTool('erase')
    if (key === 'c') setActiveTool('connect')
    if (key === 'z' && (event.metaKey || event.ctrlKey) && !event.shiftKey) {
      event.preventDefault()
      void undo()
    }
    if (key === 'z' && event.shiftKey && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      void redo()
    }
  }, [undo, redo])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  const handlePlayToggle = async () => {
    if (analysisState === 'analyzing') return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.play || !api?.pause) {
      reportError('Playback', 'API not available')
      return
    }
    try {
      if (isPlaying) {
        const result = await api.pause()
        if (result.status === 'ok') {
          setIsPlaying(false)
        } else if (result.status === 'error') {
          reportError('Playback', result.message ?? 'Failed to pause')
        }
      } else {
        const result = await api.play({ mix_ratio: mixValue / 100, loop: isLooping })
        if (result.status === 'ok') {
          setIsPlaying(true)
        } else if (result.status === 'error') {
          reportError('Playback', result.message ?? 'Failed to play')
        }
      }
    } catch (error) {
      reportException('Playback', error)
    }
  }

  const handleStop = async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.stop) {
      reportError('Stop', 'API not available')
      return
    }
    try {
      const result = await api.stop()
      if (result.status === 'ok') {
        setIsPlaying(false)
        setPlaybackPosition(0)
      } else if (result.status === 'error') {
        reportError('Stop', result.message ?? 'Failed to stop')
      }
    } catch (error) {
      reportException('Stop', error)
    }
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
      reportException('Export', error)
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

  const cursorLabel = useMemo(() => {
    if (!preview) return 'T: -- | F: -- | A: --dB'
    const note = formatNote(cursorInfo.freq)
    const ampLabel = cursorInfo.amp === null ? '--dB' : `${cursorInfo.amp.toFixed(1)}dB`
    return `T: ${cursorInfo.time.toFixed(2)}s | F: ${cursorInfo.freq.toFixed(1)}Hz (${note}) | A: ${ampLabel}`
  }, [cursorInfo, preview])

  return (
    <div className={`page ${ready ? 'is-ready' : ''} h-screen`}>
      <div className="mx-auto flex h-full w-full max-w-none flex-col gap-4 px-4 pb-6 pt-6 sm:px-6">
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
          playDisabled={analysisState === 'analyzing'}
        />

        <main className="flex h-full flex-1 min-h-0 flex-col gap-4">
          <section className="panel flex flex-1 min-h-0 flex-col rounded-none px-4 py-4">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
              <span>Workspace</span>
              <span className="font-mono text-[10px]">Zoom {Math.round(zoom * 100)}%</span>
            </div>
            <div className="mt-3 flex-1 min-h-0 h-full">
              <Workspace
                preview={preview}
                viewportPreview={viewportPreview}
                settings={settings}
                partials={partials}
                selectedIds={selection}
                selectedInfo={selectedInfo}
                activeTool={activeTool}
                analysisState={analysisState}
                isSnapping={isSnapping}
                zoom={zoom}
                pan={pan}
                playbackPosition={playbackPosition}
                onZoomChange={setZoom}
                onPanChange={setPan}
                onStageSizeChange={setStageSize}
                onTraceCommit={handleTraceCommit}
                onEraseCommit={handleEraseCommit}
                onSelectBoxCommit={handleSelectBoxCommit}
                onHitTestCommit={handleHitTestCommit}
                onUpdatePartial={handleUpdatePartial}
                onConnectPick={handleConnectPick}
                onOpenAudio={openAudio}
                onCursorMove={setCursorInfo}
                onPartialMute={handlePartialMute}
                onPartialDelete={handlePartialDelete}
              />
            </div>
          </section>

          <section>
            <AudioInfoPanel audioInfo={audioInfo} analysisError={analysisError} statusNote={statusNote} />
          </section>
        </main>

        <StatusBar statusLabel={statusLabel} cursorLabel={cursorLabel} statusNote={statusNote} />
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
