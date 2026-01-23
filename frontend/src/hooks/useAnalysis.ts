import { useState, useEffect, useCallback, useRef } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import { DEFAULT_SETTINGS } from '../app/constants'
import { toPartial } from '../app/utils'
import type { AnalysisSettings, AudioInfo, SpectrogramPreview, Partial } from '../app/types'

type ReportError = (context: string, message: string) => void

type AnalysisResult = {
  audio: AudioInfo
  preview: SpectrogramPreview | null
  settings: AnalysisSettings
  partials: Partial[]
}

export function useAnalysis(reportError: ReportError) {
  const [analysisState, setAnalysisState] = useState<'idle' | 'analyzing' | 'error'>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [audioInfo, setAudioInfo] = useState<AudioInfo | null>(null)
  const [preview, setPreview] = useState<SpectrogramPreview | null>(null)
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULT_SETTINGS)

  const flushUi = useCallback(() => new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve())), [])
  const previewThrottleRef = useRef<{
    lastApplied: number
    timerId: number | null
    pending: SpectrogramPreview | null
  }>({ lastApplied: 0, timerId: null, pending: null })

  // Subscribe preview events (push)
  useEffect(() => {
    const cleanupState = previewThrottleRef.current
    const applyPreview = (next: SpectrogramPreview) => {
      setPreview(next)
      setAnalysisState('idle')
      setAnalysisError(null)
    }

    const schedulePreview = (next: SpectrogramPreview) => {
      const state = previewThrottleRef.current
      state.pending = next
      const now = Date.now()
      const throttleMs = 1500
      const elapsed = now - state.lastApplied

      if (elapsed >= throttleMs) {
        if (state.timerId) {
          window.clearTimeout(state.timerId)
          state.timerId = null
        }
        state.lastApplied = now
        applyPreview(next)
        return
      }

      if (state.timerId) {
        return
      }

      state.timerId = window.setTimeout(() => {
        state.timerId = null
        if (state.pending) {
          state.lastApplied = Date.now()
          applyPreview(state.pending)
        }
      }, throttleMs - elapsed)
    }

    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail as unknown
      if (!detail || typeof detail !== 'object') return
      const payload = detail as Record<string, unknown>
      if (payload.type === 'spectrogram_preview_updated' && payload.kind === 'overview') {
        const next = payload.preview as SpectrogramPreview | undefined
        if (!next) return
        schedulePreview(next)
      }
      if (payload.type === 'spectrogram_preview_error' && payload.kind === 'overview') {
        const message = typeof payload.message === 'string' ? payload.message : 'Preview failed.'
        setAnalysisState('error')
        setAnalysisError(message)
        reportError('Preview', message)
      }
    }
    window.addEventListener('soma:event', handler)
    return () => {
      window.removeEventListener('soma:event', handler)
      const state = cleanupState
      if (state.timerId) {
        window.clearTimeout(state.timerId)
        state.timerId = null
      }
    }
  }, [reportError])

  const openAudio = useCallback(async (): Promise<AnalysisResult | null> => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.open_audio) {
      setAnalysisState('error')
      setAnalysisError('Pywebview API is not available.')
      reportError('Open Audio', 'API not available')
      return null
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
        setAnalysisState(result.status === 'processing' ? 'analyzing' : 'idle')
        return {
          audio: result.audio,
          preview: result.preview ?? null,
          settings: result.settings,
          partials: result.partials.map(toPartial),
        }
      } else if (result.status === 'cancelled') {
        setAnalysisState('idle')
        return null
      } else if (result.status === 'error') {
        setAnalysisState('error')
        setAnalysisError(result.message ?? 'Failed to load audio.')
        reportError('Open Audio', result.message ?? 'Failed to load audio.')
        return null
      } else {
        setAnalysisState('error')
        setAnalysisError('Unexpected response from API.')
        reportError('Open Audio', 'Unexpected response from API.')
        return null
      }
    } catch (error) {
      setAnalysisState('error')
      const message = error instanceof Error ? error.message : 'Failed to load audio.'
      setAnalysisError(message)
      reportError('Open Audio', message)
      return null
    }
  }, [reportError, flushUi])

  const openProject = useCallback(async (): Promise<AnalysisResult | null> => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.open_project) {
      reportError('Open Project', 'API not available')
      return null
    }
    setAnalysisState('analyzing')
    setAnalysisError(null)
    await flushUi()
    try {
      const result = await api.open_project()
      if (result?.status === 'ok' || result?.status === 'processing') {
        setAudioInfo(result.audio)
        setPreview(result.preview ?? null)
        setSettings(result.settings)
        if (result.status === 'processing') {
          setAnalysisState('analyzing')
        } else {
          setAnalysisState('idle')
        }
        return {
          audio: result.audio,
          preview: result.preview ?? null,
          settings: result.settings,
          partials: result.partials.map(toPartial),
        }
      } else if (result?.status === 'cancelled') {
        setAnalysisState('idle')
        return null
      } else if (result?.status === 'error') {
        setAnalysisState('error')
        setAnalysisError(result.message ?? 'Failed to open project.')
        reportError('Open Project', result.message ?? 'Failed to open project.')
        return null
      }
      return null
    } catch (error) {
      setAnalysisState('error')
      const message = error instanceof Error ? error.message : 'Failed to open project.'
      setAnalysisError(message)
      reportError('Open Project', message)
      return null
    }
  }, [reportError, flushUi])

  const newProject = useCallback(async (): Promise<boolean> => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.new_project) {
      reportError('New Project', 'API not available')
      return false
    }
    try {
      const result = await api.new_project()
      if (result?.status === 'ok') {
        setAudioInfo(null)
        setPreview(null)
        setAnalysisState('idle')
        setAnalysisError(null)
        return true
      } else if (result?.status === 'error') {
        reportError('New Project', result.message ?? 'Failed to create project')
        return false
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('New Project', message)
    }
    return false
  }, [reportError])

  const saveProject = useCallback(async (): Promise<boolean> => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.save_project) {
      reportError('Save Project', 'API not available')
      return false
    }
    try {
      const result = await api.save_project()
      if (result?.status === 'ok') {
        return true
      } else if (result?.status === 'error') {
        reportError('Save Project', result.message ?? 'Failed to save project.')
        return false
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Save Project', message)
    }
    return false
  }, [reportError])

  const saveProjectAs = useCallback(async (): Promise<boolean> => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.save_project_as) {
      reportError('Save Project', 'API not available')
      return false
    }
    try {
      const result = await api.save_project_as()
      if (result?.status === 'ok') {
        return true
      } else if (result?.status === 'error') {
        reportError('Save Project', result.message ?? 'Failed to save project.')
        return false
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Save Project', message)
    }
    return false
  }, [reportError])

  const applySettings = useCallback(
    async (newSettings: AnalysisSettings): Promise<boolean> => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.update_settings) {
        reportError('Settings', 'API not available')
        return false
      }
      setAnalysisState('analyzing')
      try {
        const result = await api.update_settings(newSettings)
        if (result.status === 'ok' || result.status === 'processing') {
          setSettings(result.settings)
          setPreview(result.preview ?? null)
          setAnalysisState(result.status === 'processing' ? 'analyzing' : 'idle')
          return true
        } else if (result.status === 'error') {
          reportError('Settings', result.message ?? 'Failed to apply settings')
          setAnalysisState('idle')
          return false
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Settings', message)
        setAnalysisState('idle')
      }
      return false
    },
    [reportError]
  )

  return {
    analysisState,
    analysisError,
    audioInfo,
    preview,
    settings,
    setSettings,
    openAudio,
    openProject,
    newProject,
    saveProject,
    saveProjectAs,
    applySettings,
  }
}
