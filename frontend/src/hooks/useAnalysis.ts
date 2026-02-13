import { useState, useCallback } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import { DEFAULT_SETTINGS } from '../app/constants'
import { toPartial } from '../app/utils'
import type { AnalysisSettings, AudioInfo, SpectrogramPreview, Partial, PlaybackSettings, RecentProjectEntry } from '../app/types'

type ReportError = (context: string, message: string) => void

type AnalysisResult = {
  audio: AudioInfo
  preview: SpectrogramPreview | null
  settings: AnalysisSettings
  playbackSettings: PlaybackSettings
  partials: Partial[]
}

export function useAnalysis(reportError: ReportError) {
  const [analysisState, setAnalysisState] = useState<'idle' | 'analyzing' | 'error'>('idle')
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [audioInfo, setAudioInfo] = useState<AudioInfo | null>(null)
  const [preview, setPreview] = useState<SpectrogramPreview | null>(null)
  const [settings, setSettings] = useState<AnalysisSettings>(DEFAULT_SETTINGS)

  const flushUi = useCallback(() => new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve())), [])

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
      if (result.status === 'ok') {
        setAudioInfo(result.audio)
        setPreview(result.preview ?? null)
        setSettings(result.settings)
        setAnalysisState('idle')
        return {
          audio: result.audio,
          preview: result.preview ?? null,
          settings: result.settings,
          playbackSettings: result.playback_settings,
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

  const openAudioPath = useCallback(
    async (path: string): Promise<AnalysisResult | null> => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.open_audio_path) {
        setAnalysisState('error')
        setAnalysisError('Pywebview API is not available.')
        reportError('Open Audio', 'API not available')
        return null
      }
      setAnalysisState('analyzing')
      setAnalysisError(null)
      await flushUi()
      try {
        const result = await api.open_audio_path({ path })
        if (result.status === 'ok') {
          setAudioInfo(result.audio)
          setPreview(result.preview ?? null)
          setSettings(result.settings)
          setAnalysisState('idle')
          return {
            audio: result.audio,
            preview: result.preview ?? null,
            settings: result.settings,
            playbackSettings: result.playback_settings,
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
    },
    [reportError, flushUi]
  )

  const openAudioFile = useCallback(
    async (file: File): Promise<AnalysisResult | null> => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.open_audio_data) {
        setAnalysisState('error')
        setAnalysisError('Pywebview API is not available.')
        reportError('Open Audio', 'API not available')
        return null
      }
      setAnalysisState('analyzing')
      setAnalysisError(null)
      await flushUi()
      try {
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader()
          reader.onload = () => resolve(String(reader.result))
          reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'))
          reader.readAsDataURL(file)
        })
        const base64 = dataUrl.split(',')[1]
        if (!base64) {
          setAnalysisState('error')
          setAnalysisError('Failed to read audio data.')
          reportError('Open Audio', 'Failed to read audio data.')
          return null
        }
        const result = await api.open_audio_data({ name: file.name, data_base64: base64 })
        if (result.status === 'ok') {
          setAudioInfo(result.audio)
          setPreview(result.preview ?? null)
          setSettings(result.settings)
          setAnalysisState('idle')
          return {
            audio: result.audio,
            preview: result.preview ?? null,
            settings: result.settings,
            playbackSettings: result.playback_settings,
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
    },
    [reportError, flushUi]
  )

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
      if (result?.status === 'ok') {
        setAudioInfo(result.audio)
        setPreview(result.preview ?? null)
        setSettings(result.settings)
        setAnalysisState('idle')
        return {
          audio: result.audio,
          preview: result.preview ?? null,
          settings: result.settings,
          playbackSettings: result.playback_settings,
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

  const listRecentProjects = useCallback(async (): Promise<RecentProjectEntry[]> => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.list_recent_projects) {
      reportError('Recent Projects', 'API not available')
      return []
    }
    try {
      const result = await api.list_recent_projects()
      if (result?.status === 'ok') {
        return result.projects
      }
      if (result?.status === 'error') {
        reportError('Recent Projects', result.message ?? 'Failed to load recent projects.')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Recent Projects', message)
    }
    return []
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

  const openProjectPath = useCallback(
    async (path: string): Promise<AnalysisResult | null> => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.open_project_path) {
        reportError('Open Project', 'API not available')
        return null
      }
      setAnalysisState('analyzing')
      setAnalysisError(null)
      await flushUi()
      try {
        const result = await api.open_project_path({ path })
        if (result?.status === 'ok') {
          setAudioInfo(result.audio)
          setPreview(result.preview ?? null)
          setSettings(result.settings)
          setAnalysisState('idle')
          return {
            audio: result.audio,
            preview: result.preview ?? null,
            settings: result.settings,
            playbackSettings: result.playback_settings,
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
    },
    [reportError, flushUi]
  )

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
        if (result.status === 'ok') {
          setSettings(result.settings)
          setPreview(result.preview ?? null)
          setAnalysisState('idle')
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
    openAudioPath,
    openAudioFile,
    openProject,
    openProjectPath,
    newProject,
    listRecentProjects,
    saveProject,
    saveProjectAs,
    applySettings,
  }
}
