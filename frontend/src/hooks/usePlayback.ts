import { useState, useEffect, useCallback } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'

type ReportError = (context: string, message: string) => void

export function usePlayback(reportError: ReportError, analysisState: 'idle' | 'analyzing' | 'error') {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLooping, setIsLooping] = useState(false)
  const [mixValue, setMixValue] = useState(55)
  const [playbackPosition, setPlaybackPosition] = useState(0)

  // Polling playback position
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

  const play = useCallback(async () => {
    if (analysisState === 'analyzing') return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.play) {
      reportError('Playback', 'API not available')
      return
    }
    try {
      const result = await api.play({ mix_ratio: mixValue / 100, loop: isLooping })
      if (result.status === 'ok') {
        setIsPlaying(true)
      } else if (result.status === 'error') {
        reportError('Playback', result.message ?? 'Failed to play')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Playback', message)
    }
  }, [analysisState, mixValue, isLooping, reportError])

  const pause = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.pause) {
      reportError('Playback', 'API not available')
      return
    }
    try {
      const result = await api.pause()
      if (result.status === 'ok') {
        setIsPlaying(false)
      } else if (result.status === 'error') {
        reportError('Playback', result.message ?? 'Failed to pause')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Playback', message)
    }
  }, [reportError])

  const togglePlay = useCallback(async () => {
    if (isPlaying) {
      await pause()
    } else {
      await play()
    }
  }, [isPlaying, play, pause])

  const stop = useCallback(async () => {
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
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Stop', message)
    }
  }, [reportError])

  const toggleLoop = useCallback(() => {
    setIsLooping((prev) => !prev)
  }, [])

  return {
    isPlaying,
    isLooping,
    mixValue,
    playbackPosition,
    setMixValue,
    togglePlay,
    stop,
    toggleLoop,
  }
}
