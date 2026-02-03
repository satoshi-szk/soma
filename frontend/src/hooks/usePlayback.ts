import { useState, useEffect, useCallback, useRef } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'

type ReportError = (context: string, message: string) => void

export function usePlayback(reportError: ReportError, analysisState: 'idle' | 'analyzing' | 'error') {
  const [isPlaying, setIsPlaying] = useState(false)
  const [mixValue, setMixValue] = useState(55)
  const [playbackPosition, setPlaybackPosition] = useState(0)
  const playbackStartRef = useRef(0)

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
    if (isPlaying) return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.play) {
      reportError('Playback', 'API not available')
      return
    }
    try {
      const startPosition = playbackPosition
      const result = await api.play({
        mix_ratio: mixValue / 100,
        loop: false,
        start_position_sec: startPosition,
      })
      if (result.status === 'ok') {
        playbackStartRef.current = startPosition
        setIsPlaying(true)
      } else if (result.status === 'error') {
        reportError('Playback', result.message ?? 'Failed to play')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Playback', message)
    }
  }, [analysisState, isPlaying, mixValue, playbackPosition, reportError])

  const stop = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.stop) {
      reportError('Stop', 'API not available')
      return
    }
    try {
      const returnPosition = playbackStartRef.current
      const result = await api.stop({ return_position_sec: returnPosition })
      if (result.status === 'ok') {
        setIsPlaying(false)
        setPlaybackPosition(returnPosition)
      } else if (result.status === 'error') {
        reportError('Stop', result.message ?? 'Failed to stop')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Stop', message)
    }
  }, [reportError])

  const setPlayheadPosition = useCallback(
    (positionSec: number) => {
      if (isPlaying) return
      setPlaybackPosition(Math.max(0, positionSec))
    },
    [isPlaying],
  )

  const togglePlayStop = useCallback(async () => {
    if (isPlaying) {
      await stop()
      return
    }
    await play()
  }, [isPlaying, play, stop])

  return {
    isPlaying,
    mixValue,
    playbackPosition,
    setMixValue,
    play,
    stop,
    togglePlayStop,
    setPlayheadPosition,
  }
}
