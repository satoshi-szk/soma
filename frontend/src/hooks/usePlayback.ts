import { useState, useEffect, useCallback, useRef } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'

type ReportError = (context: string, message: string) => void
type TimeStretchMode = 'native' | 'librosa'
const SPEED_PRESET_RATIOS = [0.125, 0.25, 0.5, 1, 2, 4, 8] as const
const DEFAULT_SPEED_PRESET_INDEX = 3

export function usePlayback(reportError: ReportError, analysisState: 'idle' | 'analyzing' | 'error') {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isProbePlaying, setIsProbePlaying] = useState(false)
  const [isPreparingPlayback, setIsPreparingPlayback] = useState(false)
  const [mixValue, setMixValue] = useState(55)
  const [masterVolume, setMasterVolumeState] = useState(100)
  const [speedPresetIndex, setSpeedPresetIndex] = useState(DEFAULT_SPEED_PRESET_INDEX)
  const [timeStretchMode, setTimeStretchMode] = useState<TimeStretchMode>('librosa')
  const [playbackPosition, setPlaybackPosition] = useState(0)
  const playbackStartRef = useRef(0)
  const wasNormalPlayingRef = useRef(false)
  const speedValue = SPEED_PRESET_RATIOS[speedPresetIndex]

  const syncStatus = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.status) return
    const result = await api.status()
    if (result?.status !== 'ok') return
    const wasPlaying = wasNormalPlayingRef.current
    const endedNaturally = wasPlaying && !result.is_playing && !result.is_probe_playing && !result.is_preparing_playback
    if (typeof result.position === 'number' && (result.is_playing || result.is_preparing_playback)) {
      setPlaybackPosition(result.position)
    }
    setIsPlaying(result.is_playing)
    setIsProbePlaying(result.is_probe_playing)
    setIsPreparingPlayback(result.is_preparing_playback)
    setMasterVolumeState(Math.round(Math.max(0, Math.min(1, result.master_volume)) * 100))
    if (endedNaturally) {
      setPlaybackPosition(playbackStartRef.current)
    }
    wasNormalPlayingRef.current = result.is_playing
  }, [])

  // 再生位置をポーリングで取得する
  useEffect(() => {
    if (!isPlaying && !isProbePlaying && !isPreparingPlayback) return
    const interval = window.setInterval(async () => {
      await syncStatus()
    }, 300)
    return () => window.clearInterval(interval)
  }, [isPlaying, isProbePlaying, isPreparingPlayback, syncStatus])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void syncStatus()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [syncStatus])

  const play = useCallback(async () => {
    if (analysisState === 'analyzing') return
    if (isPlaying) return
    if (isProbePlaying) return
    if (isPreparingPlayback) return
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
        speed_ratio: speedValue,
        time_stretch_mode: timeStretchMode,
      })
      if (result.status === 'ok') {
        playbackStartRef.current = startPosition
        wasNormalPlayingRef.current = speedValue === 1
        if (speedValue === 1) {
          setIsPlaying(true)
        } else {
          setIsPreparingPlayback(true)
        }
      } else if (result.status === 'error') {
        reportError('Playback', result.message ?? 'Failed to play')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Playback', message)
    }
  }, [
    analysisState,
    isPlaying,
    isPreparingPlayback,
    isProbePlaying,
    mixValue,
    playbackPosition,
    reportError,
    speedValue,
    timeStretchMode,
  ])

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
        setIsPreparingPlayback(false)
        wasNormalPlayingRef.current = false
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
    async (positionSec: number) => {
      if (isPlaying || isPreparingPlayback) return
      const nextPosition = Math.max(0, positionSec)
      setPlaybackPosition(nextPosition)
      if (!isProbePlaying) return
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.update_harmonic_probe) return
      try {
        const result = await api.update_harmonic_probe({ time_sec: nextPosition })
        if (result.status === 'error') {
          reportError('Harmonic Probe', result.message ?? 'Failed to update probe')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Harmonic Probe', message)
      }
    },
    [isPlaying, isPreparingPlayback, isProbePlaying, reportError],
  )

  const togglePlayStop = useCallback(async () => {
    if (isPlaying) {
      await stop()
      return
    }
    await play()
  }, [isPlaying, play, stop])

  const toggleHarmonicProbe = useCallback(async () => {
    if (analysisState === 'analyzing') return
    if (isPlaying) return
    if (isPreparingPlayback) return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.start_harmonic_probe || !api?.stop_harmonic_probe) {
      reportError('Harmonic Probe', 'API not available')
      return
    }
    try {
      if (isProbePlaying) {
        const result = await api.stop_harmonic_probe()
        if (result.status === 'ok') {
          setIsProbePlaying(false)
        } else {
          reportError('Harmonic Probe', result.message ?? 'Failed to stop probe')
        }
        return
      }
      const result = await api.start_harmonic_probe({ time_sec: playbackPosition })
      if (result.status === 'ok') {
        setIsProbePlaying(true)
      } else {
        reportError('Harmonic Probe', result.message ?? 'Failed to start probe')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Harmonic Probe', message)
    }
  }, [analysisState, isPlaying, isPreparingPlayback, isProbePlaying, playbackPosition, reportError])

  const setMasterVolume = useCallback(
    async (value: number) => {
      const next = Math.max(0, Math.min(100, Math.round(value)))
      setMasterVolumeState(next)
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.set_master_volume) {
        reportError('Master Volume', 'API not available')
        return
      }
      try {
        const result = await api.set_master_volume({ master_volume: next / 100 })
        if (result.status === 'ok') {
          setMasterVolumeState(Math.round(Math.max(0, Math.min(1, result.master_volume)) * 100))
        } else if (result.status === 'error') {
          reportError('Master Volume', result.message ?? 'Failed to update master volume')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Master Volume', message)
      }
    },
    [reportError],
  )

  const setMixValueRealtime = useCallback(
    async (value: number) => {
      const next = Math.max(0, Math.min(100, Math.round(value)))
      setMixValue(next)
      if (!isPlaying || isProbePlaying || isPreparingPlayback) return
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.update_playback_mix) {
        reportError('Playback Mix', 'API not available')
        return
      }
      try {
        const result = await api.update_playback_mix({ mix_ratio: next / 100 })
        if (result.status === 'error') {
          reportError('Playback Mix', result.message ?? 'Failed to update playback mix')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Playback Mix', message)
      }
    },
    [isPlaying, isProbePlaying, isPreparingPlayback, reportError],
  )

  const applyPlaybackSettings = useCallback((masterVolumeRatio: number) => {
    const clamped = Math.max(0, Math.min(1, masterVolumeRatio))
    setMasterVolumeState(Math.round(clamped * 100))
  }, [])

  return {
    isPlaying,
    isProbePlaying,
    isPreparingPlayback,
    mixValue,
    masterVolume,
    speedValue,
    speedPresetIndex,
    timeStretchMode,
    playbackPosition,
    setMixValue: setMixValueRealtime,
    setMasterVolume,
    applyPlaybackSettings,
    syncStatus,
    setSpeedPresetIndex,
    setTimeStretchMode,
    play,
    stop,
    togglePlayStop,
    setPlayheadPosition,
    toggleHarmonicProbe,
  }
}
