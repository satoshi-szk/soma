import { useState, useEffect, useCallback, useRef } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'

type ReportError = (context: string, message: string) => void
type TimeStretchMode = 'native' | 'librosa'
type PlaybackMode = 'audio' | 'midi'
type MidiMode = 'mpe' | 'multitrack' | 'mono'
type MidiAmplitudeMapping = 'velocity' | 'pressure' | 'cc74' | 'cc1'
type MidiAmplitudeCurve = 'linear' | 'db'

const SPEED_PRESET_RATIOS = [0.125, 0.25, 0.5, 1, 2, 4, 8] as const
const DEFAULT_SPEED_PRESET_INDEX = 3

function closestSpeedPresetIndex(value: number): number {
  let bestIndex = DEFAULT_SPEED_PRESET_INDEX
  let bestDiff = Number.POSITIVE_INFINITY
  SPEED_PRESET_RATIOS.forEach((ratio, index) => {
    const diff = Math.abs(ratio - value)
    if (diff < bestDiff) {
      bestDiff = diff
      bestIndex = index
    }
  })
  return bestIndex
}

export function usePlayback(reportError: ReportError, analysisState: 'idle' | 'analyzing' | 'error') {
  const [isPlaying, setIsPlaying] = useState(false)
  const [isProbePlaying, setIsProbePlaying] = useState(false)
  const [isPreparingPlayback, setIsPreparingPlayback] = useState(false)
  const [playbackMode, setPlaybackModeState] = useState<PlaybackMode>('audio')
  const [mixValue, setMixValue] = useState(55)
  const [masterVolume, setMasterVolumeState] = useState(100)
  const [speedPresetIndex, setSpeedPresetIndexState] = useState(DEFAULT_SPEED_PRESET_INDEX)
  const [timeStretchMode, setTimeStretchModeState] = useState<TimeStretchMode>('librosa')
  const [playbackPosition, setPlaybackPosition] = useState(0)
  const [midiMode, setMidiMode] = useState<MidiMode>('mpe')
  const [midiOutputName, setMidiOutputName] = useState('')
  const [midiPitchBendRange, setMidiPitchBendRange] = useState('48')
  const [midiAmplitudeMapping, setMidiAmplitudeMapping] = useState<MidiAmplitudeMapping>('cc74')
  const [midiAmplitudeCurve, setMidiAmplitudeCurve] = useState<MidiAmplitudeCurve>('linear')
  const [midiBpm, setMidiBpm] = useState('120')
  const [midiOutputs, setMidiOutputs] = useState<string[]>([])
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
    setPlaybackModeState(result.playback_settings.output_mode)
    setMixValue(Math.round(Math.max(0, Math.min(1, result.playback_settings.mix_ratio)) * 100))
    setSpeedPresetIndexState(closestSpeedPresetIndex(result.playback_settings.speed_ratio))
    setTimeStretchModeState(result.playback_settings.time_stretch_mode)
    setMidiMode(result.playback_settings.midi_mode)
    setMidiOutputName(result.playback_settings.midi_output_name)
    setMidiPitchBendRange(String(result.playback_settings.midi_pitch_bend_range))
    setMidiAmplitudeMapping(result.playback_settings.midi_amplitude_mapping)
    setMidiAmplitudeCurve(result.playback_settings.midi_amplitude_curve)
    setMidiBpm(String(result.playback_settings.midi_bpm))
    if (endedNaturally) {
      setPlaybackPosition(playbackStartRef.current)
    }
    wasNormalPlayingRef.current = result.is_playing
  }, [])

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

  const refreshMidiOutputs = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.list_midi_outputs) return
    try {
      const result = await api.list_midi_outputs()
      if (result.status === 'ok') {
        setMidiOutputs(result.outputs)
      }
    } catch {
      // 無視: UIで都度エラーを出さない
    }
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void refreshMidiOutputs()
    }, 0)
    return () => window.clearTimeout(timer)
  }, [refreshMidiOutputs])

  const updatePlaybackSettings = useCallback(
    async (payload: {
      output_mode?: PlaybackMode
      mix_ratio?: number
      speed_ratio?: number
      time_stretch_mode?: TimeStretchMode
      midi_mode?: MidiMode
      midi_output_name?: string
      midi_pitch_bend_range?: number
      midi_amplitude_mapping?: MidiAmplitudeMapping
      midi_amplitude_curve?: MidiAmplitudeCurve
      midi_bpm?: number
    }) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.update_playback_settings) {
        reportError('Playback Settings', 'API not available')
        return
      }
      try {
        const result = await api.update_playback_settings(payload)
        if (result.status === 'ok') {
          setPlaybackModeState(result.playback_settings.output_mode)
          setMixValue(Math.round(Math.max(0, Math.min(1, result.playback_settings.mix_ratio)) * 100))
          setSpeedPresetIndexState(closestSpeedPresetIndex(result.playback_settings.speed_ratio))
          setTimeStretchModeState(result.playback_settings.time_stretch_mode)
          setMidiMode(result.playback_settings.midi_mode)
          setMidiOutputName(result.playback_settings.midi_output_name)
          setMidiPitchBendRange(String(result.playback_settings.midi_pitch_bend_range))
          setMidiAmplitudeMapping(result.playback_settings.midi_amplitude_mapping)
          setMidiAmplitudeCurve(result.playback_settings.midi_amplitude_curve)
          setMidiBpm(String(result.playback_settings.midi_bpm))
        } else {
          reportError('Playback Settings', result.message ?? 'Failed to update playback settings')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Playback Settings', message)
      }
    },
    [reportError],
  )

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
        if (playbackMode === 'audio' && speedValue !== 1) {
          setIsPreparingPlayback(true)
        } else {
          setIsPlaying(true)
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
    playbackMode,
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
    if (playbackMode === 'midi') return
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
  }, [analysisState, isPlaying, isPreparingPlayback, isProbePlaying, playbackMode, playbackPosition, reportError])

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
      if (!isPlaying || isProbePlaying || isPreparingPlayback || playbackMode === 'midi') return
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
    [isPlaying, isProbePlaying, isPreparingPlayback, playbackMode, reportError],
  )

  const commitMixValue = useCallback(
    (value: number) => {
      const next = Math.max(0, Math.min(100, Math.round(value)))
      void updatePlaybackSettings({ mix_ratio: next / 100 })
    },
    [updatePlaybackSettings],
  )

  const setSpeedPresetIndex = useCallback((index: number) => {
    const next = Math.max(0, Math.min(SPEED_PRESET_RATIOS.length - 1, Math.round(index)))
    setSpeedPresetIndexState(next)
  }, [])

  const commitSpeedPresetIndex = useCallback(
    (index: number) => {
      const next = Math.max(0, Math.min(SPEED_PRESET_RATIOS.length - 1, Math.round(index)))
      void updatePlaybackSettings({ speed_ratio: SPEED_PRESET_RATIOS[next] })
    },
    [updatePlaybackSettings],
  )

  const setTimeStretchMode = useCallback(
    (mode: TimeStretchMode) => {
      setTimeStretchModeState(mode)
      void updatePlaybackSettings({ time_stretch_mode: mode })
    },
    [updatePlaybackSettings],
  )

  const setPlaybackMode = useCallback(
    (mode: PlaybackMode) => {
      setPlaybackModeState(mode)
      void updatePlaybackSettings({ output_mode: mode })
    },
    [updatePlaybackSettings],
  )

  const setMidiPlaybackMode = useCallback(
    (mode: MidiMode) => {
      setMidiMode(mode)
      void updatePlaybackSettings({ midi_mode: mode })
    },
    [updatePlaybackSettings],
  )

  const setMidiOutput = useCallback(
    (outputName: string) => {
      setMidiOutputName(outputName)
      void updatePlaybackSettings({ midi_output_name: outputName })
    },
    [updatePlaybackSettings],
  )

  const setMidiPitchBend = useCallback(
    (value: string) => {
      setMidiPitchBendRange(value)
      const parsed = Number(value)
      if (Number.isFinite(parsed) && parsed > 0) {
        void updatePlaybackSettings({ midi_pitch_bend_range: Math.round(parsed) })
      }
    },
    [updatePlaybackSettings],
  )

  const setMidiAmplitudeMappingSetting = useCallback(
    (value: MidiAmplitudeMapping) => {
      setMidiAmplitudeMapping(value)
      void updatePlaybackSettings({ midi_amplitude_mapping: value })
    },
    [updatePlaybackSettings],
  )

  const setMidiAmplitudeCurveSetting = useCallback(
    (value: MidiAmplitudeCurve) => {
      setMidiAmplitudeCurve(value)
      void updatePlaybackSettings({ midi_amplitude_curve: value })
    },
    [updatePlaybackSettings],
  )

  const setMidiBpmSetting = useCallback(
    (value: string) => {
      setMidiBpm(value)
      const parsed = Number(value)
      if (Number.isFinite(parsed) && parsed > 0) {
        void updatePlaybackSettings({ midi_bpm: parsed })
      }
    },
    [updatePlaybackSettings],
  )

  const applyPlaybackSettings = useCallback(
    (settings: {
      master_volume: number
      output_mode: PlaybackMode
      mix_ratio: number
      speed_ratio: number
      time_stretch_mode: TimeStretchMode
      midi_mode: MidiMode
      midi_output_name: string
      midi_pitch_bend_range: number
      midi_amplitude_mapping: MidiAmplitudeMapping
      midi_amplitude_curve: MidiAmplitudeCurve
      midi_bpm: number
    }) => {
      setMasterVolumeState(Math.round(Math.max(0, Math.min(1, settings.master_volume)) * 100))
      setPlaybackModeState(settings.output_mode)
      setMixValue(Math.round(Math.max(0, Math.min(1, settings.mix_ratio)) * 100))
      setSpeedPresetIndexState(closestSpeedPresetIndex(settings.speed_ratio))
      setTimeStretchModeState(settings.time_stretch_mode)
      setMidiMode(settings.midi_mode)
      setMidiOutputName(settings.midi_output_name)
      setMidiPitchBendRange(String(settings.midi_pitch_bend_range))
      setMidiAmplitudeMapping(settings.midi_amplitude_mapping)
      setMidiAmplitudeCurve(settings.midi_amplitude_curve)
      setMidiBpm(String(settings.midi_bpm))
    },
    [],
  )

  return {
    isPlaying,
    isProbePlaying,
    isPreparingPlayback,
    playbackMode,
    mixValue,
    masterVolume,
    speedValue,
    speedPresetIndex,
    timeStretchMode,
    playbackPosition,
    midiMode,
    midiOutputName,
    midiPitchBendRange,
    midiAmplitudeMapping,
    midiAmplitudeCurve,
    midiBpm,
    midiOutputs,
    setMixValue: setMixValueRealtime,
    commitMixValue,
    setMasterVolume,
    setPlaybackMode,
    setSpeedPresetIndex,
    commitSpeedPresetIndex,
    setTimeStretchMode,
    setMidiPlaybackMode,
    setMidiOutput,
    setMidiPitchBend,
    setMidiAmplitudeMapping: setMidiAmplitudeMappingSetting,
    setMidiAmplitudeCurve: setMidiAmplitudeCurveSetting,
    setMidiBpm: setMidiBpmSetting,
    refreshMidiOutputs,
    applyPlaybackSettings,
    syncStatus,
    play,
    stop,
    togglePlayStop,
    setPlayheadPosition,
    toggleHarmonicProbe,
  }
}
