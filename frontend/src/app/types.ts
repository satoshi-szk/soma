export type ApiStatus = 'checking' | 'connected' | 'disconnected'

export type AnalysisSettings = {
  freq_min: number
  freq_max: number
  bins_per_octave: number
  time_resolution_ms: number
  preview_freq_max: number
  preview_bins_per_octave: number
  wavelet_bandwidth: number
  wavelet_center_freq: number
  brightness: number
  contrast: number
}

export type AudioInfo = {
  path: string
  name: string
  sample_rate: number
  duration_sec: number
  channels: number
  truncated: boolean
}

export type PlaybackSettings = {
  master_volume: number
  output_mode: 'audio' | 'midi'
  mix_ratio: number
  speed_ratio: number
  time_stretch_mode: 'native' | 'librosa'
  midi_mode: 'mpe' | 'multitrack' | 'mono'
  midi_output_name: string
  midi_pitch_bend_range: number
  midi_amplitude_mapping: 'velocity' | 'pressure' | 'cc74' | 'cc1'
  midi_amplitude_curve: 'linear' | 'db'
  midi_cc_update_rate_hz: number
  midi_bpm: number
}

export type SpectrogramPreview = {
  width: number
  height: number
  data: number[]
  data_path?: string
  data_length?: number
  time_start: number
  time_end: number
  freq_min: number
  freq_max: number
  duration_sec: number
}

export type PartialPoint = {
  time: number
  freq: number
  amp: number
}

export type Partial = {
  id: string
  is_muted: boolean
  color: string
  points: PartialPoint[]
}

export type ApiResult<T> = { status: 'ok' } & T

export type ApiError = { status: 'cancelled' | 'error'; message?: string }

export type ToolId = 'select' | 'trace' | 'erase' | 'connect'
