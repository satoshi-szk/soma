import { z } from 'zod'

const analysisSettingsSchema = z.object({
  freq_min: z.number(),
  freq_max: z.number(),
  bins_per_octave: z.number(),
  time_resolution_ms: z.number(),
  preview_freq_max: z.number(),
  preview_bins_per_octave: z.number(),
  wavelet_bandwidth: z.number(),
  wavelet_center_freq: z.number(),
  gain: z.number(),
  min_db: z.number(),
  max_db: z.number(),
  gamma: z.number(),
})
const playbackSettingsSchema = z.object({
  master_volume: z.number(),
  output_mode: z.enum(['audio', 'midi']),
  mix_ratio: z.number(),
  speed_ratio: z.number(),
  time_stretch_mode: z.enum(['native', 'librosa']),
  midi_mode: z.enum(['mpe', 'multitrack', 'mono']),
  midi_output_name: z.string(),
  midi_pitch_bend_range: z.number(),
  midi_amplitude_mapping: z.enum(['velocity', 'pressure', 'cc74', 'cc1']),
  midi_amplitude_curve: z.enum(['linear', 'db']),
  midi_cc_update_rate_hz: z.number(),
  midi_bpm: z.number(),
})

const audioInfoSchema = z.object({
  path: z.string(),
  name: z.string(),
  sample_rate: z.number(),
  duration_sec: z.number(),
  channels: z.number(),
  truncated: z.boolean(),
})

const recentProjectSchema = z.object({
  path: z.string(),
  name: z.string(),
  last_opened_at: z.string(),
  exists: z.boolean(),
})

const spectrogramPreviewSchema = z.object({
  width: z.number(),
  height: z.number(),
  image_path: z.string(),
  time_start: z.number(),
  time_end: z.number(),
  freq_min: z.number(),
  freq_max: z.number(),
  duration_sec: z.number(),
})

const partialPointTupleSchema = z.tuple([z.number(), z.number(), z.number()])

const partialSchema = z.object({
  id: z.string(),
  is_muted: z.boolean(),
  color: z.string(),
  points: z.array(partialPointTupleSchema),
})

const okStatusSchema = z.object({ status: z.literal('ok') })
const cancelledStatusSchema = z.object({ status: z.literal('cancelled') })
const errorStatusSchema = z.object({ status: z.literal('error'), message: z.string().optional() })

const loadResponseSchema = z.union([
  z.object({
    status: z.literal('ok'),
    audio: audioInfoSchema,
    preview: spectrogramPreviewSchema.nullable(),
    settings: analysisSettingsSchema,
    playback_settings: playbackSettingsSchema,
    partials: z.array(partialSchema),
  }),
  cancelledStatusSchema,
  errorStatusSchema,
])

const updateSettingsResponseSchema = z.union([
  z.object({
    status: z.literal('ok'),
    settings: analysisSettingsSchema,
    preview: spectrogramPreviewSchema.nullable(),
  }),
  errorStatusSchema,
])

const statusResponseSchema = z.object({
  status: z.literal('ok'),
  is_playing: z.boolean(),
  is_probe_playing: z.boolean(),
  is_preparing_playback: z.boolean(),
  is_resynthesizing: z.boolean(),
  position: z.number(),
  master_volume: z.number(),
  playback_settings: playbackSettingsSchema,
})

const playbackStateResponseSchema = z.object({
  status: z.literal('ok'),
  position: z.number(),
})

const partialResponseSchema = z.object({ status: z.literal('ok'), partial: partialSchema })
const partialsResponseSchema = z.object({ status: z.literal('ok'), partials: z.array(partialSchema) })
const acceptedStatusSchema = z.object({ status: z.literal('accepted'), request_id: z.string() })

export const apiSchemas = {
  health: { response: okStatusSchema },
  status: { response: statusResponseSchema },
  playback_state: { response: playbackStateResponseSchema },
  frontend_log: {
    payload: z.object({ level: z.string(), message: z.string() }),
    response: okStatusSchema,
  },
  open_audio: { response: loadResponseSchema },
  open_audio_path: { payload: z.object({ path: z.string() }), response: loadResponseSchema },
  open_audio_data: {
    payload: z.object({ name: z.string(), data_base64: z.string() }),
    response: loadResponseSchema,
  },
  new_project: {
    response: z.union([okStatusSchema.extend({ playback_settings: playbackSettingsSchema }), errorStatusSchema]),
  },
  list_recent_projects: {
    response: z.union([okStatusSchema.extend({ projects: z.array(recentProjectSchema) }), errorStatusSchema]),
  },
  open_project: { response: loadResponseSchema },
  open_project_path: { payload: z.object({ path: z.string() }), response: loadResponseSchema },
  save_project: { response: z.union([okStatusSchema.extend({ path: z.string().optional() }), errorStatusSchema]) },
  save_project_as: { response: z.union([okStatusSchema.extend({ path: z.string() }), errorStatusSchema]) },
  reveal_audio_in_explorer: { response: z.union([okStatusSchema, errorStatusSchema]) },
  update_settings: { payload: analysisSettingsSchema, response: updateSettingsResponseSchema },
  trace_partial: {
    payload: z.object({ trace: z.array(z.tuple([z.number(), z.number()])) }),
    response: z.union([acceptedStatusSchema, errorStatusSchema]),
  },
  erase_partial: {
    payload: z.object({ trace: z.array(z.tuple([z.number(), z.number()])), radius_hz: z.number().optional() }),
    response: z.union([partialsResponseSchema, errorStatusSchema]),
  },
  update_partial: {
    payload: z.object({ id: z.string(), points: z.array(partialPointTupleSchema) }),
    response: z.union([partialResponseSchema, errorStatusSchema]),
  },
  merge_partials: {
    payload: z.object({ first: z.string(), second: z.string() }),
    response: z.union([partialResponseSchema, errorStatusSchema]),
  },
  delete_partials: {
    payload: z.object({ ids: z.array(z.string()) }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  toggle_mute: {
    payload: z.object({ id: z.string() }),
    response: z.union([partialResponseSchema, errorStatusSchema]),
  },
  hit_test: {
    payload: z.object({ time: z.number(), freq: z.number(), tolerance: z.number().optional() }),
    response: z.union([
      z.object({ status: z.literal('ok'), id: z.string().nullable() }),
      errorStatusSchema,
    ]),
  },
  select_in_box: {
    payload: z.object({
      time_start: z.number(),
      time_end: z.number(),
      freq_start: z.number(),
      freq_end: z.number(),
    }),
    response: z.union([z.object({ status: z.literal('ok'), ids: z.array(z.string()) }), errorStatusSchema]),
  },
  undo: { response: z.union([partialsResponseSchema, errorStatusSchema]) },
  redo: { response: z.union([partialsResponseSchema, errorStatusSchema]) },
  play: {
    payload: z.object({
      mix_ratio: z.number().optional(),
      loop: z.boolean().optional(),
      start_position_sec: z.number().optional(),
      speed_ratio: z.number().optional(),
      time_stretch_mode: z.enum(['native', 'librosa']).optional(),
    }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  start_harmonic_probe: {
    payload: z.object({ time_sec: z.number() }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  update_harmonic_probe: {
    payload: z.object({ time_sec: z.number() }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  stop_harmonic_probe: { response: z.union([okStatusSchema, errorStatusSchema]) },
  pause: { response: z.union([okStatusSchema, errorStatusSchema]) },
  stop: {
    payload: z.object({ return_position_sec: z.number().optional() }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  set_master_volume: {
    payload: z.object({ master_volume: z.number() }),
    response: z.union([okStatusSchema.extend({ master_volume: z.number() }), errorStatusSchema]),
  },
  update_playback_mix: {
    payload: z.object({ mix_ratio: z.number() }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  list_midi_outputs: {
    response: z.union([okStatusSchema.extend({ outputs: z.array(z.string()), error: z.string().nullable().optional() }), errorStatusSchema]),
  },
  update_playback_settings: {
    payload: z.object({
      output_mode: z.enum(['audio', 'midi']).optional(),
      mix_ratio: z.number().optional(),
      speed_ratio: z.number().optional(),
      time_stretch_mode: z.enum(['native', 'librosa']).optional(),
      midi_mode: z.enum(['mpe', 'multitrack', 'mono']).optional(),
      midi_output_name: z.string().optional(),
      midi_pitch_bend_range: z.number().optional(),
      midi_amplitude_mapping: z.enum(['velocity', 'pressure', 'cc74', 'cc1']).optional(),
      midi_amplitude_curve: z.enum(['linear', 'db']).optional(),
      midi_cc_update_rate_hz: z.number().optional(),
      midi_bpm: z.number().optional(),
    }),
    response: z.union([okStatusSchema.extend({ playback_settings: playbackSettingsSchema }), errorStatusSchema]),
  },
  export_mpe: {
    payload: z.object({
      pitch_bend_range: z.number().optional(),
      amplitude_mapping: z.string().optional(),
      amplitude_curve: z.string().optional(),
      cc_update_rate_hz: z.number().optional(),
      bpm: z.number().optional(),
    }),
    response: z.union([okStatusSchema.extend({ paths: z.array(z.string()) }), errorStatusSchema]),
  },
  export_multitrack_midi: {
    payload: z.object({
      pitch_bend_range: z.number().optional(),
      amplitude_mapping: z.string().optional(),
      amplitude_curve: z.string().optional(),
      cc_update_rate_hz: z.number().optional(),
      bpm: z.number().optional(),
    }),
    response: z.union([okStatusSchema.extend({ paths: z.array(z.string()) }), errorStatusSchema]),
  },
  export_monophonic_midi: {
    payload: z.object({
      pitch_bend_range: z.number().optional(),
      amplitude_mapping: z.string().optional(),
      amplitude_curve: z.string().optional(),
      cc_update_rate_hz: z.number().optional(),
      bpm: z.number().optional(),
    }),
    response: z.union([okStatusSchema.extend({ paths: z.array(z.string()) }), errorStatusSchema]),
  },
  export_audio: {
    payload: z.object({
      sample_rate: z.number().optional(),
      bit_depth: z.number().optional(),
      output_type: z.string().optional(),
      cv_base_freq: z.number().optional(),
      cv_full_scale_volts: z.number().optional(),
      cv_mode: z.string().optional(),
      amplitude_curve: z.string().optional(),
    }),
    response: z.union([okStatusSchema.extend({ path: z.string(), paths: z.array(z.string()).optional() }), errorStatusSchema]),
  },
  request_spectrogram_tile: {
    payload: z.object({
      time_start: z.number(),
      time_end: z.number(),
      freq_min: z.number(),
      freq_max: z.number(),
      width: z.number(),
      height: z.number(),
      gain: z.number().optional(),
      min_db: z.number().optional(),
      max_db: z.number().optional(),
      gamma: z.number().optional(),
    }),
    response: z.union([
      z.object({
        status: z.literal('ok'),
        quality: z.enum(['low', 'high']).optional(),
        preview: spectrogramPreviewSchema,
      }),
      errorStatusSchema,
    ]),
  },
  request_spectrogram_overview: {
    payload: z.object({
      width: z.number().optional(),
      height: z.number().optional(),
      gain: z.number().optional(),
      min_db: z.number().optional(),
      max_db: z.number().optional(),
      gamma: z.number().optional(),
    }),
    response: z.union([
      z.object({
        status: z.literal('ok'),
        quality: z.enum(['low', 'high']).optional(),
        preview: spectrogramPreviewSchema,
      }),
      errorStatusSchema,
    ]),
  },
} as const
