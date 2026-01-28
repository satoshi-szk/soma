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
  color_map: z.string(),
  brightness: z.number(),
  contrast: z.number(),
})

const audioInfoSchema = z.object({
  path: z.string(),
  name: z.string(),
  sample_rate: z.number(),
  duration_sec: z.number(),
  channels: z.number(),
  truncated: z.boolean(),
})

const spectrogramPreviewSchema = z.object({
  width: z.number(),
  height: z.number(),
  data: z.array(z.number()).optional().default([]),
  data_path: z.string().optional(),
  data_length: z.number().optional(),
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
    status: z.enum(['ok', 'processing']),
    audio: audioInfoSchema,
    preview: spectrogramPreviewSchema.nullable(),
    settings: analysisSettingsSchema,
    partials: z.array(partialSchema),
  }),
  cancelledStatusSchema,
  errorStatusSchema,
])

const updateSettingsResponseSchema = z.union([
  z.object({
    status: z.enum(['ok', 'processing']),
    settings: analysisSettingsSchema,
    preview: spectrogramPreviewSchema.nullable(),
  }),
  errorStatusSchema,
])

const statusResponseSchema = z.object({
  status: z.literal('ok'),
  is_playing: z.boolean(),
  is_resynthesizing: z.boolean(),
  position: z.number(),
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
  new_project: { response: z.union([okStatusSchema, errorStatusSchema]) },
  open_project: { response: loadResponseSchema },
  save_project: { response: z.union([okStatusSchema.extend({ path: z.string().optional() }), errorStatusSchema]) },
  save_project_as: { response: z.union([okStatusSchema.extend({ path: z.string() }), errorStatusSchema]) },
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
    payload: z.object({ mix_ratio: z.number().optional(), loop: z.boolean().optional() }),
    response: z.union([okStatusSchema, errorStatusSchema]),
  },
  pause: { response: z.union([okStatusSchema, errorStatusSchema]) },
  stop: { response: z.union([okStatusSchema, errorStatusSchema]) },
  export_mpe: {
    payload: z.object({
      pitch_bend_range: z.number().optional(),
      amplitude_mapping: z.string().optional(),
      bpm: z.number().optional(),
    }),
    response: z.union([okStatusSchema.extend({ paths: z.array(z.string()) }), errorStatusSchema]),
  },
  export_multitrack_midi: {
    payload: z.object({
      pitch_bend_range: z.number().optional(),
      amplitude_mapping: z.string().optional(),
      bpm: z.number().optional(),
    }),
    response: z.union([okStatusSchema.extend({ paths: z.array(z.string()) }), errorStatusSchema]),
  },
  export_monophonic_midi: {
    payload: z.object({
      pitch_bend_range: z.number().optional(),
      amplitude_mapping: z.string().optional(),
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
    }),
    response: z.union([okStatusSchema.extend({ path: z.string() }), errorStatusSchema]),
  },
  request_viewport_preview: {
    payload: z.object({
      time_start: z.number(),
      time_end: z.number(),
      freq_min: z.number(),
      freq_max: z.number(),
      width: z.number(),
      height: z.number(),
    }),
    response: z.union([z.object({ status: z.literal('accepted') }), errorStatusSchema]),
  },
} as const
