import type { z } from 'zod'
import { apiSchemas } from './apiSchemas'

type ApiSchemas = typeof apiSchemas
type ApiMethod = keyof ApiSchemas
type PayloadFor<M extends ApiMethod> = ApiSchemas[M] extends { payload: z.ZodTypeAny }
  ? z.infer<ApiSchemas[M]['payload']>
  : never
type ResponseFor<M extends ApiMethod> = z.infer<ApiSchemas[M]['response']>

type RawApi = NonNullable<typeof window.pywebview>['api']

const getRawApi = (): RawApi | undefined => window.pywebview?.api

const callApi = async <M extends ApiMethod>(method: M, payload?: PayloadFor<M>): Promise<ResponseFor<M>> => {
  const api = getRawApi()
  const handler = api?.[method] as ((arg?: unknown) => Promise<unknown>) | undefined
  if (!handler) {
    throw new Error(`Pywebview API is not available: ${String(method)}`)
  }
  const schema = apiSchemas[method]
  if ('payload' in schema) {
    const parsedPayload = schema.payload.parse(payload)
    const response = await handler(parsedPayload)
    return schema.response.parse(response) as ResponseFor<M>
  }
  const response = await handler()
  return schema.response.parse(response) as ResponseFor<M>
}

export const pywebviewApi = {
  health: () => callApi('health'),
  status: () => callApi('status'),
  playback_state: () => callApi('playback_state'),
  frontend_log: (payload: PayloadFor<'frontend_log'>) => callApi('frontend_log', payload),
  open_audio: () => callApi('open_audio'),
  open_audio_path: (payload: PayloadFor<'open_audio_path'>) => callApi('open_audio_path', payload),
  open_audio_data: (payload: PayloadFor<'open_audio_data'>) => callApi('open_audio_data', payload),
  new_project: () => callApi('new_project'),
  open_project: () => callApi('open_project'),
  save_project: () => callApi('save_project'),
  save_project_as: () => callApi('save_project_as'),
  update_settings: (payload: PayloadFor<'update_settings'>) => callApi('update_settings', payload),
  trace_partial: (payload: PayloadFor<'trace_partial'>) => callApi('trace_partial', payload),
  erase_partial: (payload: PayloadFor<'erase_partial'>) => callApi('erase_partial', payload),
  update_partial: (payload: PayloadFor<'update_partial'>) => callApi('update_partial', payload),
  merge_partials: (payload: PayloadFor<'merge_partials'>) => callApi('merge_partials', payload),
  delete_partials: (payload: PayloadFor<'delete_partials'>) => callApi('delete_partials', payload),
  toggle_mute: (payload: PayloadFor<'toggle_mute'>) => callApi('toggle_mute', payload),
  hit_test: (payload: PayloadFor<'hit_test'>) => callApi('hit_test', payload),
  select_in_box: (payload: PayloadFor<'select_in_box'>) => callApi('select_in_box', payload),
  undo: () => callApi('undo'),
  redo: () => callApi('redo'),
  play: (payload: PayloadFor<'play'>) => callApi('play', payload),
  pause: () => callApi('pause'),
  stop: () => callApi('stop'),
  export_mpe: (payload: PayloadFor<'export_mpe'>) => callApi('export_mpe', payload),
  export_audio: (payload: PayloadFor<'export_audio'>) => callApi('export_audio', payload),
  request_viewport_preview: (payload: PayloadFor<'request_viewport_preview'>) =>
    callApi('request_viewport_preview', payload),
}

export const isPywebviewApiAvailable = (): boolean => Boolean(getRawApi())
