import { useReducer, useCallback, useRef, useEffect } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import { toPartial } from '../app/utils'
import type { Partial, PartialPoint } from '../app/types'

type PartialsState = {
  items: Partial[]
  selection: string[]
  isSnapping: boolean
}

type PartialsAction =
  | { type: 'ADD'; partial: Partial }
  | { type: 'SET'; partials: Partial[] }
  | { type: 'UPDATE'; partial: Partial }
  | { type: 'DELETE'; ids: string[] }
  | { type: 'REMOVE_AND_ADD'; removeIds: string[]; partial: Partial }
  | { type: 'SELECT'; ids: string[] }
  | { type: 'SET_SNAPPING'; value: boolean }

function partialsReducer(state: PartialsState, action: PartialsAction): PartialsState {
  switch (action.type) {
    case 'ADD':
      return { ...state, items: [...state.items, action.partial] }
    case 'SET':
      return { ...state, items: action.partials }
    case 'UPDATE':
      return {
        ...state,
        items: state.items.map((p) => (p.id === action.partial.id ? action.partial : p)),
      }
    case 'DELETE':
      return {
        ...state,
        items: state.items.filter((p) => !action.ids.includes(p.id)),
        selection: state.selection.filter((id) => !action.ids.includes(id)),
      }
    case 'REMOVE_AND_ADD':
      return {
        ...state,
        items: [...state.items.filter((p) => !action.removeIds.includes(p.id)), action.partial],
        selection: [action.partial.id],
      }
    case 'SELECT':
      return { ...state, selection: action.ids }
    case 'SET_SNAPPING':
      return { ...state, isSnapping: action.value }
  }
}

const initialState: PartialsState = {
  items: [],
  selection: [],
  isSnapping: false,
}

type ReportError = (context: string, message: string) => void

export function usePartials(reportError: ReportError) {
  const [state, dispatch] = useReducer(partialsReducer, initialState)
  const connectQueueRef = useRef<string[]>([])
  const pendingSnapRef = useRef<{
    requestId: string
    resolve: (value: boolean) => void
  } | null>(null)

  // snap イベントを購読する
  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail as unknown
      if (!detail || typeof detail !== 'object') return
      const payload = detail as Record<string, unknown>

      if (payload.type === 'snap_completed') {
        const requestId = payload.request_id as string | undefined
        const partial = payload.partial as
          | { id: string; is_muted: boolean; color?: string; points: number[][] }
          | undefined

        // 現在待機中のリクエストかどうかを確認する
        if (pendingSnapRef.current && pendingSnapRef.current.requestId === requestId) {
          if (partial) {
            dispatch({ type: 'ADD', partial: toPartial(partial) })
          }
          dispatch({ type: 'SET_SNAPPING', value: false })
          pendingSnapRef.current.resolve(!!partial)
          pendingSnapRef.current = null
        }
      }

      if (payload.type === 'snap_error') {
        const requestId = payload.request_id as string | undefined
        const message = payload.message as string | undefined

        if (pendingSnapRef.current && pendingSnapRef.current.requestId === requestId) {
          reportError('Trace', message ?? 'Failed to create partial')
          dispatch({ type: 'SET_SNAPPING', value: false })
          pendingSnapRef.current.resolve(false)
          pendingSnapRef.current = null
        }
      }
    }

    window.addEventListener('soma:event', handler)
    return () => {
      window.removeEventListener('soma:event', handler)
    }
  }, [reportError])

  const setPartials = useCallback((partials: Partial[]) => {
    dispatch({ type: 'SET', partials })
  }, [])

  const setSelection = useCallback((ids: string[]) => {
    dispatch({ type: 'SELECT', ids })
  }, [])

  const trace = useCallback(
    async (trace: Array<[number, number]>): Promise<boolean> => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.trace_partial) {
        reportError('Trace', 'API not available')
        return false
      }
      dispatch({ type: 'SET_SNAPPING', value: true })
      try {
        const result = await api.trace_partial({ trace })
        if (result.status === 'accepted') {
          // snap_completed イベントを待つ
          return new Promise<boolean>((resolve) => {
            pendingSnapRef.current = {
              requestId: result.request_id,
              resolve,
            }
            // 5 分でタイムアウト（ワーカー側タイムアウトと同じ）
            setTimeout(() => {
              if (pendingSnapRef.current?.requestId === result.request_id) {
                reportError('Trace', 'Snap computation timed out')
                dispatch({ type: 'SET_SNAPPING', value: false })
                pendingSnapRef.current.resolve(false)
                pendingSnapRef.current = null
              }
            }, 5 * 60 * 1000)
          })
        }
        if (result.status === 'error') {
          reportError('Trace', result.message ?? 'Failed to create partial')
          dispatch({ type: 'SET_SNAPPING', value: false })
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Trace', message)
        dispatch({ type: 'SET_SNAPPING', value: false })
      }
      return false
    },
    [reportError]
  )

  const erase = useCallback(
    async (trace: Array<[number, number]>) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.erase_partial) {
        reportError('Erase', 'API not available')
        return
      }
      try {
        const result = await api.erase_partial({ trace, radius_hz: 60 })
        if (result.status === 'ok') {
          dispatch({ type: 'SET', partials: result.partials.map(toPartial) })
          dispatch({ type: 'SELECT', ids: [] })
        } else if (result.status === 'error') {
          reportError('Erase', result.message ?? 'Failed to erase')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Erase', message)
      }
    },
    [reportError]
  )

  const selectInBox = useCallback(
    async (box: { time_start: number; time_end: number; freq_start: number; freq_end: number }) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.select_in_box) {
        reportError('Select', 'API not available')
        return
      }
      try {
        const result = await api.select_in_box(box)
        if (result.status === 'ok') {
          dispatch({ type: 'SELECT', ids: result.ids })
        } else if (result.status === 'error') {
          reportError('Select', result.message ?? 'Selection failed')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Select', message)
      }
    },
    [reportError]
  )

  const hitTest = useCallback(
    async (point: { time: number; freq: number }) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.hit_test) {
        reportError('HitTest', 'API not available')
        return
      }
      try {
        const result = await api.hit_test({ ...point, tolerance: 0.08 })
        if (result.status === 'ok') {
          dispatch({ type: 'SELECT', ids: result.id ? [result.id] : [] })
        } else if (result.status === 'error') {
          reportError('HitTest', result.message ?? 'Hit test failed')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('HitTest', message)
      }
    },
    [reportError]
  )

  const updatePartial = useCallback(
    async (id: string, points: PartialPoint[]) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.update_partial) {
        reportError('Update', 'API not available')
        return
      }
      try {
        const payload = {
          id,
          points: points.map((point) => [point.time, point.freq, point.amp] as [number, number, number]),
        }
        const result = await api.update_partial(payload)
        if (result.status === 'ok') {
          dispatch({ type: 'UPDATE', partial: toPartial(result.partial) })
        } else if (result.status === 'error') {
          reportError('Update', result.message ?? 'Update failed')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Update', message)
      }
    },
    [reportError]
  )

  const connectPick = useCallback(
    async (point: { time: number; freq: number }) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api?.hit_test) {
        reportError('Connect', 'API not available')
        return
      }
      try {
        const result = await api.hit_test({ ...point, tolerance: 0.08 })
        if (result.status === 'ok' && result.id) {
          const current = connectQueueRef.current
          const next = current.includes(result.id) ? current : [...current, result.id]
          if (next.length === 2 && api?.merge_partials) {
            const mergeResult = await api.merge_partials({ first: next[0], second: next[1] })
            if (mergeResult.status === 'ok') {
              dispatch({ type: 'REMOVE_AND_ADD', removeIds: next, partial: toPartial(mergeResult.partial) })
            } else if (mergeResult.status === 'error') {
              reportError('Connect', mergeResult.message ?? 'Merge failed')
            }
            connectQueueRef.current = []
          } else {
            connectQueueRef.current = next
          }
        } else if (result.status === 'error') {
          reportError('Connect', result.message ?? 'Hit test failed')
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unexpected error'
        reportError('Connect', message)
      }
    },
    [reportError]
  )

  const toggleMute = useCallback(async () => {
    if (state.selection.length !== 1) return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.toggle_mute) {
      reportError('Mute', 'API not available')
      return
    }
    try {
      const result = await api.toggle_mute({ id: state.selection[0] })
      if (result.status === 'ok') {
        dispatch({ type: 'UPDATE', partial: toPartial(result.partial) })
      } else if (result.status === 'error') {
        reportError('Mute', result.message ?? 'Mute failed')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Mute', message)
    }
  }, [state.selection, reportError])

  const deleteSelected = useCallback(async () => {
    if (state.selection.length === 0) return
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.delete_partials) {
      reportError('Delete', 'API not available')
      return
    }
    try {
      const result = await api.delete_partials({ ids: state.selection })
      if (result.status === 'ok') {
        dispatch({ type: 'DELETE', ids: state.selection })
      } else if (result.status === 'error') {
        reportError('Delete', result.message ?? 'Delete failed')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Delete', message)
    }
  }, [state.selection, reportError])

  const undo = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.undo) {
      reportError('Undo', 'API not available')
      return
    }
    try {
      const result = await api.undo()
      if (result.status === 'ok') {
        dispatch({ type: 'SET', partials: result.partials.map(toPartial) })
      } else if (result.status === 'error') {
        reportError('Undo', result.message ?? 'Undo failed')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Undo', message)
    }
  }, [reportError])

  const redo = useCallback(async () => {
    const api = isPywebviewApiAvailable() ? pywebviewApi : null
    if (!api?.redo) {
      reportError('Redo', 'API not available')
      return
    }
    try {
      const result = await api.redo()
      if (result.status === 'ok') {
        dispatch({ type: 'SET', partials: result.partials.map(toPartial) })
      } else if (result.status === 'error') {
        reportError('Redo', result.message ?? 'Redo failed')
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected error'
      reportError('Redo', message)
    }
  }, [reportError])

  return {
    partials: state.items,
    selection: state.selection,
    isSnapping: state.isSnapping,
    setPartials,
    setSelection,
    trace,
    erase,
    selectInBox,
    hitTest,
    updatePartial,
    connectPick,
    toggleMute,
    deleteSelected,
    undo,
    redo,
  }
}
