import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import {
  ZOOM_X_MAX_PX_PER_SEC,
  ZOOM_X_MIN_PX_PER_SEC,
  ZOOM_X_STEP_RATIO,
} from '../app/constants'
import type { SpectrogramPreview } from '../app/types'

type ViewportParams = {
  timeStart: number
  timeEnd: number
  freqMin: number
  freqMax: number
}

type ViewportCacheEntry = {
  preview: SpectrogramPreview
  quality: 'low' | 'high'
  receivedAt: number
  sourceDuration: number
  sourceFreqMin: number
  sourceFreqMax: number
}

function areParamsSimilar(a: ViewportParams | null, b: ViewportParams): boolean {
  if (!a) return false
  const threshold = 0.02
  const timeDiff = Math.abs(a.timeStart - b.timeStart) + Math.abs(a.timeEnd - b.timeEnd)
  const freqDiff = Math.abs(a.freqMin - b.freqMin) / b.freqMin + Math.abs(a.freqMax - b.freqMax) / b.freqMax
  return timeDiff < threshold && freqDiff < threshold
}

export function useViewport(preview: SpectrogramPreview | null) {
  const [zoomXState, setZoomXState] = useState<number | null>(null)
  const [zoomY, setZoomY] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [viewportCache, setViewportCache] = useState<ViewportCacheEntry[]>([])
  const viewportDebounceRef = useRef<number | null>(null)
  const previewThrottleRef = useRef<{
    lastApplied: number
    timerId: number | null
    pending: SpectrogramPreview | null
    pendingQuality: 'low' | 'high'
  }>({ lastApplied: 0, timerId: null, pending: null, pendingQuality: 'low' })
  const lastRequestedParams = useRef<ViewportParams | null>(null)

  const baseZoomX = useMemo(() => {
    if (!preview) return 1
    return stageSize.width / Math.max(preview.duration_sec, 1e-6)
  }, [preview, stageSize])

  const zoomX = zoomXState ?? baseZoomX

  const clampZoomX = useCallback(
    (value: number) => Math.min(ZOOM_X_MAX_PX_PER_SEC, Math.max(ZOOM_X_MIN_PX_PER_SEC, value)),
    []
  )

  const setZoomXClamped = useCallback(
    (value: number | ((prev: number) => number)) => {
      setZoomXState((prev) => {
        const current = prev ?? baseZoomX
        const next = typeof value === 'function' ? value(current) : value
        return clampZoomX(next)
      })
    },
    [baseZoomX, clampZoomX]
  )

  const viewportPreviews = useMemo(() => {
    if (!preview || (zoomX <= baseZoomX && zoomY <= 1)) return null
    const currentDuration = preview.duration_sec
    const currentFreqMin = preview.freq_min
    const currentFreqMax = preview.freq_max
    return viewportCache
      .filter(
        (entry) =>
          entry.sourceDuration === currentDuration &&
          entry.sourceFreqMin === currentFreqMin &&
          entry.sourceFreqMax === currentFreqMax
      )
      .slice()
      .sort((a, b) => {
        if (a.quality === b.quality) return a.receivedAt - b.receivedAt
        return a.quality === 'low' ? -1 : 1
      })
      .map((entry) => entry.preview)
  }, [preview, zoomX, zoomY, viewportCache, baseZoomX])

  // Request viewport preview on zoom/pan change
  useEffect(() => {
    if (!preview || (zoomX <= baseZoomX && zoomY <= 1)) {
      lastRequestedParams.current = null
      return
    }

    if (viewportDebounceRef.current) {
      window.clearTimeout(viewportDebounceRef.current)
    }

    viewportDebounceRef.current = window.setTimeout(async () => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api) return

      const duration = preview.duration_sec
      const freqMin = preview.freq_min
      const freqMax = preview.freq_max

      // Calculate visible viewport
      const visibleTimeStart = Math.max(0, -pan.x / zoomX)
      const visibleTimeEnd = Math.min(duration, visibleTimeStart + stageSize.width / zoomX)
      const logMin = Math.log(freqMin)
      const logMax = Math.log(freqMax)
      const visibleFreqMax = Math.exp(logMax - Math.max(0, -pan.y / (stageSize.height * zoomY)) * (logMax - logMin))
      const visibleFreqMin = Math.exp(
        logMax - Math.min(1, (stageSize.height - pan.y) / (stageSize.height * zoomY)) * (logMax - logMin)
      )

      const params: ViewportParams = {
        timeStart: visibleTimeStart,
        timeEnd: visibleTimeEnd,
        freqMin: Math.max(freqMin, visibleFreqMin),
        freqMax: Math.min(freqMax, visibleFreqMax),
      }

      // Skip if params are similar to last request
      if (areParamsSimilar(lastRequestedParams.current, params)) {
        return
      }

      lastRequestedParams.current = params

      const result = await api.request_viewport_preview({
        time_start: params.timeStart,
        time_end: params.timeEnd,
        freq_min: params.freqMin,
        freq_max: params.freqMax,
        width: Math.round(stageSize.width),
        height: Math.round(stageSize.height),
      })

      // fire & forget: result status only indicates request acceptance
      void result
    }, 500)

    return () => {
      if (viewportDebounceRef.current) {
        window.clearTimeout(viewportDebounceRef.current)
      }
    }
  }, [zoomX, zoomY, pan, stageSize, preview, baseZoomX])

  // Subscribe viewport preview events (push)
  useEffect(() => {
    const cleanupState = previewThrottleRef.current
    const applyPreview = (next: SpectrogramPreview, quality: 'low' | 'high') => {
      if (!preview) return
      const entry: ViewportCacheEntry = {
        preview: next,
        quality,
        receivedAt: Date.now(),
        sourceDuration: preview.duration_sec,
        sourceFreqMin: preview.freq_min,
        sourceFreqMax: preview.freq_max,
      }
      setViewportCache((prev) => {
        const hasHigh = prev.some(
          (item) =>
            item.quality === 'high' &&
            item.preview.time_start === next.time_start &&
            item.preview.time_end === next.time_end &&
            item.preview.freq_min === next.freq_min &&
            item.preview.freq_max === next.freq_max
        )
        if (quality === 'low' && hasHigh) {
          return prev
        }
        const filtered = quality === 'high'
          ? prev.filter(
              (item) =>
                !(
                  item.preview.time_start === next.time_start &&
                  item.preview.time_end === next.time_end &&
                  item.preview.freq_min === next.freq_min &&
                  item.preview.freq_max === next.freq_max
                )
            )
          : prev
        const nextItems = [...filtered, entry]
        const limit = 3
        if (nextItems.length <= limit) return nextItems
        return nextItems.slice(nextItems.length - limit)
      })
    }

    const schedulePreview = (next: SpectrogramPreview, quality: 'low' | 'high') => {
      const state = previewThrottleRef.current
      state.pending = next
      state.pendingQuality = quality
      const now = Date.now()
      const throttleMs = 1500
      const elapsed = now - state.lastApplied

      if (elapsed >= throttleMs) {
        if (state.timerId) {
          window.clearTimeout(state.timerId)
          state.timerId = null
        }
        state.lastApplied = now
        applyPreview(next, quality)
        return
      }

      if (state.timerId) {
        return
      }

      state.timerId = window.setTimeout(() => {
        state.timerId = null
        if (state.pending) {
          state.lastApplied = Date.now()
          applyPreview(state.pending, state.pendingQuality)
        }
      }, throttleMs - elapsed)
    }

    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail as unknown
      if (!detail || typeof detail !== 'object') return
      const payload = detail as Record<string, unknown>
      if (payload.type === 'spectrogram_preview_updated' && payload.kind === 'viewport') {
        const next = payload.preview as SpectrogramPreview | undefined
        const quality = payload.quality === 'high' ? 'high' : 'low'
        if (!next) return
        schedulePreview(next, quality)
      }
    }
    window.addEventListener('soma:event', handler)
    return () => {
      window.removeEventListener('soma:event', handler)
      const state = cleanupState
      if (state.timerId) {
        window.clearTimeout(state.timerId)
        state.timerId = null
      }
    }
  }, [preview])

  const zoomInX = useCallback(() => {
    setZoomXClamped((value) => value * ZOOM_X_STEP_RATIO)
  }, [setZoomXClamped])

  const zoomOutX = useCallback(() => {
    setZoomXClamped((value) => value / ZOOM_X_STEP_RATIO)
  }, [setZoomXClamped])

  const zoomInY = useCallback(() => {
    setZoomY((value) => Math.min(10, value + 1.0))
  }, [])

  const zoomOutY = useCallback(() => {
    setZoomY((value) => Math.max(0.5, value - 1.0))
  }, [])

  const resetView = useCallback(() => {
    setZoomXState(null)
    setZoomY(1)
    setPan({ x: 0, y: 0 })
    setViewportCache([])
  }, [])

  return {
    zoomX,
    zoomY,
    pan,
    stageSize,
    viewportPreviews,
    setZoomX: setZoomXClamped,
    setPan,
    setStageSize,
    zoomInX,
    zoomOutX,
    zoomInY,
    zoomOutY,
    resetView,
  }
}
