import { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import {
  ZOOM_X_MAX_PX_PER_SEC,
  ZOOM_X_MIN_PX_PER_SEC,
  ZOOM_X_STEP_RATIO,
  ZOOM_Y_MIN,
  ZOOM_Y_MAX,
  RULER_HEIGHT,
  AUTOMATION_LANE_HEIGHT,
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

type ReportError = (context: string, message: string) => void

export function useViewport(preview: SpectrogramPreview | null, reportError: ReportError) {
  const [zoomXState, setZoomXState] = useState<number | null>(null)
  const [zoomY, setZoomY] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [viewportCache, setViewportCache] = useState<ViewportCacheEntry[]>([])
  const interactionRef = useRef(false)
  const interactionTimerRef = useRef<number | null>(null)
  const previewThrottleRef = useRef<{
    lastApplied: number
    timerId: number | null
    pending: SpectrogramPreview | null
    pendingQuality: 'low' | 'high'
  }>({ lastApplied: 0, timerId: null, pending: null, pendingQuality: 'low' })
  const lastRequestedParams = useRef<ViewportParams | null>(null)
  const pendingParamsRef = useRef<ViewportParams | null>(null)
  const lastPreviewId = useRef<string | null>(null)

  const baseZoomX = useMemo(() => {
    if (!preview) return 1
    return stageSize.width / Math.max(preview.duration_sec, 1e-6)
  }, [preview, stageSize])

  const zoomX = zoomXState ?? baseZoomX

  // Spectrogram area height (excluding ruler and automation lane)
  const spectrogramAreaHeight = useMemo(
    () => Math.max(1, stageSize.height - AUTOMATION_LANE_HEIGHT - RULER_HEIGHT),
    [stageSize.height]
  )

  // Base scale for Y axis (maps preview height to spectrogram area)
  const baseScaleY = useMemo(() => {
    if (!preview) return 1
    return spectrogramAreaHeight / preview.height
  }, [preview, spectrogramAreaHeight])

  // Clamp pan to keep content within viewport bounds
  const clampPan = useCallback(
    (
      newPan: { x: number; y: number },
      currentZoomX: number,
      currentZoomY: number
    ): { x: number; y: number } => {
      if (!preview) return newPan

      const contentWidth = preview.duration_sec * currentZoomX
      const contentHeight = preview.height * baseScaleY * currentZoomY

      // If content is smaller than viewport, pin to origin (no empty space)
      const clampedX =
        contentWidth <= stageSize.width
          ? 0
          : Math.min(0, Math.max(-(contentWidth - stageSize.width), newPan.x))

      const clampedY =
        contentHeight <= spectrogramAreaHeight
          ? 0
          : Math.min(0, Math.max(-(contentHeight - spectrogramAreaHeight), newPan.y))

      return { x: clampedX, y: clampedY }
    },
    [preview, baseScaleY, stageSize.width, spectrogramAreaHeight]
  )

  // Generate a stable ID for the current preview source
  const currentPreviewId = useMemo(() => {
    if (!preview) return null
    return `${preview.duration_sec}-${preview.freq_min}-${preview.freq_max}`
  }, [preview])

  // Clear throttle state when preview source changes
  useLayoutEffect(() => {
    if (currentPreviewId !== lastPreviewId.current) {
      console.log('[useViewport] Preview source changed, clearing throttle state')
      lastPreviewId.current = currentPreviewId
      lastRequestedParams.current = null

      const state = previewThrottleRef.current
      if (state.timerId) {
        window.clearTimeout(state.timerId)
        state.timerId = null
      }
      state.pending = null
      state.lastApplied = 0
    }
  }, [currentPreviewId])

  const clampZoomX = useCallback(
    (value: number) => Math.min(ZOOM_X_MAX_PX_PER_SEC, Math.max(ZOOM_X_MIN_PX_PER_SEC, value)),
    []
  )

  const clampZoomY = useCallback(
    (value: number) => Math.min(ZOOM_Y_MAX, Math.max(ZOOM_Y_MIN, value)),
    []
  )

  // Wrapped setPan that always applies clamping
  const setPanClamped = useCallback(
    (value: { x: number; y: number } | ((prev: { x: number; y: number }) => { x: number; y: number })) => {
      setPan((prev) => {
        const next = typeof value === 'function' ? value(prev) : value
        return clampPan(next, zoomX, zoomY)
      })
    },
    [clampPan, zoomX, zoomY]
  )

  const setZoomXClamped = useCallback(
    (value: number | ((prev: number) => number), targetPan?: { x: number; y: number }) => {
      setZoomXState((prev) => {
        const current = prev ?? baseZoomX
        const next = typeof value === 'function' ? value(current) : value
        const clamped = clampZoomX(next)
        // Use targetPan if provided, otherwise re-clamp current pan
        setPan((prevPan) => clampPan(targetPan ?? prevPan, clamped, zoomY))
        return clamped
      })
    },
    [baseZoomX, clampZoomX, clampPan, zoomY]
  )

  const viewportPreviews = useMemo(() => {
    if (!preview || (zoomX <= baseZoomX && zoomY <= 1)) return null
    const currentDuration = preview.duration_sec
    const currentFreqMin = preview.freq_min
    const currentFreqMax = preview.freq_max

    // Filter cache entries that match current preview source
    const validEntries = viewportCache.filter(
      (entry) =>
        entry.sourceDuration === currentDuration &&
        entry.sourceFreqMin === currentFreqMin &&
        entry.sourceFreqMax === currentFreqMax
    )

    // Only log once when stale entries are first detected
    // (filtering happens on every render, so avoid spam)

    return validEntries
      .slice()
      .sort((a, b) => a.receivedAt - b.receivedAt)
      .map((entry) => entry.preview)
  }, [preview, zoomX, zoomY, viewportCache, baseZoomX])

  const sendViewportRequest = useCallback(
    async (params: ViewportParams) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api) return

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
        height: Math.round(spectrogramAreaHeight),
      })

      if (result.status === 'error') {
        reportError('Viewport', result.message ?? 'Failed to request viewport preview')
      }
    },
    [reportError, stageSize.width, spectrogramAreaHeight]
  )

  // Request viewport preview on zoom/pan change (after interaction stops)
  useEffect(() => {
    if (!preview || (zoomX <= baseZoomX && zoomY <= 1)) {
      lastRequestedParams.current = null
      return
    }

    const duration = preview.duration_sec
    const freqMin = preview.freq_min
    const freqMax = preview.freq_max

    // Calculate visible viewport
    const visibleTimeStart = Math.max(0, -pan.x / zoomX)
    const visibleTimeEnd = Math.min(duration, visibleTimeStart + stageSize.width / zoomX)
    const logMin = Math.log(freqMin)
    const logMax = Math.log(freqMax)
    const visibleFreqMax = Math.exp(
      logMax - Math.max(0, -pan.y / (spectrogramAreaHeight * zoomY)) * (logMax - logMin)
    )
    const visibleFreqMin = Math.exp(
      logMax -
        Math.min(1, (spectrogramAreaHeight - pan.y) / (spectrogramAreaHeight * zoomY)) * (logMax - logMin)
    )

    pendingParamsRef.current = {
      timeStart: visibleTimeStart,
      timeEnd: visibleTimeEnd,
      freqMin: Math.max(freqMin, visibleFreqMin),
      freqMax: Math.min(freqMax, visibleFreqMax),
    }

    interactionRef.current = true
    if (interactionTimerRef.current) {
      window.clearTimeout(interactionTimerRef.current)
    }
    interactionTimerRef.current = window.setTimeout(() => {
      interactionRef.current = false
      const params = pendingParamsRef.current
      if (params) {
        void sendViewportRequest(params)
      }
    }, 300)

    return () => {
      if (interactionTimerRef.current) {
        window.clearTimeout(interactionTimerRef.current)
      }
    }
  }, [zoomX, zoomY, pan, stageSize.width, spectrogramAreaHeight, preview, baseZoomX, sendViewportRequest])

  // Subscribe viewport preview events (push)
  useEffect(() => {
    const cleanupState = previewThrottleRef.current
    const applyPreview = (next: SpectrogramPreview, quality: 'low' | 'high') => {
      if (!preview) {
        console.warn('[useViewport] applyPreview called but preview is null, skipping')
        return
      }

      // Validate that the incoming preview is compatible with current source
      // Check if time ranges are within valid bounds
      if (next.time_start < 0 || next.time_end > preview.duration_sec + 0.01) {
        console.warn(
          '[useViewport] Invalid preview time range:',
          next.time_start,
          '-',
          next.time_end,
          'vs duration:',
          preview.duration_sec
        )
        return
      }

      // Sanity check: preview dimensions should be reasonable
      if (next.width <= 0 || next.height <= 0 || next.data.length !== next.width * next.height) {
        console.error(
          '[useViewport] Invalid preview dimensions:',
          next.width,
          'x',
          next.height,
          'data length:',
          next.data.length
        )
        return
      }

      const entry: ViewportCacheEntry = {
        preview: next,
        quality,
        receivedAt: Date.now(),
        sourceDuration: preview.duration_sec,
        sourceFreqMin: preview.freq_min,
        sourceFreqMax: preview.freq_max,
      }
      setViewportCache((prev) => {
        // Skip if we already have a high-quality version of the same region
        const hasHigh = prev.some(
          (item) =>
            item.quality === 'high' &&
            Math.abs(item.preview.time_start - next.time_start) < 0.001 &&
            Math.abs(item.preview.time_end - next.time_end) < 0.001 &&
            Math.abs(item.preview.freq_min - next.freq_min) < 0.1 &&
            Math.abs(item.preview.freq_max - next.freq_max) < 0.1
        )
        if (quality === 'low' && hasHigh) {
          return prev
        }
        // Remove old entries for the same region when adding high-quality version
        const filtered = quality === 'high'
          ? prev.filter(
              (item) =>
                !(
                  Math.abs(item.preview.time_start - next.time_start) < 0.001 &&
                  Math.abs(item.preview.time_end - next.time_end) < 0.001 &&
                  Math.abs(item.preview.freq_min - next.freq_min) < 0.1 &&
                  Math.abs(item.preview.freq_max - next.freq_max) < 0.1
                )
            )
          : prev
        const nextItems = [...filtered, entry]
        const limit = 3
        if (nextItems.length <= limit) return nextItems
        // Keep only the most recent entries
        return nextItems.slice(nextItems.length - limit)
      })
    }

    const schedulePreview = (next: SpectrogramPreview, quality: 'low' | 'high') => {
      // Validate against current preview before scheduling
      if (!preview) {
        console.warn('[useViewport] schedulePreview called but preview is null')
        return
      }

      // Check if incoming preview is for the current source
      if (next.time_end > preview.duration_sec + 0.01) {
        console.warn(
          '[useViewport] Ignoring stale viewport preview (time_end:',
          next.time_end,
          '> duration:',
          preview.duration_sec,
          ')'
        )
        return
      }

      if (interactionRef.current) {
        return
      }

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

      const currentPreviewIdAtSchedule = currentPreviewId
      state.timerId = window.setTimeout(() => {
        state.timerId = null
        // Double-check preview hasn't changed before applying
        if (state.pending && currentPreviewIdAtSchedule === lastPreviewId.current) {
          state.lastApplied = Date.now()
          applyPreview(state.pending, state.pendingQuality)
        } else if (state.pending) {
          console.log('[useViewport] Discarding pending preview due to source change')
          state.pending = null
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
      if (payload.type === 'spectrogram_preview_error' && payload.kind === 'viewport') {
        const message = typeof payload.message === 'string' ? payload.message : 'Viewport preview failed.'
        reportError('Viewport', message)
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
  }, [preview, currentPreviewId, reportError])

  const zoomInX = useCallback(() => {
    setZoomXClamped((value) => value * ZOOM_X_STEP_RATIO)
  }, [setZoomXClamped])

  const zoomOutX = useCallback(() => {
    setZoomXClamped((value) => value / ZOOM_X_STEP_RATIO)
  }, [setZoomXClamped])

  const zoomInY = useCallback(() => {
    if (!preview) return
    const oldZoom = zoomY
    const newZoom = clampZoomY(oldZoom + 1.0)
    if (newZoom === oldZoom) return

    // Keep viewport center at the same frequency
    const centerY = spectrogramAreaHeight / 2
    const oldContentY = (centerY - pan.y) / (baseScaleY * oldZoom)
    const newPanY = centerY - oldContentY * baseScaleY * newZoom

    setZoomY(newZoom)
    setPan((prev) => clampPan({ x: prev.x, y: newPanY }, zoomX, newZoom))
  }, [preview, zoomY, zoomX, pan.y, baseScaleY, spectrogramAreaHeight, clampZoomY, clampPan])

  const zoomOutY = useCallback(() => {
    if (!preview) return
    const oldZoom = zoomY
    const newZoom = clampZoomY(oldZoom - 1.0)
    if (newZoom === oldZoom) return

    // Keep viewport center at the same frequency
    const centerY = spectrogramAreaHeight / 2
    const oldContentY = (centerY - pan.y) / (baseScaleY * oldZoom)
    const newPanY = centerY - oldContentY * baseScaleY * newZoom

    setZoomY(newZoom)
    setPan((prev) => clampPan({ x: prev.x, y: newPanY }, zoomX, newZoom))
  }, [preview, zoomY, zoomX, pan.y, baseScaleY, spectrogramAreaHeight, clampZoomY, clampPan])

  const resetView = useCallback(() => {
    setZoomXState(null)
    setZoomY(ZOOM_Y_MIN)
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
    setPan: setPanClamped,
    setStageSize,
    zoomInX,
    zoomOutX,
    zoomInY,
    zoomOutY,
    resetView,
  }
}
