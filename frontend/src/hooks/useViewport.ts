import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import type { SpectrogramPreview, ViewportPreview } from '../app/types'

type ViewportParams = {
  timeStart: number
  timeEnd: number
  freqMin: number
  freqMax: number
}

function areParamsSimilar(a: ViewportParams | null, b: ViewportParams): boolean {
  if (!a) return false
  const threshold = 0.02
  const timeDiff = Math.abs(a.timeStart - b.timeStart) + Math.abs(a.timeEnd - b.timeEnd)
  const freqDiff = Math.abs(a.freqMin - b.freqMin) / b.freqMin + Math.abs(a.freqMax - b.freqMax) / b.freqMax
  return timeDiff < threshold && freqDiff < threshold
}

export function useViewport(preview: SpectrogramPreview | null) {
  const [zoomX, setZoomX] = useState(1)
  const [zoomY, setZoomY] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [viewportPreviewData, setViewportPreviewData] = useState<ViewportPreview | null>(null)
  const [viewportRequestId, setViewportRequestId] = useState<string | null>(null)
  const viewportDebounceRef = useRef<number | null>(null)
  const lastRequestedParams = useRef<ViewportParams | null>(null)

  // Derive viewportPreview: null when not zoomed
  const viewportPreview = useMemo(() => {
    if (!preview || (zoomX <= 1 && zoomY <= 1)) return null
    return viewportPreviewData
  }, [preview, zoomX, zoomY, viewportPreviewData])

  // Request viewport preview on zoom/pan change
  useEffect(() => {
    if (!preview || (zoomX <= 1 && zoomY <= 1)) {
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
      const visibleTimeStart = Math.max(0, (-pan.x / (stageSize.width * zoomX)) * duration)
      const visibleTimeEnd = Math.min(duration, visibleTimeStart + duration / zoomX)
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
        width: Math.round(stageSize.width * 0.5),
        height: Math.round(stageSize.height * 0.5),
      })

      if (result.status === 'processing') {
        setViewportRequestId(result.request_id)
      }
    }, 500)

    return () => {
      if (viewportDebounceRef.current) {
        window.clearTimeout(viewportDebounceRef.current)
      }
    }
  }, [zoomX, zoomY, pan, stageSize, preview])

  // Poll viewport preview status
  useEffect(() => {
    if (!viewportRequestId) return
    let alive = true

    const poll = async () => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api || !alive) return
      const result = await api.viewport_preview_status()
      if (!alive || result?.status !== 'ok') return
      if (result.request_id !== viewportRequestId) return

      if (result.state === 'ready' && result.preview) {
        setViewportPreviewData(result.preview)
        setViewportRequestId(null)
      } else if (result.state === 'error' || result.state === 'cancelled') {
        setViewportRequestId(null)
      }
    }

    void poll()
    const interval = window.setInterval(poll, 400)
    return () => {
      alive = false
      window.clearInterval(interval)
    }
  }, [viewportRequestId])

  const zoomInX = useCallback(() => {
    setZoomX((value) => Math.min(4, value + 0.2))
  }, [])

  const zoomOutX = useCallback(() => {
    setZoomX((value) => Math.max(0.5, value - 0.2))
  }, [])

  const zoomInY = useCallback(() => {
    setZoomY((value) => Math.min(4, value + 0.2))
  }, [])

  const zoomOutY = useCallback(() => {
    setZoomY((value) => Math.max(0.5, value - 0.2))
  }, [])

  const resetView = useCallback(() => {
    setZoomX(1)
    setZoomY(1)
    setPan({ x: 0, y: 0 })
  }, [])

  return {
    zoomX,
    zoomY,
    pan,
    stageSize,
    viewportPreview,
    setZoomX,
    setPan,
    setStageSize,
    zoomInX,
    zoomOutX,
    zoomInY,
    zoomOutY,
    resetView,
  }
}
