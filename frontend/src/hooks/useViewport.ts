import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import type { SpectrogramPreview, ViewportPreview } from '../app/types'

export function useViewport(preview: SpectrogramPreview | null) {
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [viewportPreviewData, setViewportPreviewData] = useState<ViewportPreview | null>(null)
  const [viewportRequestId, setViewportRequestId] = useState<string | null>(null)
  const viewportDebounceRef = useRef<number | null>(null)

  // Derive viewportPreview: null when not zoomed
  const viewportPreview = useMemo(() => {
    if (!preview || zoom <= 1) return null
    return viewportPreviewData
  }, [preview, zoom, viewportPreviewData])

  // Request viewport preview on zoom/pan change
  useEffect(() => {
    if (!preview || zoom <= 1) {
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
      const visibleTimeStart = Math.max(0, (-pan.x / (stageSize.width * zoom)) * duration)
      const visibleTimeEnd = Math.min(duration, visibleTimeStart + duration / zoom)
      const logMin = Math.log(freqMin)
      const logMax = Math.log(freqMax)
      const visibleFreqMax = Math.exp(logMax - Math.max(0, -pan.y / (stageSize.height * zoom)) * (logMax - logMin))
      const visibleFreqMin = Math.exp(
        logMax - Math.min(1, (stageSize.height - pan.y) / (stageSize.height * zoom)) * (logMax - logMin)
      )

      const result = await api.request_viewport_preview({
        time_start: visibleTimeStart,
        time_end: visibleTimeEnd,
        freq_min: Math.max(freqMin, visibleFreqMin),
        freq_max: Math.min(freqMax, visibleFreqMax),
        width: Math.round(stageSize.width),
        height: Math.round(stageSize.height),
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
  }, [zoom, pan, stageSize, preview])

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
    const interval = window.setInterval(poll, 200)
    return () => {
      alive = false
      window.clearInterval(interval)
    }
  }, [viewportRequestId])

  const zoomIn = useCallback(() => {
    setZoom((value) => Math.min(4, value + 0.2))
  }, [])

  const zoomOut = useCallback(() => {
    setZoom((value) => Math.max(0.5, value - 0.2))
  }, [])

  const resetView = useCallback(() => {
    setZoom(1)
    setPan({ x: 0, y: 0 })
  }, [])

  return {
    zoom,
    pan,
    stageSize,
    viewportPreview,
    setZoom,
    setPan,
    setStageSize,
    zoomIn,
    zoomOut,
    resetView,
  }
}
