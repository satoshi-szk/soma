import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AnalysisSettings, SpectrogramPreview } from '../../../app/types'

type ViewportImageItem = { preview: SpectrogramPreview; image: ImageBitmap }

export function useViewportImageCache(
  viewportPreviews: SpectrogramPreview[] | null,
  settings: AnalysisSettings
): {
  buildViewportKey: (current: SpectrogramPreview) => string
  viewportImages: ViewportImageItem[]
} {
  const [viewportImageCache, setViewportImageCache] = useState(new Map<string, ImageBitmap>())
  const viewportWorkerRef = useRef<Worker | null>(null)
  const viewportPendingRef = useRef(new Set<string>())
  const desiredViewportKeysRef = useRef<Set<string>>(new Set())

  const buildViewportKey = useCallback(
    (current: SpectrogramPreview) =>
      [
        current.time_start,
        current.time_end,
        current.freq_min,
        current.freq_max,
        current.width,
        current.height,
        settings.color_map,
        settings.brightness,
        settings.contrast,
      ].join('|'),
    [settings],
  )

  useEffect(() => {
    if (typeof OffscreenCanvas === 'undefined') {
      console.warn('[ViewportWorker] OffscreenCanvas not available, viewport rendering disabled')
      return
    }
    const worker = new Worker(new URL('../../../workers/viewportRenderer.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (event) => {
      const payload = event.data as {
        type: 'result' | 'error'
        key: string
        bitmap?: ImageBitmap
        message?: string
      }
      if (payload.type === 'result') {
        const bitmap = payload.bitmap
        if (!bitmap) return
        viewportPendingRef.current.delete(payload.key)
        setViewportImageCache((prev) => {
          const next = new Map(prev)
          next.set(payload.key, bitmap)
          for (const [key, image] of next) {
            if (!desiredViewportKeysRef.current.has(key)) {
              image.close()
              next.delete(key)
            }
          }
          return next
        })
      } else if (payload.type === 'error') {
        viewportPendingRef.current.delete(payload.key)
        console.warn(`[ViewportWorker] ${payload.message ?? 'render failed'}`)
      }
    }
    viewportWorkerRef.current = worker
    return () => {
      worker.terminate()
      viewportWorkerRef.current = null
      setViewportImageCache((prev) => {
        for (const bitmap of prev.values()) {
          bitmap.close()
        }
        return new Map()
      })
    }
  }, [])

  useEffect(() => {
    const previews = viewportPreviews ?? []
    const desiredKeys = new Set<string>()

    for (const current of previews) {
      if (current.width <= 0 || current.height <= 0) {
        console.warn('[Workspace] Invalid viewport preview dimensions:', current.width, 'x', current.height)
        continue
      }
      const expectedLength = current.width * current.height
      if (current.data.length !== expectedLength) {
        console.error(
          '[Workspace] Viewport preview data length mismatch: expected',
          expectedLength,
          'got',
          current.data.length
        )
        continue
      }
      const key = buildViewportKey(current)
      desiredKeys.add(key)

      if (viewportImageCache.has(key) || viewportPendingRef.current.has(key)) {
        continue
      }

      const worker = viewportWorkerRef.current
      if (worker) {
        viewportPendingRef.current.add(key)
        worker.postMessage({
          type: 'render',
          key,
          width: current.width,
          height: current.height,
          data: current.data,
          color_map: settings.color_map,
          brightness: settings.brightness,
          contrast: settings.contrast,
        })
      }
    }

    desiredViewportKeysRef.current = desiredKeys
    for (const key of viewportPendingRef.current) {
      if (!desiredKeys.has(key)) {
        viewportPendingRef.current.delete(key)
      }
    }

    const cleanupTimer = window.setTimeout(() => {
      setViewportImageCache((prev) => {
        if (desiredViewportKeysRef.current.size === 0 && prev.size === 0) return prev
        let updated = false
        const next = new Map(prev)
        for (const [key, image] of next) {
          if (!desiredViewportKeysRef.current.has(key)) {
            image.close()
            next.delete(key)
            updated = true
          }
        }
        return updated ? next : prev
      })
    }, 0)

    return () => {
      window.clearTimeout(cleanupTimer)
    }
  }, [viewportPreviews, settings, buildViewportKey, viewportImageCache])

  const viewportImages = useMemo(() => {
    if (!viewportPreviews || viewportPreviews.length === 0) return []
    return viewportPreviews
      .map((current) => {
        const key = buildViewportKey(current)
        const image = viewportImageCache.get(key)
        if (!image) return null
        return { preview: current, image }
      })
      .filter((item): item is ViewportImageItem => item !== null)
  }, [viewportPreviews, viewportImageCache, buildViewportKey])

  return { buildViewportKey, viewportImages }
}

