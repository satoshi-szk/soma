import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { SpectrogramPreview } from '../../../app/types'

type ViewportImageItem = { preview: SpectrogramPreview; image: HTMLImageElement }

export function useViewportImageCache(viewportPreviews: SpectrogramPreview[] | null): {
  buildViewportKey: (current: SpectrogramPreview) => string
  viewportImages: ViewportImageItem[]
} {
  const [viewportImageCache, setViewportImageCache] = useState(new Map<string, HTMLImageElement>())
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
        current.image_path ?? '',
      ].join('|'),
    [],
  )

  useEffect(() => {
    const previews = viewportPreviews ?? []
    const desiredKeys = new Set<string>()

    for (const current of previews) {
      if (current.width <= 0 || current.height <= 0) {
        console.warn('[Workspace] Invalid viewport preview dimensions:', current.width, 'x', current.height)
        continue
      }
      if (!current.image_path) {
        console.warn('[Workspace] Missing image_path for viewport preview')
        continue
      }
      const key = buildViewportKey(current)
      desiredKeys.add(key)

      if (viewportImageCache.has(key) || viewportPendingRef.current.has(key)) {
        continue
      }

      viewportPendingRef.current.add(key)
      const image = new window.Image()
      image.onload = () => {
        viewportPendingRef.current.delete(key)
        setViewportImageCache((prev) => {
          const next = new Map(prev)
          next.set(key, image)
          for (const staleKey of next.keys()) {
            if (!desiredViewportKeysRef.current.has(staleKey)) {
              next.delete(staleKey)
            }
          }
          return next
        })
      }
      image.onerror = () => {
        viewportPendingRef.current.delete(key)
        console.warn('[Workspace] Failed to load viewport image:', current.image_path)
      }
      image.src = current.image_path
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
        for (const key of next.keys()) {
          if (!desiredViewportKeysRef.current.has(key)) {
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
  }, [viewportPreviews, buildViewportKey, viewportImageCache])

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
