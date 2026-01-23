import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent } from 'react'
import { Circle, Group, Layer, Line, Rect, Shape, Stage, Image as KonvaImage, Text } from 'react-konva'
import { ZOOM_X_MAX_PX_PER_SEC, ZOOM_X_MIN_PX_PER_SEC } from '../app/constants'
import type { AnalysisSettings, Partial, PartialPoint, SpectrogramPreview, ToolId } from '../app/types'
import { mapColor } from '../app/utils'
import { SelectionHud } from './SelectionHud'
import type { KonvaEventObject } from 'konva/lib/Node'
import type { Context } from 'konva/lib/Context'

const hexToRgba = (hex: string, alpha: number) => {
  const cleaned = hex.trim().replace('#', '')
  if (!/^[0-9a-fA-F]{6}$/.test(cleaned)) {
    return `rgba(248, 209, 154, ${alpha})`
  }
  const r = parseInt(cleaned.slice(0, 2), 16)
  const g = parseInt(cleaned.slice(2, 4), 16)
  const b = parseInt(cleaned.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export type WorkspaceProps = {
  preview: SpectrogramPreview | null
  viewportPreviews: SpectrogramPreview[] | null
  settings: AnalysisSettings
  partials: Partial[]
  selectedIds: string[]
  selectedInfo: Partial | null
  activeTool: ToolId
  analysisState: 'idle' | 'analyzing' | 'error'
  isSnapping: boolean
  zoomX: number
  zoomY: number
  pan: { x: number; y: number }
  playbackPosition: number
  onZoomXChange: (zoom: number) => void
  onPanChange: (pan: { x: number; y: number }) => void
  onStageSizeChange: (size: { width: number; height: number }) => void
  onTraceCommit: (trace: Array<[number, number]>) => Promise<boolean>
  onEraseCommit: (trace: Array<[number, number]>) => void
  onSelectBoxCommit: (selection: { time_start: number; time_end: number; freq_start: number; freq_end: number }) => void
  onHitTestCommit: (point: { time: number; freq: number }) => void
  onUpdatePartial: (id: string, points: PartialPoint[]) => void
  onConnectPick: (point: { time: number; freq: number }) => void
  onOpenAudio: () => void
  onOpenAudioPath: (path: string) => void
  onOpenAudioFile: (file: File) => void
  allowDrop: boolean
  onCursorMove: (cursor: { time: number; freq: number; amp: number | null }) => void
  onPartialMute: () => void
  onPartialDelete: () => void
  onZoomInY: () => void
  onZoomOutY: () => void
}

export function Workspace({
  preview,
  viewportPreviews,
  settings,
  partials,
  selectedIds,
  selectedInfo,
  activeTool,
  analysisState,
  isSnapping,
  zoomX,
  zoomY,
  pan,
  playbackPosition,
  onZoomXChange,
  onPanChange,
  onStageSizeChange,
  onTraceCommit,
  onEraseCommit,
  onSelectBoxCommit,
  onHitTestCommit,
  onUpdatePartial,
  onConnectPick,
  onOpenAudio,
  onOpenAudioPath,
  onOpenAudioFile,
  allowDrop,
  onCursorMove,
  onPartialMute,
  onPartialDelete,
  onZoomInY,
  onZoomOutY,
}: WorkspaceProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [tracePath, setTracePath] = useState<PartialPoint[]>([])
  const [isTracing, setIsTracing] = useState(false)
  const [committedTrace, setCommittedTrace] = useState<PartialPoint[]>([])
  const [selectionBox, setSelectionBox] = useState<null | { x: number; y: number; w: number; h: number }>(null)
  const [draggedPartial, setDraggedPartial] = useState<Partial | null>(null)
  const [hudPosition, setHudPosition] = useState({ x: 16, y: 16 })
  const [isDragActive, setIsDragActive] = useState(false)
  const [viewportImageVersion, setViewportImageVersion] = useState(0)
  const viewportWorkerRef = useRef<Worker | null>(null)
  const viewportImageCacheRef = useRef(new Map<string, ImageBitmap | HTMLCanvasElement>())
  const viewportPendingRef = useRef(new Set<string>())
  const perfRef = useRef({
    lastFrame: 0,
    lastLog: 0,
    activeUntil: 0,
    rafId: 0,
  })
  const automationLaneHeight = 120
  const automationPadding = { top: 18, bottom: 16 }

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setStageSize({ width, height })
        onStageSizeChange({ width, height })
      }
    })
    if (containerRef.current) {
      observer.observe(containerRef.current)
    }
    return () => observer.disconnect()
  }, [onStageSizeChange])

  useEffect(() => {
    if (typeof OffscreenCanvas === 'undefined') {
      console.warn('[ViewportWorker] OffscreenCanvas not available, falling back to main thread')
      return
    }
    const worker = new Worker(new URL('../workers/viewportRenderer.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (event) => {
      const payload = event.data as {
        type: 'result' | 'error'
        key: string
        bitmap?: ImageBitmap
        message?: string
      }
      if (payload.type === 'result' && payload.bitmap) {
        viewportPendingRef.current.delete(payload.key)
        viewportImageCacheRef.current.set(payload.key, payload.bitmap)
        setViewportImageVersion((value) => value + 1)
      } else if (payload.type === 'error') {
        viewportPendingRef.current.delete(payload.key)
        console.warn(`[ViewportWorker] ${payload.message ?? 'render failed'}`)
      }
    }
    viewportWorkerRef.current = worker
    return () => {
      worker.terminate()
      viewportWorkerRef.current = null
    }
  }, [])

  const resolveDroppedPath = useCallback((dataTransfer: DataTransfer | null) => {
    const directFile = dataTransfer?.files?.[0]
    if (directFile) {
      const candidate = (directFile as File & { path?: string }).path
      if (candidate) return candidate
    }

    const items = dataTransfer?.items
    if (items) {
      for (const item of Array.from(items)) {
        if (item.kind !== 'file') continue
        const file = item.getAsFile()
        const candidate = (file as File & { path?: string } | null)?.path
        if (candidate) return candidate
      }
    }

    const uriList = dataTransfer?.getData('text/uri-list')
    if (uriList) {
      const entry = uriList
        .split('\n')
        .map((line) => line.trim())
        .find((line) => line && !line.startsWith('#'))
      if (entry) {
        try {
          const url = new URL(entry)
          if (url.protocol === 'file:') {
            return decodeURIComponent(url.pathname)
          }
        } catch {
          // Ignore invalid URI data.
        }
      }
    }

    return null
  }, [])

  const isEventInsideContainer = useCallback((event: Event) => {
    const container = containerRef.current
    if (!container) return false
    const withPath = event as Event & { composedPath?: () => EventTarget[] }
    if (typeof withPath.composedPath === 'function') {
      return withPath.composedPath().includes(container)
    }
    return container.contains(event.target as Node)
  }, [])

  const handleDragOver = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'copy'
    },
    [allowDrop]
  )

  const handleDragEnter = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      setIsDragActive(true)
    },
    [allowDrop]
  )

  const handleDragLeave = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      setIsDragActive(false)
    },
    [allowDrop]
  )

  const handleDrop = useCallback(
    (event: DragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      setIsDragActive(false)
      const filePath = resolveDroppedPath(event.dataTransfer)
      if (filePath) {
        onOpenAudioPath(filePath)
        return
      }
      const fallbackFile = event.dataTransfer?.files?.[0] ?? null
      if (!fallbackFile) return
      onOpenAudioFile(fallbackFile)
    },
    [allowDrop, onOpenAudioPath, onOpenAudioFile, resolveDroppedPath]
  )

  useEffect(() => {
    if (!allowDrop) return

    const handleWindowDragOver = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'copy'
      }
    }

    const handleWindowDragEnter = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      setIsDragActive(true)
    }

    const handleWindowDragLeave = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      setIsDragActive(false)
    }

    const handleWindowDrop = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      setIsDragActive(false)
      const filePath = resolveDroppedPath(event.dataTransfer)
      if (filePath) {
        onOpenAudioPath(filePath)
        return
      }
      const fallbackFile = event.dataTransfer?.files?.[0] ?? null
      if (!fallbackFile) return
      onOpenAudioFile(fallbackFile)
    }

    window.addEventListener('dragover', handleWindowDragOver, true)
    window.addEventListener('dragenter', handleWindowDragEnter, true)
    window.addEventListener('dragleave', handleWindowDragLeave, true)
    window.addEventListener('drop', handleWindowDrop, true)

    return () => {
      window.removeEventListener('dragover', handleWindowDragOver, true)
      window.removeEventListener('dragenter', handleWindowDragEnter, true)
      window.removeEventListener('dragleave', handleWindowDragLeave, true)
      window.removeEventListener('drop', handleWindowDrop, true)
    }
  }, [allowDrop, isEventInsideContainer, resolveDroppedPath, onOpenAudioPath, onOpenAudioFile])


  const previewImage = useMemo(() => {
    if (!preview) return null
    // eslint-disable-next-line react-hooks/purity
    const start = performance.now()
    const canvas = document.createElement('canvas')
    canvas.width = preview.width
    canvas.height = preview.height
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    const image = ctx.createImageData(preview.width, preview.height)
    for (let i = 0; i < preview.data.length; i += 1) {
      const normalized = preview.data[i] / 255
      const adjusted = Math.min(1, Math.max(0, (normalized - 0.5) * settings.contrast + 0.5 + settings.brightness))
      const value = Math.round(adjusted * 255)
      const color = mapColor(settings.color_map, value)
      const offset = i * 4
      image.data[offset] = color[0]
      image.data[offset + 1] = color[1]
      image.data[offset + 2] = color[2]
      image.data[offset + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
    // eslint-disable-next-line react-hooks/purity
    const elapsed = performance.now() - start
    if (elapsed > 50) {
      console.warn(`[Perf] previewImage build ${elapsed.toFixed(1)}ms (${preview.width}x${preview.height})`)
    }
    return canvas
  }, [preview, settings])

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

  const buildViewportCanvas = useCallback(
    (current: SpectrogramPreview) => {
      const canvas = document.createElement('canvas')
      canvas.width = current.width
      canvas.height = current.height
      const ctx = canvas.getContext('2d')
      if (!ctx) return null
      const image = ctx.createImageData(current.width, current.height)
      for (let i = 0; i < current.data.length; i += 1) {
        const normalized = current.data[i] / 255
        const adjusted = Math.min(
          1,
          Math.max(0, (normalized - 0.5) * settings.contrast + 0.5 + settings.brightness)
        )
        const value = Math.round(adjusted * 255)
        const color = mapColor(settings.color_map, value)
        const offset = i * 4
        image.data[offset] = color[0]
        image.data[offset + 1] = color[1]
        image.data[offset + 2] = color[2]
        image.data[offset + 3] = 255
      }
      ctx.putImageData(image, 0, 0)
      return canvas
    },
    [settings],
  )

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

      if (viewportImageCacheRef.current.has(key) || viewportPendingRef.current.has(key)) {
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
      } else {
        const canvas = buildViewportCanvas(current)
        if (canvas) {
          viewportImageCacheRef.current.set(key, canvas)
          setViewportImageVersion((value) => value + 1)
        }
      }
    }

    for (const key of viewportImageCacheRef.current.keys()) {
      if (!desiredKeys.has(key)) {
        const image = viewportImageCacheRef.current.get(key)
        if (image && 'close' in image) {
          image.close()
        }
        viewportImageCacheRef.current.delete(key)
      }
    }
    for (const key of viewportPendingRef.current) {
      if (!desiredKeys.has(key)) {
        viewportPendingRef.current.delete(key)
      }
    }
  }, [viewportPreviews, settings, buildViewportKey, buildViewportCanvas])

  const viewportPreviewImages = useMemo(() => {
    if (!viewportPreviews || viewportPreviews.length === 0) return []
    void viewportImageVersion
    return viewportPreviews
      .map((current) => {
        const key = buildViewportKey(current)
        const image = viewportImageCacheRef.current.get(key)
        if (!image) return null
        return { preview: current, image }
      })
      .filter(
        (item): item is { preview: SpectrogramPreview; image: HTMLCanvasElement | ImageBitmap } => item !== null
      )
  }, [viewportPreviews, buildViewportKey, viewportImageVersion])

  const contentOffset = useMemo(() => ({ x: 0, y: 28 }), [])
  const spectrogramAreaHeight = Math.max(1, stageSize.height - automationLaneHeight)
  const automationContentHeight = Math.max(1, automationLaneHeight - automationPadding.top - automationPadding.bottom)
  const automationTop = spectrogramAreaHeight
  const automationContentTop = automationTop + automationPadding.top

  const baseScale = useMemo(() => {
    if (!preview) return { y: 1 }
    return {
      y: Math.max(1, spectrogramAreaHeight - contentOffset.y) / preview.height,
    }
  }, [preview, spectrogramAreaHeight, contentOffset])

  const duration = preview?.duration_sec ?? 1
  const timeScale = useMemo(() => {
    if (!preview) return 1
    return (duration * zoomX) / preview.width
  }, [preview, duration, zoomX])
  const scale = useMemo(() => ({ x: timeScale, y: baseScale.y * zoomY }), [timeScale, baseScale, zoomY])
  const freqMin = preview?.freq_min ?? settings.freq_min
  const freqMax = preview?.freq_max ?? settings.freq_max

  const viewportPositions = useMemo(() => {
    if (!viewportPreviews || !preview) return []
    const logMin = Math.log(freqMin)
    const logMax = Math.log(freqMax)
    return viewportPreviews
      .filter((current) => {
        // Validate viewport preview is within bounds of main preview
        if (current.time_end > duration + 0.01) {
          console.warn(
            '[Workspace] Viewport preview time_end',
            current.time_end,
            'exceeds duration',
            duration,
            '- filtering out'
          )
          return false
        }
        if (current.freq_max > freqMax * 1.01 || current.freq_min < freqMin * 0.99) {
          console.warn(
            '[Workspace] Viewport preview freq range',
            current.freq_min,
            '-',
            current.freq_max,
            'outside main range',
            freqMin,
            '-',
            freqMax,
            '- filtering out'
          )
          return false
        }
        return true
      })
      .map((current) => {
        const vpLogMin = Math.log(current.freq_min)
        const vpLogMax = Math.log(current.freq_max)
        const yTop = ((logMax - vpLogMax) / (logMax - logMin)) * preview.height
        const yBottom = ((logMax - vpLogMin) / (logMax - logMin)) * preview.height
        return {
          preview: current,
          x: (current.time_start / duration) * preview.width,
          y: yTop,
          width: ((current.time_end - current.time_start) / duration) * preview.width,
          height: yBottom - yTop,
        }
      })
  }, [viewportPreviews, preview, duration, freqMin, freqMax])

  const timeToX = useCallback(
    (time: number) => {
      if (!preview) return 0
      return (time / duration) * preview.width
    },
    [preview, duration],
  )

  const freqToY = useCallback(
    (freq: number) => {
      if (!preview) return 0
      const logMin = Math.log(freqMin)
      const logMax = Math.log(freqMax)
      const norm = (logMax - Math.log(Math.max(freq, freqMin))) / (logMax - logMin)
      return norm * preview.height
    },
    [preview, freqMin, freqMax],
  )

  const freqRulerMarks = useMemo(() => {
    if (!preview) return []
    const marks: { freq: number; label: string }[] = []
    const frequencies = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    for (const freq of frequencies) {
      if (freq >= freqMin && freq <= freqMax) {
        const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`
        marks.push({ freq, label })
      }
    }
    return marks
  }, [preview, freqMin, freqMax])

  const positionToTimeFreq = useCallback(
    (x: number, y: number) => {
      if (!preview) return { time: 0, freq: freqMin }
      const contentX = (x - pan.x - contentOffset.x) / scale.x
      const contentY = (y - pan.y - contentOffset.y) / scale.y
      const time = (contentX / preview.width) * duration
      const logMin = Math.log(freqMin)
      const logMax = Math.log(freqMax)
      const ratio = Math.min(1, Math.max(0, contentY / preview.height))
      const freq = Math.exp(logMax - ratio * (logMax - logMin))
      return { time, freq }
    },
    [preview, duration, pan, scale, freqMin, freqMax, contentOffset],
  )

  const startPerfMonitor = () => {
    const now = performance.now()
    const ref = perfRef.current
    ref.activeUntil = now + 1200
    if (ref.rafId) return
    const tick = (timestamp: number) => {
      const delta = ref.lastFrame ? timestamp - ref.lastFrame : 0
      ref.lastFrame = timestamp
      if (delta > 50 && timestamp - ref.lastLog > 500) {
        console.warn(`[Perf] frame gap ${delta.toFixed(1)}ms during zoom/pan`)
        ref.lastLog = timestamp
      }
      if (timestamp < ref.activeUntil) {
        ref.rafId = window.requestAnimationFrame(tick)
      } else {
        ref.rafId = 0
      }
    }
    ref.rafId = window.requestAnimationFrame(tick)
  }

  const onStageWheel = (event: KonvaEventObject<WheelEvent>) => {
    event.evt.preventDefault()
    const stage = event.target.getStage()
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    if (!preview) return
    const isZoomGesture = event.evt.ctrlKey || event.evt.metaKey
    if (!isZoomGesture) {
      const nextPan = {
        x: pan.x - event.evt.deltaX,
        y: pan.y - event.evt.deltaY,
      }
      onPanChange(nextPan)
      startPerfMonitor()
      return
    }
    // Horizontal zoom only (time axis)
    const scaleBy = 1.05
    const oldScaleX = zoomX
    const direction = event.evt.deltaY > 0 ? -1 : 1
    const newScaleX = direction > 0 ? oldScaleX * scaleBy : oldScaleX / scaleBy
    const clampedX = Math.min(ZOOM_X_MAX_PX_PER_SEC, Math.max(ZOOM_X_MIN_PX_PER_SEC, newScaleX))

    const oldTimeScale = (duration * oldScaleX) / preview.width
    const newTimeScale = (duration * clampedX) / preview.width
    const mousePointToX = (pointer.x - pan.x) / oldTimeScale

    const newPan = {
      x: pointer.x - mousePointToX * newTimeScale,
      y: pan.y,
    }

    onZoomXChange(clampedX)
    onPanChange(newPan)
    startPerfMonitor()
  }

  const handleStageMouseDown = (event: KonvaEventObject<MouseEvent>) => {
    if (!preview) return
    const stage = event.target.getStage()
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return
    if (activeTool === 'trace' || activeTool === 'erase') {
      const { time, freq } = positionToTimeFreq(pointer.x, pointer.y)
      setTracePath([{ time, freq, amp: 0.5 }])
      setIsTracing(true)
      setCommittedTrace([])
    }
    if (activeTool === 'select') {
      setSelectionBox({ x: pointer.x, y: pointer.y, w: 0, h: 0 })
    }
  }

  const handleStageMouseMove = (event: KonvaEventObject<MouseEvent>) => {
    if (!preview) return
    const stage = event.target.getStage()
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return
    const cursor = positionToTimeFreq(pointer.x, pointer.y)
    onCursorMove({ time: cursor.time, freq: cursor.freq, amp: null })
    setHudPosition((prev) => {
      const margin = 12
      const width = 260
      const height = 120
      const nextX = Math.min(Math.max(margin, pointer.x + 16), stageSize.width - width - margin)
      const nextY = Math.min(Math.max(margin, pointer.y + 16), stageSize.height - height - margin)
      if (prev.x === nextX && prev.y === nextY) return prev
      return { x: nextX, y: nextY }
    })
    if (isTracing) {
      setTracePath((prev) => [...prev, { ...cursor, amp: 0.5 }])
    }
    if (selectionBox) {
      setSelectionBox((prev) =>
        prev
          ? { ...prev, w: pointer.x - prev.x, h: pointer.y - prev.y }
          : { x: pointer.x, y: pointer.y, w: 0, h: 0 },
      )
    }
  }

  const handleStageMouseUp = () => {
    if (!preview) return
    if (activeTool === 'trace' && isTracing) {
      const trace = tracePath.map((point) => [point.time, point.freq] as [number, number])
      setCommittedTrace(tracePath)
      setTracePath([])
      setIsTracing(false)
      void onTraceCommit(trace).then((ok) => {
        if (ok) {
          setCommittedTrace([])
        }
      })
      return
    }

    if (activeTool === 'erase' && isTracing) {
      const trace = tracePath.map((point) => [point.time, point.freq] as [number, number])
      onEraseCommit(trace)
      setTracePath([])
      setIsTracing(false)
      return
    }

    if (activeTool === 'select' && selectionBox) {
      const box = selectionBox
      setSelectionBox(null)
      const start = positionToTimeFreq(box.x, box.y)
      const end = positionToTimeFreq(box.x + box.w, box.y + box.h)
      onSelectBoxCommit({
        time_start: start.time,
        time_end: end.time,
        freq_start: start.freq,
        freq_end: end.freq,
      })
    }
  }

  const handleStageClick = (event: KonvaEventObject<MouseEvent>) => {
    if (!preview) return
    const stage = event.target.getStage()
    if (!stage) return
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return
    const { time, freq } = positionToTimeFreq(pointer.x, pointer.y)
    if (activeTool === 'connect') {
      onConnectPick({ time, freq })
      return
    }
    if (activeTool === 'select' && !selectionBox) {
      onHitTestCommit({ time, freq })
    }
  }

  const selectedPartials = useMemo(() => {
    return partials.filter((partial) => selectedIds.includes(partial.id))
  }, [partials, selectedIds])

  const renderPartials = useMemo(() => {
    if (!draggedPartial || !selectedIds.includes(draggedPartial.id)) return selectedPartials
    return selectedPartials.map((partial) => (partial.id === draggedPartial.id ? draggedPartial : partial))
  }, [selectedPartials, draggedPartial, selectedIds])

  const drawPartials = useCallback(
    (context: Context) => {
      const ctx = context
      ctx.save()
      ctx.lineWidth = 1
      for (const partial of partials) {
        if (partial.points.length < 2) continue
        ctx.strokeStyle = hexToRgba(partial.color, partial.is_muted ? 0.25 : 0.6)
        ctx.beginPath()
        partial.points.forEach((point, index) => {
          const x = timeToX(point.time)
          const y = freqToY(point.freq)
          if (index === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        })
        ctx.stroke()
      }
      ctx.restore()
    },
    [partials, timeToX, freqToY],
  )

  const handleEndpointDrag = (partial: Partial, index: number, position: { x: number; y: number }) => {
    const point = positionToTimeFreq(position.x * scale.x + pan.x, position.y * scale.y + pan.y)
    const updated = partial.points.map((item, idx) =>
      idx === index ? { ...item, time: point.time, freq: point.freq } : item,
    )
    setDraggedPartial({ ...partial, points: updated })
  }

  const rulerWidth = 72
  const rulerHeight = contentOffset.y

  const maxAmp = useMemo(() => {
    let max = 0
    for (const partial of partials) {
      for (const point of partial.points) {
        if (point.amp > max) {
          max = point.amp
        }
      }
    }
    return max > 0 ? max : 1
  }, [partials])

  const committedTracePath = useMemo(() => {
    if (committedTrace.length < 2 || !preview) return ''
    return committedTrace
      .map((point, index) => {
        const x = pan.x + contentOffset.x + timeToX(point.time) * scale.x
        const y = pan.y + contentOffset.y + freqToY(point.freq) * scale.y
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')
  }, [committedTrace, preview, pan, contentOffset, timeToX, freqToY, scale])

  const tracePathD = useMemo(() => {
    if (tracePath.length < 2 || !preview) return ''
    return tracePath
      .map((point, index) => {
        const x = pan.x + contentOffset.x + timeToX(point.time) * scale.x
        const y = pan.y + contentOffset.y + freqToY(point.freq) * scale.y
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')
  }, [tracePath, preview, pan, contentOffset, timeToX, freqToY, scale])

  const ampToLaneY = useCallback(
    (amp: number) => {
      const normalized = Math.min(1, Math.max(0, amp / maxAmp))
      return automationContentHeight - normalized * automationContentHeight
    },
    [automationContentHeight, maxAmp],
  )

  const visibleRange = useMemo(() => {
    if (!preview) return { start: 0, end: 0 }
    const start = Math.max(0, ((-pan.x - contentOffset.x) / scale.x / preview.width) * duration)
    const end = Math.min(
      duration,
      ((stageSize.width - pan.x - contentOffset.x) / scale.x / preview.width) * duration,
    )
    return { start, end }
  }, [preview, pan, scale, stageSize, duration, contentOffset])

  const timeMarks = useMemo(() => {
    if (!preview) return []
    const range = Math.max(0.001, visibleRange.end - visibleRange.start)
    const step =
      range > 120
        ? 10
        : range > 60
          ? 5
          : range > 30
            ? 2
            : range > 10
              ? 1
              : range > 5
                ? 0.5
                : range > 1
                  ? 0.25
                  : 0.1
    const first = Math.ceil(visibleRange.start / step) * step
    const marks = []
    for (let t = first; t <= visibleRange.end; t += step) {
      marks.push(t)
    }
    return marks
  }, [preview, visibleRange])

  return (
    <div
      ref={containerRef}
      className={`canvas-surface relative h-full min-h-[520px] rounded-none p-4 ${
        allowDrop && isDragActive ? 'ring-2 ring-[var(--accent)] ring-offset-2 ring-offset-[var(--canvas)]' : ''
      }`}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Stage
        width={stageSize.width}
        height={stageSize.height}
        onWheel={onStageWheel}
        onMouseDown={handleStageMouseDown}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
        onClick={handleStageClick}
      >
        <Layer>
          <Group x={pan.x + contentOffset.x} y={pan.y + contentOffset.y} scaleX={scale.x} scaleY={scale.y}>
            {previewImage ? (
              <KonvaImage image={previewImage} width={preview?.width} height={preview?.height} opacity={0.85} />
            ) : null}
            {viewportPreviewImages.length > 0 && viewportPositions.length > 0
              ? viewportPreviewImages.map((item) => {
                  const position = viewportPositions.find((entry) => entry.preview === item.preview)
                  if (!position) return null
                  return (
                    <KonvaImage
                      key={`${item.preview.time_start}-${item.preview.time_end}-${item.preview.freq_min}-${item.preview.freq_max}`}
                      image={item.image}
                      x={position.x}
                      y={position.y}
                      width={position.width}
                      height={position.height}
                      opacity={0.95}
                    />
                  )
                })
              : null}
          </Group>
        </Layer>
        <Layer>
          <Group x={pan.x + contentOffset.x} y={pan.y + contentOffset.y} scaleX={scale.x} scaleY={scale.y}>
            <Shape sceneFunc={(ctx, shape) => { drawPartials(ctx); ctx.fillStrokeShape(shape) }} />
            {renderPartials.map((partial) => (
              <Line
                key={partial.id}
                points={partial.points.flatMap((point) => [timeToX(point.time), freqToY(point.freq)])}
                stroke={hexToRgba(partial.color, partial.is_muted ? 0.35 : 0.95)}
                strokeWidth={2}
              />
            ))}
            {renderPartials.length === 1 && renderPartials[0].points.length >= 2 ? (
              <>
                {[0, renderPartials[0].points.length - 1].map((index) => (
                  <Circle
                    key={index}
                    x={timeToX(renderPartials[0].points[index].time)}
                    y={freqToY(renderPartials[0].points[index].freq)}
                    radius={4}
                    fill={hexToRgba(renderPartials[0].color, 0.95)}
                    draggable
                    onDragMove={(event) => {
                      const { x, y } = event.target.position()
                      handleEndpointDrag(renderPartials[0], index, { x, y })
                    }}
                    onDragEnd={(event) => {
                      const { x, y } = event.target.position()
                      const point = positionToTimeFreq(x * scale.x + pan.x, y * scale.y + pan.y)
                      const updated = renderPartials[0].points.map((item, idx) =>
                        idx === index ? { ...item, time: point.time, freq: point.freq } : item,
                      )
                      setDraggedPartial(null)
                      onUpdatePartial(renderPartials[0].id, updated)
                    }}
                  />
                ))}
              </>
            ) : null}
          </Group>
          {selectionBox ? (
            <Rect
              x={selectionBox.x}
              y={selectionBox.y}
              width={selectionBox.w}
              height={selectionBox.h}
              stroke="#f59f8b"
              dash={[4, 4]}
            />
          ) : null}
          {preview ? (
            <Line
              points={[
                pan.x + contentOffset.x + timeToX(playbackPosition) * scale.x,
                rulerHeight,
                pan.x + contentOffset.x + timeToX(playbackPosition) * scale.x,
                stageSize.height,
              ]}
              stroke="rgba(247, 245, 242, 0.8)"
              strokeWidth={1}
              dash={[6, 4]}
            />
          ) : null}
        </Layer>
        <Layer>
          <Rect x={0} y={0} width={stageSize.width} height={rulerHeight} fill="rgba(12, 18, 30, 0.7)" />
          <Line points={[0, rulerHeight, stageSize.width, rulerHeight]} stroke="rgba(248, 209, 154, 0.35)" />
          {timeMarks.map((time) => {
            const x = pan.x + contentOffset.x + timeToX(time) * scale.x
            if (x < 0 || x > stageSize.width) return null
            return (
              <Group key={time}>
                <Line points={[x, rulerHeight - 6, x, rulerHeight]} stroke="rgba(248, 209, 154, 0.6)" />
                <Text
                  x={x + 4}
                  y={2}
                  text={`${time.toFixed(time < 1 ? 2 : 1)}s`}
                  fontSize={9}
                  fill="rgba(248, 209, 154, 0.75)"
                  fontFamily="monospace"
                />
              </Group>
            )
          })}
          <Rect
            x={0}
            y={automationTop}
            width={stageSize.width}
            height={automationLaneHeight}
            fill="rgba(10, 14, 20, 0.82)"
          />
          <Line
            points={[0, automationTop, stageSize.width, automationTop]}
            stroke="rgba(248, 209, 154, 0.25)"
            strokeWidth={1}
          />
          <Text
            x={12}
            y={automationTop + 6}
            text="Partial Amplitude"
            fontSize={9}
            fill="rgba(248, 209, 154, 0.7)"
            fontFamily="monospace"
          />
          <Group x={pan.x + contentOffset.x} y={automationContentTop} scaleX={scale.x}>
            {partials.map((partial) => (
              <Line
                key={`amp-${partial.id}`}
                points={partial.points.flatMap((point) => [timeToX(point.time), ampToLaneY(point.amp)])}
                stroke={hexToRgba(partial.color, partial.is_muted ? 0.25 : 0.85)}
                strokeWidth={1.25}
              />
            ))}
          </Group>
          <Group x={stageSize.width - rulerWidth} y={pan.y + contentOffset.y} scaleY={scale.y}>
            {freqRulerMarks.map((mark) => {
              const y = freqToY(mark.freq)
              return (
                <Group key={mark.freq}>
                  <Line
                    points={[0, y, 8, y]}
                    stroke="rgba(248, 209, 154, 0.5)"
                    strokeWidth={1}
                  />
                  <Text
                    x={12}
                    y={y}
                    text={mark.label}
                    fontSize={10}
                    fill="rgba(248, 209, 154, 0.8)"
                    fontFamily="monospace"
                    scaleY={1 / scale.y}
                    offsetY={5}
                  />
                </Group>
              )
            })}
          </Group>
        </Layer>
      </Stage>
      {tracePathD || committedTracePath ? (
        <svg
          className="pointer-events-none absolute inset-4"
          width={stageSize.width}
          height={stageSize.height}
          viewBox={`0 0 ${stageSize.width} ${stageSize.height}`}
        >
          {tracePathD ? (
            <path d={tracePathD} fill="none" stroke="#7feeff" strokeWidth={3.75} />
          ) : null}
          <path
            d={committedTracePath}
            fill="none"
            stroke={isSnapping ? '#7feeff' : 'rgba(245, 159, 139, 0.5)'}
            strokeWidth={isSnapping ? 3.75 : 1.25}
            className={isSnapping ? 'animate-pulse' : ''}
          />
        </svg>
      ) : null}
      {!preview ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center text-white/70">
          <div className="text-xs uppercase tracking-[0.3em]">No Audio Loaded</div>
          {allowDrop ? (
            <div className="text-[10px] uppercase tracking-[0.24em] text-white/60">Drop a WAV file here</div>
          ) : null}
          <button
            className="rounded-md bg-white/90 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]"
            onClick={onOpenAudio}
          >
            Open Audio
          </button>
        </div>
      ) : null}
      {allowDrop && isDragActive ? (
        <div className="pointer-events-none absolute inset-3 rounded-md border border-dashed border-white/60" />
      ) : null}
      {analysisState === 'analyzing' ? (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-xs uppercase tracking-[0.24em] text-white">
          Analyzing...
        </div>
      ) : null}
      {selectedInfo ? (
        <SelectionHud
          selected={selectedInfo}
          canMute={selectedIds.length === 1}
          canDelete={selectedIds.length > 0}
          onMute={onPartialMute}
          onDelete={onPartialDelete}
          position={hudPosition}
        />
      ) : null}
      {preview ? (
        <div className="absolute right-6 top-1/2 -translate-y-1/2 flex flex-col gap-1">
          <button
            className="flex h-7 w-7 items-center justify-center rounded-sm bg-[rgba(12,18,30,0.85)] text-[var(--muted)] hover:bg-[rgba(20,28,45,0.95)] hover:text-white transition-colors"
            onClick={onZoomInY}
            title="Zoom in frequency"
          >
            <span className="text-sm font-bold">+</span>
          </button>
          <div className="flex h-5 items-center justify-center text-[9px] text-[var(--muted)] font-mono">
            {Math.round(zoomY * 100)}%
          </div>
          <button
            className="flex h-7 w-7 items-center justify-center rounded-sm bg-[rgba(12,18,30,0.85)] text-[var(--muted)] hover:bg-[rgba(20,28,45,0.95)] hover:text-white transition-colors"
            onClick={onZoomOutY}
            title="Zoom out frequency"
          >
            <span className="text-sm font-bold">−</span>
          </button>
        </div>
      ) : null}
    </div>
  )
}
