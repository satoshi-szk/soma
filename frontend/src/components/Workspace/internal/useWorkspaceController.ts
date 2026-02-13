import { useCallback, useEffect, useMemo, useState } from 'react'
import type { KonvaEventObject } from 'konva/lib/Node'
import { AUTOMATION_LANE_HEIGHT, RULER_HEIGHT, ZOOM_X_MAX_PX_PER_SEC, ZOOM_X_MIN_PX_PER_SEC } from '../../../app/constants'
import type { Partial } from '../../../app/types'
import { formatNoteWithCents } from '../../../app/utils'
import { freqToY, positionToTime, positionToTimeFreq, timeToX } from './coordinate'
import { useRulerMetrics } from './useRulerMetrics'
import { useTraceSelectionInteraction } from './useTraceSelectionInteraction'
import { useViewportImageCache } from './useViewportImageCache'
import type { WorkspaceProps } from '../types'

export type EndpointDragParams = {
  partial: Partial
  index: number
  position: { x: number; y: number }
}

export function useWorkspaceController(props: WorkspaceProps, stageSize: { width: number; height: number }) {
  const {
    preview,
    viewportPreviews,
    settings,
    partials,
    selectedIds,
    activeTool,
    zoomX,
    zoomY,
    pan,
    playbackPosition,
    canEditPlayhead,
    onZoomXChange,
    onPanChange,
    onTraceCommit,
    onEraseCommit,
    onSelectBoxCommit,
    onHitTestCommit,
    onUpdatePartial,
    onConnectPick,
    onPlayheadChange,
    onCursorMove,
  } = props

  const [draggedPartial, setDraggedPartial] = useState<Partial | null>(null)
  const [hudPosition, setHudPosition] = useState({ x: 16, y: 16 })
  const [hoverPointer, setHoverPointer] = useState<{ x: number; y: number } | null>(null)
  const [isRulerDragging, setIsRulerDragging] = useState(false)
  const automationPadding = { top: 18, bottom: 16 }

  const [previewImage, setPreviewImage] = useState<HTMLImageElement | null>(null)
  const [previewImagePath, setPreviewImagePath] = useState<string | null>(null)
  useEffect(() => {
    if (!preview?.image_path) return
    let cancelled = false
    const image = new window.Image()
    image.onload = () => {
      if (!cancelled) {
        setPreviewImage(image)
        setPreviewImagePath(preview.image_path ?? null)
      }
    }
    image.onerror = () => {
      if (!cancelled) {
        setPreviewImage(null)
        setPreviewImagePath(null)
      }
    }
    image.src = preview.image_path
    return () => {
      cancelled = true
    }
  }, [preview?.image_path])
  const displayPreviewImage = previewImagePath === (preview?.image_path ?? null) ? previewImage : null

  const { buildViewportKey, viewportImages } = useViewportImageCache(viewportPreviews)

  const contentOffset = useMemo(() => ({ x: 0, y: RULER_HEIGHT }), [])
  const rulerHeight = contentOffset.y
  const rulerWidth = 72
  const spectrogramAreaHeight = Math.max(1, stageSize.height - AUTOMATION_LANE_HEIGHT)
  const automationContentHeight = Math.max(1, AUTOMATION_LANE_HEIGHT - automationPadding.top - automationPadding.bottom)
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
  const freqMin = preview?.freq_min ?? settings.spectrogram.freq_min
  const freqMax = preview?.freq_max ?? settings.spectrogram.freq_max

  const viewportPositionsByKey = useMemo(() => {
    if (!viewportPreviews || !preview) return new Map<string, { x: number; y: number; width: number; height: number }>()
    const logMin = Math.log(freqMin)
    const logMax = Math.log(freqMax)
    const positions = new Map<string, { x: number; y: number; width: number; height: number }>()

    for (const current of viewportPreviews) {
      if (current.time_end > duration + 0.01) {
        console.warn(
          '[Workspace] Viewport preview time_end',
          current.time_end,
          'exceeds duration',
          duration,
          '- filtering out'
        )
        continue
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
        continue
      }
      const vpLogMin = Math.log(current.freq_min)
      const vpLogMax = Math.log(current.freq_max)
      const yTop = ((logMax - vpLogMax) / (logMax - logMin)) * preview.height
      const yBottom = ((logMax - vpLogMin) / (logMax - logMin)) * preview.height
      positions.set(buildViewportKey(current), {
        x: (current.time_start / duration) * preview.width,
        y: yTop,
        width: ((current.time_end - current.time_start) / duration) * preview.width,
        height: yBottom - yTop,
      })
    }
    return positions
  }, [viewportPreviews, preview, duration, freqMin, freqMax, buildViewportKey])

  const timeToXValue = useCallback((time: number) => timeToX(time, preview, duration), [preview, duration])
  const freqToYValue = useCallback((freq: number) => freqToY(freq, preview, freqMin, freqMax), [preview, freqMin, freqMax])

  const positionToTimeFreqValue = useCallback(
    (x: number, y: number) =>
      positionToTimeFreq(x, y, {
        preview,
        duration,
        freqMin,
        freqMax,
        pan,
        scale,
        contentOffset,
      }),
    [preview, duration, freqMin, freqMax, pan, scale, contentOffset]
  )

  const positionToTimeValue = useCallback(
    (x: number) => positionToTime(x, { preview, duration, freqMin, freqMax, pan, scale, contentOffset }),
    [preview, duration, freqMin, freqMax, pan, scale, contentOffset]
  )

  const { tracePath, committedTraces, selectionBox, beginAt, moveAt, endInteraction } = useTraceSelectionInteraction({
    activeTool,
    positionToTimeFreq: positionToTimeFreqValue,
    onTraceCommit,
    onEraseCommit,
    onSelectBoxCommit,
  })

  const onStageWheel = useCallback(
    (event: KonvaEventObject<WheelEvent>) => {
      event.evt.preventDefault()
      const stage = event.target.getStage()
      if (!stage || !preview) return
      const pointer = stage.getPointerPosition()
      if (!pointer) return

      const isZoomGesture = event.evt.ctrlKey || event.evt.metaKey
      if (!isZoomGesture) {
        const nextPan = {
          x: pan.x - event.evt.deltaX,
          y: pan.y - event.evt.deltaY,
        }
        onPanChange(nextPan)
        return
      }

      const scaleBy = 1.05
      const oldScaleX = zoomX
      const direction = event.evt.deltaY > 0 ? -1 : 1
      const newScaleX = direction > 0 ? oldScaleX * scaleBy : oldScaleX / scaleBy
      const minZoomX = Math.max(ZOOM_X_MIN_PX_PER_SEC, stageSize.width / Math.max(duration, 1e-6))
      const clampedX = Math.min(ZOOM_X_MAX_PX_PER_SEC, Math.max(minZoomX, newScaleX))

      const oldTimeScale = (duration * oldScaleX) / preview.width
      const newTimeScale = (duration * clampedX) / preview.width
      const mousePointToX = (pointer.x - pan.x) / oldTimeScale
      const newPan = {
        x: pointer.x - mousePointToX * newTimeScale,
        y: pan.y,
      }
      onZoomXChange(clampedX, newPan)
    },
    [preview, pan, onPanChange, zoomX, duration, onZoomXChange, stageSize.width]
  )

  const handleStageMouseDown = useCallback(
    (event: KonvaEventObject<MouseEvent>) => {
      if (!preview) return
      const stage = event.target.getStage()
      if (!stage) return
      const pointer = stage.getPointerPosition()
      if (!pointer) return
      if (pointer.y < rulerHeight) {
        if (!canEditPlayhead) return
        setIsRulerDragging(true)
        onPlayheadChange(positionToTimeValue(pointer.x))
        return
      }
      if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return
      beginAt(pointer)
    },
    [preview, rulerHeight, canEditPlayhead, onPlayheadChange, positionToTimeValue, spectrogramAreaHeight, beginAt]
  )

  const handleStageMouseMove = useCallback(
    (event: KonvaEventObject<MouseEvent>) => {
      if (!preview) return
      const stage = event.target.getStage()
      if (!stage) return
      const pointer = stage.getPointerPosition()
      if (!pointer) return
      setHoverPointer(pointer)
      if (isRulerDragging) {
        if (canEditPlayhead) {
          onPlayheadChange(positionToTimeValue(pointer.x))
        }
        return
      }
      if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return

      const cursor = positionToTimeFreqValue(pointer.x, pointer.y)
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

      moveAt(pointer, cursor)
    },
    [
      preview,
      isRulerDragging,
      canEditPlayhead,
      onPlayheadChange,
      positionToTimeValue,
      rulerHeight,
      spectrogramAreaHeight,
      positionToTimeFreqValue,
      onCursorMove,
      stageSize,
      moveAt,
    ]
  )

  const handleStageMouseUp = useCallback(() => {
    setIsRulerDragging(false)
    if (!preview) return
    endInteraction()
  }, [preview, endInteraction])

  const handleStageMouseLeave = useCallback(() => {
    setHoverPointer(null)
    setIsRulerDragging(false)
    endInteraction()
  }, [endInteraction])

  const handleStageClick = useCallback(
    (event: KonvaEventObject<MouseEvent>) => {
      if (!preview) return
      const stage = event.target.getStage()
      if (!stage) return
      const pointer = stage.getPointerPosition()
      if (!pointer) return
      if (pointer.y < rulerHeight) {
        if (!canEditPlayhead) return
        onPlayheadChange(positionToTimeValue(pointer.x))
        return
      }
      if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return
      const { time, freq } = positionToTimeFreqValue(pointer.x, pointer.y)
      if (activeTool === 'connect') {
        onConnectPick({ time, freq })
        return
      }
      if (activeTool === 'select' && !selectionBox) {
        onHitTestCommit({ time, freq })
      }
    },
    [
      preview,
      rulerHeight,
      canEditPlayhead,
      onPlayheadChange,
      positionToTimeValue,
      spectrogramAreaHeight,
      positionToTimeFreqValue,
      activeTool,
      onConnectPick,
      selectionBox,
      onHitTestCommit,
    ]
  )

  const selectedPartials = useMemo(() => partials.filter((partial) => selectedIds.includes(partial.id)), [partials, selectedIds])

  const renderPartials = useMemo(() => {
    if (!draggedPartial || !selectedIds.includes(draggedPartial.id)) return selectedPartials
    return selectedPartials.map((partial) => (partial.id === draggedPartial.id ? draggedPartial : partial))
  }, [selectedPartials, draggedPartial, selectedIds])

  const unselectedPartials = useMemo(
    () => partials.filter((partial) => !selectedIds.includes(partial.id)),
    [partials, selectedIds],
  )

  const handleEndpointDragMove = useCallback(
    ({ partial, index, position }: EndpointDragParams) => {
      const point = positionToTimeFreqValue(position.x * scale.x + pan.x, position.y * scale.y + pan.y)
      const updated = partial.points.map((item, idx) =>
        idx === index ? { ...item, time: point.time, freq: point.freq } : item,
      )
      setDraggedPartial({ ...partial, points: updated })
    },
    [positionToTimeFreqValue, scale, pan],
  )

  const handleEndpointDragEnd = useCallback(
    ({ partial, index, position }: EndpointDragParams) => {
      const point = positionToTimeFreqValue(position.x * scale.x + pan.x, position.y * scale.y + pan.y)
      const updated = partial.points.map((item, idx) =>
        idx === index ? { ...item, time: point.time, freq: point.freq } : item,
      )
      setDraggedPartial(null)
      onUpdatePartial(partial.id, updated)
    },
    [positionToTimeFreqValue, scale, pan, onUpdatePartial],
  )

  const committedTracePaths = useMemo(() => {
    if (!preview) return []
    return committedTraces
      .map((trace) => {
        if (trace.points.length < 2) return ''
        return trace.points
          .map((point, index) => {
            const x = pan.x + contentOffset.x + timeToXValue(point.time) * scale.x
            const y = pan.y + contentOffset.y + freqToYValue(point.freq) * scale.y
            return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
          })
          .join(' ')
      })
      .filter((path) => path.length > 0)
  }, [committedTraces, preview, pan, contentOffset, timeToXValue, freqToYValue, scale])

  const tracePathD = useMemo(() => {
    if (tracePath.length < 2 || !preview) return ''
    return tracePath
      .map((point, index) => {
        const x = pan.x + contentOffset.x + timeToXValue(point.time) * scale.x
        const y = pan.y + contentOffset.y + freqToYValue(point.freq) * scale.y
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')
  }, [tracePath, preview, pan, contentOffset, timeToXValue, freqToYValue, scale])

  const { pxPerOctave, freqRulerMarks, timeMarks, ampToLaneY, maxAmp } = useRulerMetrics({
    preview,
    freqMin,
    freqMax,
    zoomY,
    spectrogramAreaHeight,
    contentOffsetX: contentOffset.x,
    contentOffsetY: contentOffset.y,
    pan,
    scale,
    stageSize,
    duration,
    automationContentHeight,
    partials,
  })

  const playheadIntersections = useMemo(() => {
    if (!preview) return []
    const x = pan.x + contentOffset.x + timeToXValue(playbackPosition) * scale.x
    return partials
      .map((partial) => {
        const sample = samplePartialAtTime(partial, playbackPosition)
        if (!sample) return null
        const y = pan.y + contentOffset.y + freqToYValue(sample.freq) * scale.y
        if (y < rulerHeight || y > spectrogramAreaHeight) return null
        return {
          id: partial.id,
          x,
          y,
          text: `${formatNoteWithCents(sample.freq)} | ${formatAmpDb(sample.amp, maxAmp)}`,
        }
      })
      .filter((item): item is { id: string; x: number; y: number; text: string } => item !== null)
  }, [
    preview,
    pan,
    contentOffset,
    timeToXValue,
    playbackPosition,
    scale,
    partials,
    maxAmp,
    freqToYValue,
    rulerHeight,
    spectrogramAreaHeight,
  ])

  const showResolutionAssist = zoomX / 1000 >= 0.8

  const hoverResolutionCell = useMemo(() => {
    if (!preview || !hoverPointer) return null
    if (hoverPointer.y < rulerHeight || hoverPointer.y > spectrogramAreaHeight) return null
    if (freqMin <= 0 || freqMax <= freqMin) return null
    if (!showResolutionAssist) return null

    const timeResolutionSec = Math.max(0.001, settings.snap.time_resolution_ms / 1000)
    const binsPerOctave = Math.max(1, Math.round(settings.snap.bins_per_octave))
    const cursor = positionToTimeFreqValue(hoverPointer.x, hoverPointer.y)

    const clampedTime = Math.max(0, Math.min(duration, cursor.time))
    const timeStart = Math.floor(clampedTime / timeResolutionSec) * timeResolutionSec
    const timeEnd = Math.min(duration, timeStart + timeResolutionSec)

    const logMin = Math.log2(freqMin)
    const logCursor = Math.log2(Math.max(freqMin, Math.min(freqMax, cursor.freq)))
    const binIndex = Math.floor((logCursor - logMin) * binsPerOctave)
    const freqStart = Math.max(freqMin, 2 ** (logMin + binIndex / binsPerOctave))
    const freqEnd = Math.min(freqMax, 2 ** (logMin + (binIndex + 1) / binsPerOctave))

    const x0 = pan.x + contentOffset.x + timeToXValue(timeStart) * scale.x
    const x1 = pan.x + contentOffset.x + timeToXValue(timeEnd) * scale.x
    const y0 = pan.y + contentOffset.y + freqToYValue(freqStart) * scale.y
    const y1 = pan.y + contentOffset.y + freqToYValue(freqEnd) * scale.y

    const rectX = Math.min(x0, x1)
    const rectY = Math.min(y0, y1)
    const rectW = Math.abs(x1 - x0)
    const rectH = Math.abs(y1 - y0)

    if (rectW < 1 || rectH < 1) return null
    return { x: rectX, y: rectY, w: rectW, h: rectH }
  }, [
    preview,
    hoverPointer,
    rulerHeight,
    spectrogramAreaHeight,
    freqMin,
    freqMax,
    settings.snap.time_resolution_ms,
    settings.snap.bins_per_octave,
    showResolutionAssist,
    positionToTimeFreqValue,
    duration,
    pan,
    contentOffset,
    timeToXValue,
    scale,
    freqToYValue,
  ])

  return {
    stageSize,
    hudPosition,
    tracePathD,
    committedTracePaths,
    selectionBox,
    previewImage: displayPreviewImage,
    viewportImages,
    viewportPositionsByKey,
    contentOffset,
    rulerHeight,
    rulerWidth,
    spectrogramAreaHeight,
    automationTop,
    automationContentTop,
    scale,
    pxPerOctave,
    renderPartials,
    unselectedPartials,
    freqRulerMarks,
    timeMarks,
    ampToLaneY,
    timeToX: timeToXValue,
    freqToY: freqToYValue,
    onStageWheel,
    handleStageMouseDown,
    handleStageMouseMove,
    handleStageMouseUp,
    handleStageMouseLeave,
    handleStageClick,
    handleEndpointDragMove,
    handleEndpointDragEnd,
    playbackPosition,
    playheadIntersections,
    pan,
    preview,
    buildViewportKey,
    partials,
    hoverResolutionCell,
    showResolutionAssist,
  }
}

const samplePartialAtTime = (partial: Partial, timeSec: number): { freq: number; amp: number } | null => {
  const points = [...partial.points].sort((a, b) => a.time - b.time)
  if (points.length < 2) return null
  if (timeSec < points[0].time || timeSec > points[points.length - 1].time) return null
  for (let idx = 0; idx < points.length - 1; idx += 1) {
    const start = points[idx]
    const end = points[idx + 1]
    if (timeSec < start.time || timeSec > end.time) continue
    const dt = end.time - start.time
    if (dt <= 1e-9) return { freq: end.freq, amp: end.amp }
    const ratio = (timeSec - start.time) / dt
    return {
      freq: start.freq + (end.freq - start.freq) * ratio,
      amp: start.amp + (end.amp - start.amp) * ratio,
    }
  }
  const tail = points[points.length - 1]
  return { freq: tail.freq, amp: tail.amp }
}

const formatAmpDb = (amp: number, maxAmp: number): string => {
  const floorDb = -60
  if (!Number.isFinite(amp) || amp <= 0 || !Number.isFinite(maxAmp) || maxAmp <= 0) {
    return `${floorDb.toFixed(1)}dB`
  }
  const db = 20 * Math.log10(Math.max(amp, 1e-12) / maxAmp)
  const clamped = Math.max(floorDb, Math.min(0, db))
  return `${clamped.toFixed(1)}dB`
}
