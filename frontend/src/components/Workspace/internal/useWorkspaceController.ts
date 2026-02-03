import { useCallback, useEffect, useMemo, useState } from 'react'
import type { KonvaEventObject } from 'konva/lib/Node'
import { AUTOMATION_LANE_HEIGHT, RULER_HEIGHT, ZOOM_X_MAX_PX_PER_SEC, ZOOM_X_MIN_PX_PER_SEC } from '../../../app/constants'
import type { Partial } from '../../../app/types'
import { mapColor } from '../../../app/utils'
import { freqToY, positionToTime, positionToTimeFreq, timeToX } from './coordinate'
import { useDropAudio } from './useDropAudio'
import { useTraceSelectionInteraction } from './useTraceSelectionInteraction'
import { useViewportImageCache } from './useViewportImageCache'
import type { WorkspaceProps } from '../types'

export type EndpointDragParams = {
  partial: Partial
  index: number
  position: { x: number; y: number }
}

export function useWorkspaceController(props: WorkspaceProps) {
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
    onStageSizeChange,
    onTraceCommit,
    onEraseCommit,
    onSelectBoxCommit,
    onHitTestCommit,
    onUpdatePartial,
    onConnectPick,
    onPlayheadChange,
    allowDrop,
    onCursorMove,
  } = props

  const { containerRef, isDragActive, handleDragOver, handleDragEnter, handleDragLeave, handleDrop } = useDropAudio({
    allowDrop,
    onOpenAudioPath: props.onOpenAudioPath,
    onOpenAudioFile: props.onOpenAudioFile,
  })
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [draggedPartial, setDraggedPartial] = useState<Partial | null>(null)
  const [hudPosition, setHudPosition] = useState({ x: 16, y: 16 })
  const automationPadding = { top: 18, bottom: 16 }

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setStageSize({ width, height })
        onStageSizeChange({ width, height })
      }
    })
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [containerRef, onStageSizeChange])

  const previewImage = useMemo(() => {
    if (!preview) return null
    if (preview.data.length !== preview.width * preview.height) return null
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
    return canvas
  }, [preview, settings])

  const { buildViewportKey, viewportImages } = useViewportImageCache(viewportPreviews, settings)

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
  const freqMin = preview?.freq_min ?? settings.freq_min
  const freqMax = preview?.freq_max ?? settings.freq_max

  const pxPerOctave = useMemo(() => {
    if (!preview) return 0
    const effectiveHeight = spectrogramAreaHeight - contentOffset.y
    const totalOctaves = Math.log2(freqMax / freqMin)
    const basePxPerOctave = effectiveHeight / totalOctaves
    return basePxPerOctave * zoomY
  }, [preview, spectrogramAreaHeight, contentOffset.y, freqMax, freqMin, zoomY])

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

  const { tracePath, committedTrace, selectionBox, beginAt, moveAt, endInteraction } = useTraceSelectionInteraction({
    activeTool,
    positionToTimeFreq: positionToTimeFreqValue,
    onTraceCommit,
    onEraseCommit,
    onSelectBoxCommit,
  })

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
      const clampedX = Math.min(ZOOM_X_MAX_PX_PER_SEC, Math.max(ZOOM_X_MIN_PX_PER_SEC, newScaleX))

      const oldTimeScale = (duration * oldScaleX) / preview.width
      const newTimeScale = (duration * clampedX) / preview.width
      const mousePointToX = (pointer.x - pan.x) / oldTimeScale
      const newPan = {
        x: pointer.x - mousePointToX * newTimeScale,
        y: pan.y,
      }
      onZoomXChange(clampedX, newPan)
    },
    [preview, pan, onPanChange, zoomX, duration, onZoomXChange]
  )

  const handleStageMouseDown = useCallback(
    (event: KonvaEventObject<MouseEvent>) => {
      if (!preview) return
      const stage = event.target.getStage()
      if (!stage) return
      const pointer = stage.getPointerPosition()
      if (!pointer) return
      if (pointer.y < rulerHeight || pointer.y > spectrogramAreaHeight) return
      beginAt(pointer)
    },
    [preview, rulerHeight, spectrogramAreaHeight, beginAt]
  )

  const handleStageMouseMove = useCallback(
    (event: KonvaEventObject<MouseEvent>) => {
      if (!preview) return
      const stage = event.target.getStage()
      if (!stage) return
      const pointer = stage.getPointerPosition()
      if (!pointer) return
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
    [preview, rulerHeight, spectrogramAreaHeight, positionToTimeFreqValue, onCursorMove, stageSize, moveAt]
  )

  const handleStageMouseUp = useCallback(() => {
    if (!preview) return
    endInteraction()
  }, [preview, endInteraction])

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

  const maxAmp = useMemo(() => {
    let max = 0
    for (const partial of partials) {
      for (const point of partial.points) {
        if (point.amp > max) max = point.amp
      }
    }
    return max > 0 ? max : 1
  }, [partials])

  const committedTracePath = useMemo(() => {
    if (committedTrace.length < 2 || !preview) return ''
    return committedTrace
      .map((point, index) => {
        const x = pan.x + contentOffset.x + timeToXValue(point.time) * scale.x
        const y = pan.y + contentOffset.y + freqToYValue(point.freq) * scale.y
        return `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`
      })
      .join(' ')
  }, [committedTrace, preview, pan, contentOffset, timeToXValue, freqToYValue, scale])

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

  return {
    containerRef,
    stageSize,
    isDragActive,
    hudPosition,
    tracePathD,
    committedTracePath,
    selectionBox,
    previewImage,
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
    handleDragOver,
    handleDragEnter,
    handleDragLeave,
    handleDrop,
    onStageWheel,
    handleStageMouseDown,
    handleStageMouseMove,
    handleStageMouseUp,
    handleStageClick,
    handleEndpointDragMove,
    handleEndpointDragEnd,
    playbackPosition,
    pan,
    preview,
    buildViewportKey,
    partials,
  }
}
