import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
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


  const previewImage = useMemo(() => {
    if (!preview) return null
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

  const viewportPreviewImages = useMemo(() => {
    if (!viewportPreviews || viewportPreviews.length === 0) return []
    return viewportPreviews
      .map((current) => {
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
        return { preview: current, image: canvas }
      })
      .filter((item): item is { preview: SpectrogramPreview; image: HTMLCanvasElement } => item !== null)
  }, [viewportPreviews, settings])

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
    return viewportPreviews.map((current) => {
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
    <div ref={containerRef} className="canvas-surface relative h-full min-h-[520px] rounded-none p-4">
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
        </Layer>
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
        </Layer>
        <Layer>
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
        </Layer>
        <Layer>
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
        <Layer>
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
          <button
            className="rounded-none bg-white/90 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]"
            onClick={onOpenAudio}
          >
            Open Audio
          </button>
        </div>
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
