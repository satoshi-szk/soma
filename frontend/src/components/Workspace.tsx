import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Circle, Group, Layer, Line, Rect, Shape, Stage, Image as KonvaImage } from 'react-konva'
import type { AnalysisSettings, Partial, PartialPoint, SpectrogramPreview, ToolId } from '../app/types'
import { mapColor } from '../app/utils'

export type WorkspaceProps = {
  preview: SpectrogramPreview | null
  settings: AnalysisSettings
  partials: Partial[]
  selectedIds: string[]
  activeTool: ToolId
  analysisState: 'idle' | 'analyzing' | 'error'
  zoom: number
  pan: { x: number; y: number }
  onZoomChange: (zoom: number) => void
  onPanChange: (pan: { x: number; y: number }) => void
  onTraceCommit: (trace: Array<[number, number]>) => Promise<boolean>
  onEraseCommit: (trace: Array<[number, number]>) => void
  onSelectBoxCommit: (selection: { time_start: number; time_end: number; freq_start: number; freq_end: number }) => void
  onHitTestCommit: (point: { time: number; freq: number }) => void
  onUpdatePartial: (id: string, points: PartialPoint[]) => void
  onConnectPick: (point: { time: number; freq: number }) => void
  onOpenAudio: () => void
}

export function Workspace({
  preview,
  settings,
  partials,
  selectedIds,
  activeTool,
  analysisState,
  zoom,
  pan,
  onZoomChange,
  onPanChange,
  onTraceCommit,
  onEraseCommit,
  onSelectBoxCommit,
  onHitTestCommit,
  onUpdatePartial,
  onConnectPick,
  onOpenAudio,
}: WorkspaceProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [tracePath, setTracePath] = useState<PartialPoint[]>([])
  const [isTracing, setIsTracing] = useState(false)
  const [committedTrace, setCommittedTrace] = useState<PartialPoint[]>([])
  const [selectionBox, setSelectionBox] = useState<null | { x: number; y: number; w: number; h: number }>(null)
  const [draggedPartial, setDraggedPartial] = useState<Partial | null>(null)

  useEffect(() => {
    setDraggedPartial(null)
  }, [selectedIds])

  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setStageSize({ width, height })
      }
    })
    if (containerRef.current) {
      observer.observe(containerRef.current)
    }
    return () => observer.disconnect()
  }, [])

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

  const baseScale = useMemo(() => {
    if (!preview) return { x: 1, y: 1 }
    return {
      x: stageSize.width / preview.width,
      y: stageSize.height / preview.height,
    }
  }, [preview, stageSize])

  const scale = useMemo(() => ({ x: baseScale.x * zoom, y: baseScale.y * zoom }), [baseScale, zoom])

  const duration = preview?.duration_sec ?? 1
  const freqMin = preview?.freq_min ?? settings.freq_min
  const freqMax = preview?.freq_max ?? settings.freq_max

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

  const positionToTimeFreq = useCallback(
    (x: number, y: number) => {
      if (!preview) return { time: 0, freq: freqMin }
      const contentX = (x - pan.x) / scale.x
      const contentY = (y - pan.y) / scale.y
      const time = (contentX / preview.width) * duration
      const logMin = Math.log(freqMin)
      const logMax = Math.log(freqMax)
      const ratio = Math.min(1, Math.max(0, contentY / preview.height))
      const freq = Math.exp(logMax - ratio * (logMax - logMin))
      return { time, freq }
    },
    [preview, duration, pan, scale, freqMin, freqMax],
  )

  const onStageWheel = (event: any) => {
    event.evt.preventDefault()
    const stage = event.target.getStage()
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
    const oldScale = zoom
    const direction = event.evt.deltaY > 0 ? -1 : 1
    const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy
    const clamped = Math.min(4, Math.max(0.5, newScale))

    const mousePointTo = {
      x: (pointer.x - pan.x) / (baseScale.x * oldScale),
      y: (pointer.y - pan.y) / (baseScale.y * oldScale),
    }

    const newPan = {
      x: pointer.x - mousePointTo.x * baseScale.x * clamped,
      y: pointer.y - mousePointTo.y * baseScale.y * clamped,
    }

    onZoomChange(clamped)
    onPanChange(newPan)
  }

  const handleStageMouseDown = (event: any) => {
    if (!preview) return
    const stage = event.target.getStage()
    const pointer = stage.getPointerPosition()
    if (!pointer) return
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

  const handleStageMouseMove = (event: any) => {
    if (!preview) return
    const stage = event.target.getStage()
    const pointer = stage.getPointerPosition()
    if (!pointer) return
    if (isTracing) {
      const { time, freq } = positionToTimeFreq(pointer.x, pointer.y)
      setTracePath((prev) => [...prev, { time, freq, amp: 0.5 }])
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

  const handleStageClick = (event: any) => {
    if (!preview) return
    const stage = event.target.getStage()
    const pointer = stage.getPointerPosition()
    if (!pointer) return
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
    if (!draggedPartial) return selectedPartials
    return selectedPartials.map((partial) => (partial.id === draggedPartial.id ? draggedPartial : partial))
  }, [selectedPartials, draggedPartial])

  const drawPartials = useCallback(
    (context: any) => {
      const ctx = context as CanvasRenderingContext2D
      ctx.save()
      ctx.lineWidth = 1
      for (const partial of partials) {
        if (partial.points.length < 2) continue
        ctx.strokeStyle = partial.is_muted ? 'rgba(248, 209, 154, 0.3)' : '#f8d19a'
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

  return (
    <div ref={containerRef} className="canvas-surface relative rounded-none p-4 min-h-[520px]">
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
          <Group x={pan.x} y={pan.y} scaleX={scale.x} scaleY={scale.y}>
            {previewImage ? (
              <KonvaImage image={previewImage} width={preview?.width} height={preview?.height} opacity={0.85} />
            ) : null}
          </Group>
        </Layer>
        <Layer>
          <Group x={pan.x} y={pan.y} scaleX={scale.x} scaleY={scale.y}>
            <Shape sceneFunc={(ctx, shape) => { drawPartials(ctx); ctx.fillStrokeShape(shape) }} />
            {tracePath.length > 1 ? (
              <Line
                points={tracePath.flatMap((point) => [timeToX(point.time), freqToY(point.freq)])}
                stroke="#f59f8b"
                strokeWidth={1.5}
              />
            ) : null}
            {committedTrace.length > 1 ? (
              <Line
                points={committedTrace.flatMap((point) => [timeToX(point.time), freqToY(point.freq)])}
                stroke="rgba(245, 159, 139, 0.5)"
                strokeWidth={1.25}
              />
            ) : null}
            {renderPartials.map((partial) => (
              <Line
                key={partial.id}
                points={partial.points.flatMap((point) => [timeToX(point.time), freqToY(point.freq)])}
                stroke="#fdf5d3"
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
                    fill="#f59f8b"
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
      </Stage>
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
    </div>
  )
}
