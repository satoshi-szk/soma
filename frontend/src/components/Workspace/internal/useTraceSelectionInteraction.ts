import { useCallback, useRef, useState } from 'react'
import type { PartialPoint, ToolId } from '../../../app/types'

type SelectionBox = { x: number; y: number; w: number; h: number }
type CursorPoint = { time: number; freq: number }

type Params = {
  activeTool: ToolId
  positionToTimeFreq: (x: number, y: number) => { time: number; freq: number }
  onTraceCommit: (trace: Array<[number, number]>) => Promise<boolean>
  onEraseCommit: (trace: Array<[number, number]>) => void
  onSelectBoxCommit: (selection: { time_start: number; time_end: number; freq_start: number; freq_end: number }) => void
}

export function useTraceSelectionInteraction({
  activeTool,
  positionToTimeFreq,
  onTraceCommit,
  onEraseCommit,
  onSelectBoxCommit,
}: Params) {
  const [tracePath, setTracePath] = useState<PartialPoint[]>([])
  const [isTracing, setIsTracing] = useState(false)
  const [committedTraces, setCommittedTraces] = useState<Array<{ id: number; points: PartialPoint[] }>>([])
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null)
  const traceDirectionRef = useRef<-1 | 0 | 1>(0)
  const nextTraceIdRef = useRef(1)

  const beginAt = useCallback(
    (pointer: { x: number; y: number }) => {
      if (activeTool === 'trace' || activeTool === 'erase') {
        const { time, freq } = positionToTimeFreq(pointer.x, pointer.y)
        setTracePath([{ time, freq, amp: 0.5 }])
        setIsTracing(true)
        traceDirectionRef.current = 0
      }
      if (activeTool === 'select') {
        setSelectionBox({ x: pointer.x, y: pointer.y, w: 0, h: 0 })
      }
    },
    [activeTool, positionToTimeFreq],
  )

  const moveAt = useCallback(
    (pointer: { x: number; y: number }, cursor: CursorPoint) => {
      if (isTracing) {
        setTracePath((prev) => {
          if (prev.length === 0) return prev
          const last = prev[prev.length - 1]
          const delta = cursor.time - last.time
          if (Math.abs(delta) <= 1e-9) {
            return prev
          }
          if (traceDirectionRef.current === 0) {
            traceDirectionRef.current = delta > 0 ? 1 : -1
          } else if ((delta > 0 ? 1 : -1) !== traceDirectionRef.current) {
            return prev
          }
          return [...prev, { ...cursor, amp: 0.5 }]
        })
      }
      if (selectionBox) {
        setSelectionBox((prev) =>
          prev
            ? { ...prev, w: pointer.x - prev.x, h: pointer.y - prev.y }
            : { x: pointer.x, y: pointer.y, w: 0, h: 0 },
        )
      }
    },
    [isTracing, selectionBox],
  )

  const endInteraction = useCallback(() => {
    if (activeTool === 'trace' && isTracing) {
      const trace = tracePath.map((point) => [point.time, point.freq] as [number, number])
      const committedId = nextTraceIdRef.current
      nextTraceIdRef.current += 1
      setCommittedTraces((prev) => [...prev, { id: committedId, points: tracePath }])
      setTracePath([])
      setIsTracing(false)
      traceDirectionRef.current = 0
      void onTraceCommit(trace).then(() => {
        setCommittedTraces((prev) => prev.filter((item) => item.id !== committedId))
      })
      return
    }

    if (activeTool === 'erase' && isTracing) {
      const trace = tracePath.map((point) => [point.time, point.freq] as [number, number])
      onEraseCommit(trace)
      setTracePath([])
      setIsTracing(false)
      traceDirectionRef.current = 0
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
  }, [
    activeTool,
    isTracing,
    tracePath,
    onTraceCommit,
    onEraseCommit,
    selectionBox,
    positionToTimeFreq,
    onSelectBoxCommit,
  ])

  return {
    tracePath,
    committedTraces,
    selectionBox,
    beginAt,
    moveAt,
    endInteraction,
  }
}
