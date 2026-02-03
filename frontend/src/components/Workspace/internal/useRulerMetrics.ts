import { useCallback, useMemo } from 'react'
import type { Partial, SpectrogramPreview } from '../../../app/types'

type Params = {
  preview: SpectrogramPreview | null
  freqMin: number
  freqMax: number
  zoomY: number
  spectrogramAreaHeight: number
  contentOffsetX: number
  contentOffsetY: number
  pan: { x: number; y: number }
  scale: { x: number; y: number }
  stageSize: { width: number; height: number }
  duration: number
  automationContentHeight: number
  partials: Partial[]
}

export function useRulerMetrics({
  preview,
  freqMin,
  freqMax,
  zoomY,
  spectrogramAreaHeight,
  contentOffsetX,
  contentOffsetY,
  pan,
  scale,
  stageSize,
  duration,
  automationContentHeight,
  partials,
}: Params) {
  const pxPerOctave = useMemo(() => {
    if (!preview) return 0
    const effectiveHeight = spectrogramAreaHeight - contentOffsetY
    const totalOctaves = Math.log2(freqMax / freqMin)
    const basePxPerOctave = effectiveHeight / totalOctaves
    return basePxPerOctave * zoomY
  }, [preview, spectrogramAreaHeight, contentOffsetY, freqMax, freqMin, zoomY])

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

  const visibleRange = useMemo(() => {
    if (!preview) return { start: 0, end: 0 }
    const start = Math.max(0, ((-pan.x - contentOffsetX) / scale.x / preview.width) * duration)
    const end = Math.min(
      duration,
      ((stageSize.width - pan.x - contentOffsetX) / scale.x / preview.width) * duration,
    )
    return { start, end }
  }, [preview, pan, scale, stageSize, duration, contentOffsetX])

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

  const maxAmp = useMemo(() => {
    let max = 0
    for (const partial of partials) {
      for (const point of partial.points) {
        if (point.amp > max) max = point.amp
      }
    }
    return max > 0 ? max : 1
  }, [partials])

  const ampToLaneY = useCallback(
    (amp: number) => {
      const normalized = Math.min(1, Math.max(0, amp / maxAmp))
      return automationContentHeight - normalized * automationContentHeight
    },
    [automationContentHeight, maxAmp],
  )

  return { pxPerOctave, freqRulerMarks, timeMarks, ampToLaneY }
}
