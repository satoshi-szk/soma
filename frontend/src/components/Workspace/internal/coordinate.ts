import type { SpectrogramPreview } from '../../../app/types'

type TransformParams = {
  preview: SpectrogramPreview | null
  duration: number
  freqMin: number
  freqMax: number
  pan: { x: number; y: number }
  scale: { x: number; y: number }
  contentOffset: { x: number; y: number }
}

export function timeToX(time: number, preview: SpectrogramPreview | null, duration: number): number {
  if (!preview) return 0
  return (time / duration) * preview.width
}

export function freqToY(freq: number, preview: SpectrogramPreview | null, freqMin: number, freqMax: number): number {
  if (!preview) return 0
  const logMin = Math.log(freqMin)
  const logMax = Math.log(freqMax)
  const norm = (logMax - Math.log(Math.max(freq, freqMin))) / (logMax - logMin)
  return norm * preview.height
}

export function positionToTimeFreq(
  x: number,
  y: number,
  { preview, duration, freqMin, freqMax, pan, scale, contentOffset }: TransformParams
): { time: number; freq: number } {
  if (!preview) return { time: 0, freq: freqMin }
  const contentX = (x - pan.x - contentOffset.x) / scale.x
  const contentY = (y - pan.y - contentOffset.y) / scale.y
  const time = (contentX / preview.width) * duration
  const logMin = Math.log(freqMin)
  const logMax = Math.log(freqMax)
  const ratio = Math.min(1, Math.max(0, contentY / preview.height))
  const freq = Math.exp(logMax - ratio * (logMax - logMin))
  return { time, freq }
}

export function positionToTime(x: number, { preview, duration, pan, scale, contentOffset }: TransformParams): number {
  if (!preview) return 0
  const contentX = (x - pan.x - contentOffset.x) / scale.x
  const time = (contentX / preview.width) * duration
  return Math.max(0, Math.min(duration, time))
}

