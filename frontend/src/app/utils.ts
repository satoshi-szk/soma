import type { AnalysisSettings, Partial } from './types'

export const formatDuration = (seconds: number) => {
  const total = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(total / 60)
  const secs = total % 60
  const millis = Math.floor((seconds % 1) * 1000)
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${millis
    .toString()
    .padStart(3, '0')}`
}

export const toPartial = (raw: { id: string; is_muted: boolean; points: number[][] }): Partial => ({
  id: raw.id,
  is_muted: raw.is_muted,
  points: raw.points.map((point) => ({ time: point[0], freq: point[1], amp: point[2] })),
})

export const mapColor = (mapName: AnalysisSettings['color_map'], value: number): [number, number, number] => {
  if (mapName === 'gray') {
    return [value, value, value]
  }
  if (mapName === 'viridis') {
    return interpolateColor(
      value,
      [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 97],
        [253, 231, 37],
      ],
    )
  }
  return interpolateColor(
    value,
    [
      [0, 0, 4],
      [50, 18, 91],
      [121, 40, 130],
      [189, 55, 84],
      [249, 142, 8],
      [252, 253, 191],
    ],
  )
}

const interpolateColor = (value: number, stops: number[][]): [number, number, number] => {
  const t = Math.min(1, Math.max(0, value / 255))
  const scaled = t * (stops.length - 1)
  const idx = Math.floor(scaled)
  const frac = scaled - idx
  const start = stops[idx]
  const end = stops[Math.min(stops.length - 1, idx + 1)]
  const r = Math.round(start[0] + (end[0] - start[0]) * frac)
  const g = Math.round(start[1] + (end[1] - start[1]) * frac)
  const b = Math.round(start[2] + (end[2] - start[2]) * frac)
  return [r, g, b]
}
