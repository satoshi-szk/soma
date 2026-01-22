import type { AnalysisSettings, Partial } from './types'

export const formatDuration = (seconds: number) => {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const millis = Math.floor((seconds % 1) * 1000)
  return `${hours.toString().padStart(2, '0')}:${minutes
    .toString()
    .padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`
}

export const formatNote = (freq: number) => {
  if (!Number.isFinite(freq) || freq <= 0) return '--'
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  const midi = Math.round(69 + 12 * Math.log2(freq / 440))
  const name = noteNames[((midi % 12) + 12) % 12]
  const octave = Math.floor(midi / 12) - 1
  return `${name}${octave}`
}

export const toPartial = (raw: { id: string; is_muted: boolean; color?: string; points: number[][] }): Partial => ({
  id: raw.id,
  is_muted: raw.is_muted,
  color: normalizeHexColor(raw.color),
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

const normalizeHexColor = (value?: string): string => {
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^#?[0-9a-fA-F]{6}$/.test(trimmed)) {
      return trimmed.startsWith('#') ? trimmed : `#${trimmed}`
    }
  }
  return '#f8d19a'
}
