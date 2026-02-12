import type { Partial } from './types'

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
  const midi = Math.round(69 + 12 * Math.log2(freq / 440))
  const { name, octave } = midiToNoteParts(midi)
  return `${name}${octave}`
}

export const formatNoteWithCents = (freq: number) => {
  if (!Number.isFinite(freq) || freq <= 0) return '--'
  const midiFloat = 69 + 12 * Math.log2(freq / 440)
  let nearestMidi = Math.round(midiFloat)
  let cents = Math.round((midiFloat - nearestMidi) * 100)
  if (cents >= 100) {
    nearestMidi += 1
    cents = 0
  } else if (cents <= -100) {
    nearestMidi -= 1
    cents = 0
  }
  const { name, octave } = midiToNoteParts(nearestMidi)
  const centsLabel = `${cents >= 0 ? '+' : ''}${cents}c`
  return `${name}${octave} ${centsLabel}`
}

export const toPartial = (raw: { id: string; is_muted: boolean; color?: string; points: number[][] }): Partial => ({
  id: raw.id,
  is_muted: raw.is_muted,
  color: normalizeHexColor(raw.color),
  points: raw.points.map((point) => ({ time: point[0], freq: point[1], amp: point[2] })),
})

const midiToNoteParts = (midi: number) => {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  const name = noteNames[((midi % 12) + 12) % 12]
  const octave = Math.floor(midi / 12) - 1
  return { name, octave }
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
