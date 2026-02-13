import { useEffect, useState } from 'react'
import type { AnalysisSettings } from '../../app/types'

type DraftState = {
  [key: string]: string
}

const intFields = new Set<string>(['bins_per_octave'])
const windowSizeScaleOptions = ['0.25', '0.5', '1', '2', '4'] as const
const spectrogramMethodOptions = [
  { value: 'multires_stft', label: 'Multi-Res STFT' },
  { value: 'reassigned_stft', label: 'Reassigned STFT' },
] as const

const toDraft = (settings: AnalysisSettings): DraftState => ({
  spectrogram_freq_min: String(settings.spectrogram.freq_min),
  spectrogram_freq_max: String(settings.spectrogram.freq_max),
  preview_freq_max: String(settings.spectrogram.preview_freq_max),
  method: settings.spectrogram.method,
  multires_blend_octaves: String(settings.spectrogram.multires_blend_octaves),
  multires_window_size_scale: String(settings.spectrogram.multires_window_size_scale),
  reassigned_ref_power: String(settings.spectrogram.reassigned_ref_power),
  gain: String(settings.spectrogram.gain),
  min_db: String(settings.spectrogram.min_db),
  max_db: String(settings.spectrogram.max_db),
  gamma: String(settings.spectrogram.gamma),
  snap_freq_min: String(settings.snap.freq_min),
  snap_freq_max: String(settings.snap.freq_max),
  bins_per_octave: String(settings.snap.bins_per_octave),
  time_resolution_ms: String(settings.snap.time_resolution_ms),
  wavelet_bandwidth: String(settings.snap.wavelet_bandwidth),
  wavelet_center_freq: String(settings.snap.wavelet_center_freq),
})

const isTransientValue = (value: string) => {
  const trimmed = value.trim()
  return trimmed === '' || trimmed === '-' || trimmed === '.' || trimmed === '-.'
}

const parseDraftNumber = (key: string, value: string): number | null => {
  if (isTransientValue(value)) return null
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  if (intFields.has(key)) return Math.round(parsed)
  return parsed
}

export type AnalysisSettingsSidebarProps = {
  settings: AnalysisSettings
  disabled: boolean
  onClose: () => void
  onApply: (settings: AnalysisSettings) => Promise<void> | void
}

export function AnalysisSettingsSidebar({ settings, disabled, onClose, onApply }: AnalysisSettingsSidebarProps) {
  const [draft, setDraft] = useState<DraftState>(() => toDraft(settings))

  useEffect(() => {
    setDraft(toDraft(settings))
  }, [settings])

  const updateField = (key: string, value: string) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  const normalizeField = (key: string, fallback: number) => {
    const parsed = parseDraftNumber(key, draft[key] ?? '')
    setDraft((prev) => ({ ...prev, [key]: String(parsed ?? fallback) }))
  }

  const buildSettings = (): AnalysisSettings => {
    const spectrogram = { ...settings.spectrogram }
    const snap = { ...settings.snap }
    spectrogram.method = draft.method === 'reassigned_stft' ? 'reassigned_stft' : 'multires_stft'

    const setIf = (key: string, apply: (value: number) => void) => {
      const parsed = parseDraftNumber(key, draft[key] ?? '')
      if (parsed !== null) apply(parsed)
    }

    setIf('spectrogram_freq_min', (value) => {
      spectrogram.freq_min = value
    })
    setIf('spectrogram_freq_max', (value) => {
      spectrogram.freq_max = value
    })
    setIf('preview_freq_max', (value) => {
      spectrogram.preview_freq_max = value
    })
    setIf('multires_blend_octaves', (value) => {
      spectrogram.multires_blend_octaves = value
    })
    setIf('multires_window_size_scale', (value) => {
      spectrogram.multires_window_size_scale = value
    })
    setIf('reassigned_ref_power', (value) => {
      spectrogram.reassigned_ref_power = Math.max(0, value)
    })
    setIf('gain', (value) => {
      spectrogram.gain = value
    })
    setIf('min_db', (value) => {
      spectrogram.min_db = value
    })
    setIf('max_db', (value) => {
      spectrogram.max_db = value
    })
    setIf('gamma', (value) => {
      spectrogram.gamma = value
    })

    setIf('snap_freq_min', (value) => {
      snap.freq_min = value
    })
    setIf('snap_freq_max', (value) => {
      snap.freq_max = value
    })
    setIf('bins_per_octave', (value) => {
      snap.bins_per_octave = value
    })
    setIf('time_resolution_ms', (value) => {
      snap.time_resolution_ms = value
    })
    setIf('wavelet_bandwidth', (value) => {
      snap.wavelet_bandwidth = value
    })
    setIf('wavelet_center_freq', (value) => {
      snap.wavelet_center_freq = value
    })

    return { spectrogram, snap }
  }

  const inputClass = disabled
    ? 'rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[11px] text-[var(--muted)] opacity-60'
    : 'rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[11px] text-[var(--ink)]'
  const isReassigned = draft.method === 'reassigned_stft'

  return (
    <aside className="panel flex h-full w-[320px] min-w-[320px] flex-col rounded-none px-4 py-3">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="m-0 text-[12px] font-semibold uppercase tracking-[0.16em] text-[var(--ink)]">Analysis Settings</h2>
        <button
          type="button"
          className="rounded-md border border-[var(--panel-border)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]"
          onClick={onClose}
        >
          Close
        </button>
      </div>

      <div className="flex-1 overflow-y-auto pr-1">
        <section className="mb-4">
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Spectrogram Settings</h3>
          <div className="grid gap-2">
            <label className="text-[11px] text-[var(--ink)]">
              Spectrogram Method
              <select
                className={`mt-1 w-full ${inputClass}`}
                value={draft.method}
                onChange={(event) => updateField('method', event.target.value)}
                disabled={disabled}
              >
                {spectrogramMethodOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Min Frequency (Hz)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.spectrogram_freq_min}
                onChange={(event) => updateField('spectrogram_freq_min', event.target.value)}
                onBlur={() => normalizeField('spectrogram_freq_min', settings.spectrogram.freq_min)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Max Frequency (Hz)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.spectrogram_freq_max}
                onChange={(event) => updateField('spectrogram_freq_max', event.target.value)}
                onBlur={() => normalizeField('spectrogram_freq_max', settings.spectrogram.freq_max)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Preview Max Frequency (Hz)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.preview_freq_max}
                onChange={(event) => updateField('preview_freq_max', event.target.value)}
                onBlur={() => normalizeField('preview_freq_max', settings.spectrogram.preview_freq_max)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Multi-Res Blend Width (oct)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.multires_blend_octaves}
                onChange={(event) => updateField('multires_blend_octaves', event.target.value)}
                onBlur={() => normalizeField('multires_blend_octaves', settings.spectrogram.multires_blend_octaves)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Window Size Scale
              <select
                className={`mt-1 w-full ${inputClass}`}
                value={draft.multires_window_size_scale}
                onChange={(event) => updateField('multires_window_size_scale', event.target.value)}
                disabled={disabled}
              >
                {windowSizeScaleOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Reassigned Ref Power
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.reassigned_ref_power}
                onChange={(event) => updateField('reassigned_ref_power', event.target.value)}
                onBlur={() => normalizeField('reassigned_ref_power', settings.spectrogram.reassigned_ref_power)}
                disabled={disabled || !isReassigned}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Gain
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.gain}
                onChange={(event) => updateField('gain', event.target.value)}
                onBlur={() => normalizeField('gain', settings.spectrogram.gain)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Min dB
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.min_db}
                onChange={(event) => updateField('min_db', event.target.value)}
                onBlur={() => normalizeField('min_db', settings.spectrogram.min_db)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Max dB
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.max_db}
                onChange={(event) => updateField('max_db', event.target.value)}
                onBlur={() => normalizeField('max_db', settings.spectrogram.max_db)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Gamma
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.gamma}
                onChange={(event) => updateField('gamma', event.target.value)}
                onBlur={() => normalizeField('gamma', settings.spectrogram.gamma)}
                disabled={disabled}
              />
            </label>
          </div>
        </section>

        <section>
          <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Snap Settings</h3>
          <div className="grid gap-2">
            <label className="text-[11px] text-[var(--ink)]">
              Min Frequency (Hz)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.snap_freq_min}
                onChange={(event) => updateField('snap_freq_min', event.target.value)}
                onBlur={() => normalizeField('snap_freq_min', settings.snap.freq_min)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Max Frequency (Hz)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.snap_freq_max}
                onChange={(event) => updateField('snap_freq_max', event.target.value)}
                onBlur={() => normalizeField('snap_freq_max', settings.snap.freq_max)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Bins per Octave
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.bins_per_octave}
                onChange={(event) => updateField('bins_per_octave', event.target.value)}
                onBlur={() => normalizeField('bins_per_octave', settings.snap.bins_per_octave)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Time Resolution (ms)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.time_resolution_ms}
                onChange={(event) => updateField('time_resolution_ms', event.target.value)}
                onBlur={() => normalizeField('time_resolution_ms', settings.snap.time_resolution_ms)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Wavelet Bandwidth (cmor)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.wavelet_bandwidth}
                onChange={(event) => updateField('wavelet_bandwidth', event.target.value)}
                onBlur={() => normalizeField('wavelet_bandwidth', settings.snap.wavelet_bandwidth)}
                disabled={disabled}
              />
            </label>
            <label className="text-[11px] text-[var(--ink)]">
              Wavelet Center Freq (cmor)
              <input
                className={`mt-1 w-full ${inputClass}`}
                value={draft.wavelet_center_freq}
                onChange={(event) => updateField('wavelet_center_freq', event.target.value)}
                onBlur={() => normalizeField('wavelet_center_freq', settings.snap.wavelet_center_freq)}
                disabled={disabled}
              />
            </label>
          </div>
        </section>
      </div>

      <div className="mt-3 flex justify-end gap-2">
        <button
          type="button"
          className="rounded-md border border-[var(--panel-border)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]"
          onClick={() => setDraft(toDraft(settings))}
          disabled={disabled}
        >
          Reset
        </button>
        <button
          type="button"
          className="rounded-md bg-[var(--accent)] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-white disabled:opacity-60"
          onClick={() => onApply(buildSettings())}
          disabled={disabled}
        >
          Apply
        </button>
      </div>
    </aside>
  )
}
