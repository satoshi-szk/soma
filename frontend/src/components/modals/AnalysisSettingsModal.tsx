import type { AnalysisSettings } from '../../app/types'
import { useEffect, useMemo, useState } from 'react'

type DraftSettings = Record<keyof AnalysisSettings, string>
type NumericKey = keyof AnalysisSettings

const intFields = new Set<NumericKey>(['bins_per_octave', 'preview_bins_per_octave'])
const numericKeys: NumericKey[] = [
  'freq_min',
  'freq_max',
  'bins_per_octave',
  'time_resolution_ms',
  'preview_freq_max',
  'preview_bins_per_octave',
  'wavelet_bandwidth',
  'wavelet_center_freq',
  'gain',
  'min_db',
  'max_db',
  'gamma',
]

const toDraft = (settings: AnalysisSettings): DraftSettings => ({
  freq_min: String(settings.freq_min),
  freq_max: String(settings.freq_max),
  bins_per_octave: String(settings.bins_per_octave),
  time_resolution_ms: String(settings.time_resolution_ms),
  preview_freq_max: String(settings.preview_freq_max),
  preview_bins_per_octave: String(settings.preview_bins_per_octave),
  wavelet_bandwidth: String(settings.wavelet_bandwidth),
  wavelet_center_freq: String(settings.wavelet_center_freq),
  gain: String(settings.gain),
  min_db: String(settings.min_db),
  max_db: String(settings.max_db),
  gamma: String(settings.gamma),
})

const isTransientValue = (value: string) => {
  const trimmed = value.trim()
  return trimmed === '' || trimmed === '-' || trimmed === '.' || trimmed === '-.'
}

const parseValue = (key: NumericKey, value: string): number | null => {
  if (isTransientValue(value)) return null
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return null
  if (intFields.has(key)) {
    return Math.round(parsed)
  }
  return parsed
}

const helpText: Record<string, string> = {
  common: 'Base range shared by both Snap and Preview.',
  preview: 'Settings for the lightweight spectrogram used for display.',
  snap: 'Settings for automatic snap analysis (CWT).',
  freq_min: 'Minimum frequency for display/analysis. Lower values may increase noise.',
  freq_max: 'Maximum frequency for display/analysis. Higher values may reduce density.',
  preview_freq_max: 'Upper frequency limit for preview only. Higher values increase render cost.',
  preview_bins_per_octave: 'Frequency resolution for preview. Higher values provide finer detail.',
  gain: 'Linear gain multiplier after dB normalization.',
  min_db: 'Minimum dB shown (values below are clipped to black).',
  max_db: 'Maximum dB shown (values above are clipped to white).',
  gamma: 'Gamma correction for mid-tone balance.',
  bins_per_octave: 'Frequency resolution for snap analysis. Higher values provide finer detail.',
  time_resolution_ms: 'Resampling interval for snap points. Smaller values increase point count.',
  wavelet_bandwidth: 'Higher B sharpens frequency detail but smears time. Typical range: 2-12.',
  wavelet_center_freq: 'Higher C emphasizes higher bands; lower C emphasizes lower bands. Typical range: 0.5-3.0.',
}

export type AnalysisSettingsModalProps = {
  settings: AnalysisSettings
  onCancel: () => void
  onApply: (settings: AnalysisSettings) => void
}

export function AnalysisSettingsModal({ settings, onCancel, onApply }: AnalysisSettingsModalProps) {
  const [draft, setDraft] = useState<DraftSettings>(() => toDraft(settings))
  const [helpOpen, setHelpOpen] = useState<Record<string, boolean>>({})

  useEffect(() => {
    setDraft(toDraft(settings))
  }, [settings])

  const handleNumberChange = (key: NumericKey, value: string) => {
    setDraft((prev) => ({ ...prev, [key]: value }))
  }

  const handleNumberBlur = (key: NumericKey) => {
    const parsed = parseValue(key, draft[key])
    if (parsed === null) {
      setDraft((prev) => ({ ...prev, [key]: String(settings[key]) }))
      return
    }
    setDraft((prev) => ({ ...prev, [key]: String(parsed) }))
  }

  const buildSettingsFromDraft = () => {
    const nextSettings: AnalysisSettings = { ...settings }
    numericKeys.forEach((key) => {
      const parsed = parseValue(key, draft[key])
      if (parsed !== null) {
        nextSettings[key] = parsed
      }
    })
    return nextSettings
  }

  const toggleHelp = (key: string) => {
    setHelpOpen((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  const previewNote = useMemo(() => helpText.preview, [])

  return (
    <div className="modal-backdrop">
      <div className="modal square">
        <h2 className="modal-title">Analysis Settings</h2>
        <div className="modal-section">
          <div className="modal-section-header">
            <div className="modal-section-title">Common</div>
            <button type="button" className="modal-help" onClick={() => toggleHelp('common')}>
              ?
            </button>
          </div>
          {helpOpen.common ? <div className="modal-help-detail">{helpText.common}</div> : null}
          <div className="modal-grid">
            <label>
              <span className="modal-label-row">
                Min Frequency (Hz)
                <button type="button" className="modal-help" onClick={() => toggleHelp('freq_min')}>
                  ?
                </button>
              </span>
              {helpOpen.freq_min ? <span className="modal-help-detail">{helpText.freq_min}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.freq_min}
                onChange={(event) => handleNumberChange('freq_min', event.target.value)}
                onBlur={() => handleNumberBlur('freq_min')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Max Frequency (Hz)
                <button type="button" className="modal-help" onClick={() => toggleHelp('freq_max')}>
                  ?
                </button>
              </span>
              {helpOpen.freq_max ? <span className="modal-help-detail">{helpText.freq_max}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.freq_max}
                onChange={(event) => handleNumberChange('freq_max', event.target.value)}
                onBlur={() => handleNumberBlur('freq_max')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Wavelet Bandwidth (cmor)
                <button type="button" className="modal-help" onClick={() => toggleHelp('wavelet_bandwidth')}>
                  ?
                </button>
              </span>
              {helpOpen.wavelet_bandwidth ? (
                <span className="modal-help-detail">{helpText.wavelet_bandwidth}</span>
              ) : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.wavelet_bandwidth}
                onChange={(event) => handleNumberChange('wavelet_bandwidth', event.target.value)}
                onBlur={() => handleNumberBlur('wavelet_bandwidth')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Wavelet Center Freq (cmor)
                <button type="button" className="modal-help" onClick={() => toggleHelp('wavelet_center_freq')}>
                  ?
                </button>
              </span>
              {helpOpen.wavelet_center_freq ? (
                <span className="modal-help-detail">{helpText.wavelet_center_freq}</span>
              ) : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.wavelet_center_freq}
                onChange={(event) => handleNumberChange('wavelet_center_freq', event.target.value)}
                onBlur={() => handleNumberBlur('wavelet_center_freq')}
              />
            </label>
          </div>
        </div>
        <div className="modal-section">
          <div className="modal-section-header">
            <div className="modal-section-title">Preview</div>
            <button type="button" className="modal-help" onClick={() => toggleHelp('preview')}>
              ?
            </button>
          </div>
          {helpOpen.preview ? <div className="modal-help-detail">{previewNote}</div> : null}
          <div className="modal-grid">
            <label>
              <span className="modal-label-row">
                Preview Max Frequency (Hz)
                <button type="button" className="modal-help" onClick={() => toggleHelp('preview_freq_max')}>
                  ?
                </button>
              </span>
              {helpOpen.preview_freq_max ? <span className="modal-help-detail">{helpText.preview_freq_max}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.preview_freq_max}
                onChange={(event) => handleNumberChange('preview_freq_max', event.target.value)}
                onBlur={() => handleNumberBlur('preview_freq_max')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Preview Bins per Octave
                <button type="button" className="modal-help" onClick={() => toggleHelp('preview_bins_per_octave')}>
                  ?
                </button>
              </span>
              {helpOpen.preview_bins_per_octave ? (
                <span className="modal-help-detail">{helpText.preview_bins_per_octave}</span>
              ) : null}
              <input
                type="text"
                inputMode="numeric"
                value={draft.preview_bins_per_octave}
                onChange={(event) => handleNumberChange('preview_bins_per_octave', event.target.value)}
                onBlur={() => handleNumberBlur('preview_bins_per_octave')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Gain
                <button type="button" className="modal-help" onClick={() => toggleHelp('gain')}>
                  ?
                </button>
              </span>
              {helpOpen.gain ? <span className="modal-help-detail">{helpText.gain}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.gain}
                onChange={(event) => handleNumberChange('gain', event.target.value)}
                onBlur={() => handleNumberBlur('gain')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Min dB
                <button type="button" className="modal-help" onClick={() => toggleHelp('min_db')}>
                  ?
                </button>
              </span>
              {helpOpen.min_db ? <span className="modal-help-detail">{helpText.min_db}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.min_db}
                onChange={(event) => handleNumberChange('min_db', event.target.value)}
                onBlur={() => handleNumberBlur('min_db')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Max dB
                <button type="button" className="modal-help" onClick={() => toggleHelp('max_db')}>
                  ?
                </button>
              </span>
              {helpOpen.max_db ? <span className="modal-help-detail">{helpText.max_db}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.max_db}
                onChange={(event) => handleNumberChange('max_db', event.target.value)}
                onBlur={() => handleNumberBlur('max_db')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Gamma
                <button type="button" className="modal-help" onClick={() => toggleHelp('gamma')}>
                  ?
                </button>
              </span>
              {helpOpen.gamma ? <span className="modal-help-detail">{helpText.gamma}</span> : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.gamma}
                onChange={(event) => handleNumberChange('gamma', event.target.value)}
                onBlur={() => handleNumberBlur('gamma')}
              />
            </label>
          </div>
        </div>
        <div className="modal-section">
          <div className="modal-section-header">
            <div className="modal-section-title">Snap</div>
            <button type="button" className="modal-help" onClick={() => toggleHelp('snap')}>
              ?
            </button>
          </div>
          {helpOpen.snap ? <div className="modal-help-detail">{helpText.snap}</div> : null}
          <div className="modal-grid">
            <label>
              <span className="modal-label-row">
                Bins per Octave
                <button type="button" className="modal-help" onClick={() => toggleHelp('bins_per_octave')}>
                  ?
                </button>
              </span>
              {helpOpen.bins_per_octave ? <span className="modal-help-detail">{helpText.bins_per_octave}</span> : null}
              <input
                type="text"
                inputMode="numeric"
                value={draft.bins_per_octave}
                onChange={(event) => handleNumberChange('bins_per_octave', event.target.value)}
                onBlur={() => handleNumberBlur('bins_per_octave')}
              />
            </label>
            <label>
              <span className="modal-label-row">
                Time Resolution (ms)
                <button type="button" className="modal-help" onClick={() => toggleHelp('time_resolution_ms')}>
                  ?
                </button>
              </span>
              {helpOpen.time_resolution_ms ? (
                <span className="modal-help-detail">{helpText.time_resolution_ms}</span>
              ) : null}
              <input
                type="text"
                inputMode="decimal"
                value={draft.time_resolution_ms}
                onChange={(event) => handleNumberChange('time_resolution_ms', event.target.value)}
                onBlur={() => handleNumberBlur('time_resolution_ms')}
              />
            </label>
          </div>
        </div>
        <div className="modal-actions">
          <button onClick={onCancel}>Cancel</button>
          <button
            className="primary"
            onClick={() => {
              const nextSettings = buildSettingsFromDraft()
              onApply(nextSettings)
            }}
          >
            Apply & Re-analyze
          </button>
        </div>
      </div>
    </div>
  )
}
