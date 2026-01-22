import type { AnalysisSettings } from '../../app/types'

export type AnalysisSettingsModalProps = {
  settings: AnalysisSettings
  onChange: (settings: AnalysisSettings) => void
  onCancel: () => void
  onApply: () => void
}

export function AnalysisSettingsModal({ settings, onChange, onCancel, onApply }: AnalysisSettingsModalProps) {
  return (
    <div className="modal-backdrop">
      <div className="modal square">
        <h2 className="modal-title">Analysis Settings</h2>
        <div className="modal-grid">
          <label>
            Min Frequency (Hz)
            <input
              type="number"
              value={settings.freq_min}
              onChange={(event) => onChange({ ...settings, freq_min: Number(event.target.value) })}
            />
          </label>
          <label>
            Max Frequency (Hz)
            <input
              type="number"
              value={settings.freq_max}
              onChange={(event) => onChange({ ...settings, freq_max: Number(event.target.value) })}
            />
          </label>
          <label>
            Bins per Octave
            <input
              type="number"
              value={settings.bins_per_octave}
              onChange={(event) => onChange({ ...settings, bins_per_octave: Number(event.target.value) })}
            />
          </label>
          <label>
            Preview Max Frequency (Hz)
            <input
              type="number"
              value={settings.preview_freq_max}
              onChange={(event) => onChange({ ...settings, preview_freq_max: Number(event.target.value) })}
            />
          </label>
          <label>
            Preview Bins per Octave
            <input
              type="number"
              value={settings.preview_bins_per_octave}
              onChange={(event) => onChange({ ...settings, preview_bins_per_octave: Number(event.target.value) })}
            />
          </label>
          <label>
            <span className="modal-label-row">
              Wavelet Bandwidth (cmor)
              <button
                type="button"
                className="modal-help"
                title="cmorB-C bandwidth (B) = TIME vs FREQUENCY resolution knob. Higher B => thinner frequency ridges but more time smearing. Tune based on what you want to emphasize (transients vs stable tones). Typical range: 2-12."
              >
                ?
              </button>
            </span>
            <input
              type="number"
              step={0.1}
              value={settings.wavelet_bandwidth}
              onChange={(event) => onChange({ ...settings, wavelet_bandwidth: Number(event.target.value) })}
            />
          </label>
          <label>
            <span className="modal-label-row">
              Wavelet Center Freq (cmor)
              <button
                type="button"
                className="modal-help"
                title="cmorB-C center frequency (C). Larger C = faster oscillation and more high-frequency focus. Lower C for low-frequency focus, higher C for high-frequency focus. Typical range: 0.5-3.0."
              >
                ?
              </button>
            </span>
            <input
              type="number"
              step={0.1}
              value={settings.wavelet_center_freq}
              onChange={(event) => onChange({ ...settings, wavelet_center_freq: Number(event.target.value) })}
            />
          </label>
          <label>
            Time Resolution (ms)
            <input
              type="number"
              value={settings.time_resolution_ms}
              onChange={(event) => onChange({ ...settings, time_resolution_ms: Number(event.target.value) })}
            />
          </label>
          <label>
            Color Map
            <select value={settings.color_map} onChange={(event) => onChange({ ...settings, color_map: event.target.value })}>
              <option value="magma">Magma</option>
              <option value="viridis">Viridis</option>
              <option value="gray">Gray</option>
            </select>
          </label>
          <label>
            Brightness
            <input
              type="number"
              step={0.1}
              value={settings.brightness}
              onChange={(event) => onChange({ ...settings, brightness: Number(event.target.value) })}
            />
          </label>
          <label>
            Contrast
            <input
              type="number"
              step={0.1}
              value={settings.contrast}
              onChange={(event) => onChange({ ...settings, contrast: Number(event.target.value) })}
            />
          </label>
        </div>
        <div className="modal-actions">
          <button onClick={onCancel}>Cancel</button>
          <button className="primary" onClick={onApply}>
            Apply & Re-analyze
          </button>
        </div>
      </div>
    </div>
  )
}
