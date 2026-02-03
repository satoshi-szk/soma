export type ExportModalProps = {
  tab: 'mpe' | 'multitrack' | 'mono' | 'audio' | 'cv'
  pitchBendRange: string
  amplitudeMapping: string
  amplitudeCurve: string
  exportSampleRate: string
  exportBitDepth: string
  exportCvBaseFreq: string
  exportCvFullScaleVolts: string
  exportCvMode: 'mono' | 'poly'
  onTabChange: (tab: 'mpe' | 'multitrack' | 'mono' | 'audio' | 'cv') => void
  onPitchBendChange: (value: string) => void
  onAmplitudeMappingChange: (value: string) => void
  onAmplitudeCurveChange: (value: string) => void
  onSampleRateChange: (value: string) => void
  onBitDepthChange: (value: string) => void
  onCvBaseFreqChange: (value: string) => void
  onCvFullScaleVoltsChange: (value: string) => void
  onCvModeChange: (value: 'mono' | 'poly') => void
  onCancel: () => void
  onExport: () => void
}

export function ExportModal({
  tab,
  pitchBendRange,
  amplitudeMapping,
  amplitudeCurve,
  exportSampleRate,
  exportBitDepth,
  exportCvBaseFreq,
  exportCvFullScaleVolts,
  exportCvMode,
  onTabChange,
  onPitchBendChange,
  onAmplitudeMappingChange,
  onAmplitudeCurveChange,
  onSampleRateChange,
  onBitDepthChange,
  onCvBaseFreqChange,
  onCvFullScaleVoltsChange,
  onCvModeChange,
  onCancel,
  onExport,
}: ExportModalProps) {
  return (
    <div className="modal-backdrop">
      <div className="modal square">
        <h2 className="modal-title">Export</h2>
        <div className="modal-tabs">
          <button className={tab === 'mpe' ? 'active' : ''} onClick={() => onTabChange('mpe')}>
            MPE (MIDI)
          </button>
          <button className={tab === 'multitrack' ? 'active' : ''} onClick={() => onTabChange('multitrack')}>
            Multi-Track MIDI
          </button>
          <button className={tab === 'mono' ? 'active' : ''} onClick={() => onTabChange('mono')}>
            Monophonic MIDI
          </button>
          <button className={tab === 'audio' ? 'active' : ''} onClick={() => onTabChange('audio')}>
            Audio
          </button>
          <button className={tab === 'cv' ? 'active' : ''} onClick={() => onTabChange('cv')}>
            CV
          </button>
        </div>
        <p className="modal-note">
          {tab === 'mpe'
            ? 'MPE: Exports MIDI with per-note expression across dedicated channels.'
            : tab === 'multitrack'
              ? 'Multi-Track: Exports one track per voice, with all tracks using MIDI Channel 1.'
              : tab === 'mono'
                ? 'Monophonic: Exports monophonic MIDI with legato-friendly control behavior.'
                : tab === 'audio'
                  ? 'Audio: Exports the resynthesized sine-wave audio signal.'
                  : 'CV: Exports pitch and amplitude as control-voltage signals.'}
        </p>
        {tab === 'mpe' || tab === 'multitrack' || tab === 'mono' ? (
          <div className="modal-grid">
            <label>
              Pitch Bend Range
              <input
                type="text"
                inputMode="numeric"
                value={pitchBendRange}
                onChange={(event) => onPitchBendChange(event.target.value)}
              />
            </label>
            <label>
              Amplitude Mapping
              <select value={amplitudeMapping} onChange={(event) => onAmplitudeMappingChange(event.target.value)}>
                <option value="pressure">Pressure</option>
                <option value="cc74">CC74</option>
                <option value="cc1">CC1 (Mod Wheel)</option>
              </select>
            </label>
            <label>
              Amplitude Curve
              <select value={amplitudeCurve} onChange={(event) => onAmplitudeCurveChange(event.target.value)}>
                <option value="linear">Linear</option>
                <option value="db">dB</option>
              </select>
            </label>
          </div>
        ) : tab === 'audio' ? (
          <div className="modal-grid">
            <label>
              Sample Rate
              <input
                type="text"
                inputMode="numeric"
                value={exportSampleRate}
                onChange={(event) => onSampleRateChange(event.target.value)}
              />
            </label>
            <label>
              Bit Depth
              <input
                type="text"
                inputMode="numeric"
                value={exportBitDepth}
                onChange={(event) => onBitDepthChange(event.target.value)}
              />
            </label>
          </div>
        ) : (
          <div className="modal-grid">
            <label>
              CV Mode
              <select value={exportCvMode} onChange={(event) => onCvModeChange(event.target.value as 'mono' | 'poly')}>
                <option value="mono">Monophonic</option>
                <option value="poly">Polyphonic</option>
              </select>
            </label>
            <label>
              CV Base Frequency (0V, Hz)
              <input
                type="text"
                inputMode="numeric"
                value={exportCvBaseFreq}
                onChange={(event) => onCvBaseFreqChange(event.target.value)}
              />
            </label>
            <label>
              Amplitude Curve
              <select value={amplitudeCurve} onChange={(event) => onAmplitudeCurveChange(event.target.value)}>
                <option value="linear">Linear</option>
                <option value="db">dB</option>
              </select>
            </label>
            <label>
              CV Full Scale (±V)
              <input
                type="text"
                inputMode="numeric"
                value={exportCvFullScaleVolts}
                onChange={(event) => onCvFullScaleVoltsChange(event.target.value)}
              />
            </label>
            <label>
              Sample Rate
              <input
                type="text"
                inputMode="numeric"
                value={exportSampleRate}
                onChange={(event) => onSampleRateChange(event.target.value)}
              />
            </label>
            <label>
              Bit Depth
              <input
                type="text"
                inputMode="numeric"
                value={exportBitDepth}
                onChange={(event) => onBitDepthChange(event.target.value)}
              />
            </label>
          </div>
        )}
        <div className="modal-actions">
          <button onClick={onCancel}>Cancel</button>
          <button className="primary" onClick={onExport}>
            Export File...
          </button>
        </div>
      </div>
    </div>
  )
}
