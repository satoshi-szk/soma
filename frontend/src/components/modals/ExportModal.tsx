export type ExportModalProps = {
  tab: 'mpe' | 'multitrack' | 'mono' | 'audio'
  pitchBendRange: string
  amplitudeMapping: string
  exportSampleRate: string
  exportBitDepth: string
  exportType: 'sine' | 'cv'
  exportCvBaseFreq: string
  exportCvFullScaleVolts: string
  onTabChange: (tab: 'mpe' | 'multitrack' | 'mono' | 'audio') => void
  onPitchBendChange: (value: string) => void
  onAmplitudeMappingChange: (value: string) => void
  onSampleRateChange: (value: string) => void
  onBitDepthChange: (value: string) => void
  onOutputTypeChange: (value: 'sine' | 'cv') => void
  onCvBaseFreqChange: (value: string) => void
  onCvFullScaleVoltsChange: (value: string) => void
  onCancel: () => void
  onExport: () => void
}

export function ExportModal({
  tab,
  pitchBendRange,
  amplitudeMapping,
  exportSampleRate,
  exportBitDepth,
  exportType,
  exportCvBaseFreq,
  exportCvFullScaleVolts,
  onTabChange,
  onPitchBendChange,
  onAmplitudeMappingChange,
  onSampleRateChange,
  onBitDepthChange,
  onOutputTypeChange,
  onCvBaseFreqChange,
  onCvFullScaleVoltsChange,
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
            Audio / CV
          </button>
        </div>
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
                <option value="velocity">Velocity</option>
                <option value="pressure">Pressure</option>
                <option value="cc74">CC74</option>
              </select>
            </label>
          </div>
        ) : (
          <div className="modal-grid">
            <label>
              Output Type
              <select value={exportType} onChange={(event) => onOutputTypeChange(event.target.value as 'sine' | 'cv')}>
                <option value="sine">Sine Synthesis</option>
                <option value="cv">CV Control</option>
              </select>
            </label>
            {exportType === 'cv' ? (
              <label>
                CV Base Frequency (0V, Hz)
                <input
                  type="text"
                  inputMode="numeric"
                  value={exportCvBaseFreq}
                  onChange={(event) => onCvBaseFreqChange(event.target.value)}
                />
              </label>
            ) : null}
            {exportType === 'cv' ? (
              <label>
                CV Full Scale (±V)
                <input
                  type="text"
                  inputMode="numeric"
                  value={exportCvFullScaleVolts}
                  onChange={(event) => onCvFullScaleVoltsChange(event.target.value)}
                />
              </label>
            ) : null}
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
