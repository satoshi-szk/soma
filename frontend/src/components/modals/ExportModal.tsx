export type ExportModalProps = {
  tab: 'mpe' | 'audio'
  pitchBendRange: number
  amplitudeMapping: string
  exportSampleRate: number
  exportBitDepth: number
  exportType: 'sine' | 'cv'
  onTabChange: (tab: 'mpe' | 'audio') => void
  onPitchBendChange: (value: number) => void
  onAmplitudeMappingChange: (value: string) => void
  onSampleRateChange: (value: number) => void
  onBitDepthChange: (value: number) => void
  onOutputTypeChange: (value: 'sine' | 'cv') => void
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
  onTabChange,
  onPitchBendChange,
  onAmplitudeMappingChange,
  onSampleRateChange,
  onBitDepthChange,
  onOutputTypeChange,
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
          <button className={tab === 'audio' ? 'active' : ''} onClick={() => onTabChange('audio')}>
            Audio / CV
          </button>
        </div>
        {tab === 'mpe' ? (
          <div className="modal-grid">
            <label>
              Pitch Bend Range
              <input type="number" value={pitchBendRange} onChange={(event) => onPitchBendChange(Number(event.target.value))} />
            </label>
            <label>
              Amplitude Mapping
              <select value={amplitudeMapping} onChange={(event) => onAmplitudeMappingChange(event.target.value)}>
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
            <label>
              Sample Rate
              <input type="number" value={exportSampleRate} onChange={(event) => onSampleRateChange(Number(event.target.value))} />
            </label>
            <label>
              Bit Depth
              <input type="number" value={exportBitDepth} onChange={(event) => onBitDepthChange(Number(event.target.value))} />
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
