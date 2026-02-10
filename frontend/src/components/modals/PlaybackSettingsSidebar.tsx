type TimeStretchMode = 'native' | 'librosa'
type PlaybackMode = 'audio' | 'midi'
type MidiMode = 'mpe' | 'multitrack' | 'mono'
type MidiAmplitudeMapping = 'velocity' | 'pressure' | 'cc74' | 'cc1'
type MidiAmplitudeCurve = 'linear' | 'db'
type MidiCcUpdateRateHz = 50 | 100 | 200 | 400 | 800

export type PlaybackSettingsSidebarProps = {
  playbackMode: PlaybackMode
  mixValue: number
  speedPresetIndex: number
  speedValue: number
  timeStretchMode: TimeStretchMode
  midiMode: MidiMode
  midiOutputName: string
  midiOutputs: string[]
  midiPitchBendRange: string
  midiAmplitudeMapping: MidiAmplitudeMapping
  midiAmplitudeCurve: MidiAmplitudeCurve
  midiCcUpdateRateHz: MidiCcUpdateRateHz
  midiBpm: string
  controlsDisabled: boolean
  onClose: () => void
  onPlaybackModeChange: (mode: PlaybackMode) => void
  onMixChange: (value: number) => void
  onMixCommit: (value: number) => void
  onSpeedChange: (value: number) => void
  onSpeedCommit: (value: number) => void
  onTimeStretchModeChange: (mode: TimeStretchMode) => void
  onMidiModeChange: (value: MidiMode) => void
  onMidiOutputChange: (value: string) => void
  onMidiOutputsRefresh: () => void
  onMidiPitchBendRangeChange: (value: string) => void
  onMidiAmplitudeMappingChange: (value: MidiAmplitudeMapping) => void
  onMidiAmplitudeCurveChange: (value: MidiAmplitudeCurve) => void
  onMidiCcUpdateRateChange: (value: MidiCcUpdateRateHz) => void
  onMidiBpmChange: (value: string) => void
}

export function PlaybackSettingsSidebar({
  playbackMode,
  mixValue,
  speedPresetIndex,
  speedValue,
  timeStretchMode,
  midiMode,
  midiOutputName,
  midiOutputs,
  midiPitchBendRange,
  midiAmplitudeMapping,
  midiAmplitudeCurve,
  midiCcUpdateRateHz,
  midiBpm,
  controlsDisabled,
  onClose,
  onPlaybackModeChange,
  onMixChange,
  onMixCommit,
  onSpeedChange,
  onSpeedCommit,
  onTimeStretchModeChange,
  onMidiModeChange,
  onMidiOutputChange,
  onMidiOutputsRefresh,
  onMidiPitchBendRangeChange,
  onMidiAmplitudeMappingChange,
  onMidiAmplitudeCurveChange,
  onMidiCcUpdateRateChange,
  onMidiBpmChange,
}: PlaybackSettingsSidebarProps) {
  const lockedClass = controlsDisabled ? 'opacity-55' : ''
  const inputClass = controlsDisabled
    ? 'h-1 w-full cursor-not-allowed accent-[var(--muted)]'
    : 'h-1 w-full accent-[var(--accent)]'
  const selectClass = controlsDisabled
    ? 'cursor-not-allowed rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[11px] text-[var(--muted)] opacity-60'
    : 'rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[11px] text-[var(--ink)]'

  return (
    <aside className="panel flex h-full w-[320px] min-w-[320px] flex-col rounded-none px-4 py-3">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="m-0 text-[12px] font-semibold uppercase tracking-[0.16em] text-[var(--ink)]">Playback Settings</h2>
        <button
          type="button"
          className="rounded-md border border-[var(--panel-border)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]"
          onClick={onClose}
        >
          Close
        </button>
      </div>
      <div className="mb-3 grid grid-cols-2 gap-1 rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] p-1">
        <button
          type="button"
          className={`rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${
            playbackMode === 'audio' ? 'bg-[var(--accent)] text-white' : 'text-[var(--muted)]'
          }`}
          onClick={() => onPlaybackModeChange('audio')}
        >
          Audio
        </button>
        <button
          type="button"
          className={`rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${
            playbackMode === 'midi' ? 'bg-[var(--accent)] text-white' : 'text-[var(--muted)]'
          }`}
          onClick={() => onPlaybackModeChange('midi')}
        >
          MIDI
        </button>
      </div>
      {controlsDisabled ? (
        <div className="mb-3 rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--accent-warm)]">
          Some settings are unavailable during playback
        </div>
      ) : null}

      {playbackMode === 'audio' ? (
        <div className="grid gap-4">
          <div className="flex flex-col">
            <span className="text-[11px] tracking-normal text-[var(--ink)]">Original {100 - mixValue}% / Resynth {mixValue}%</span>
            <input
              aria-label="Original and resynth mix"
              className="h-1 w-full accent-[var(--accent)]"
              type="range"
              min={0}
              max={100}
              step={1}
              value={mixValue}
              onChange={(event) => onMixChange(Number(event.target.value))}
              onMouseUp={(event) => onMixCommit(Number((event.target as HTMLInputElement).value))}
              onTouchEnd={(event) => onMixCommit(Number((event.target as HTMLInputElement).value))}
              onKeyUp={(event) => onMixCommit(Number((event.target as HTMLInputElement).value))}
            />
          </div>
          <div className={`flex flex-col ${lockedClass}`}>
            <span className="text-[11px] tracking-normal text-[var(--ink)]">
              Speed {formatSpeedLabel(speedValue)} {controlsDisabled ? '(Unavailable)' : ''}
            </span>
            <input
              aria-label="Playback speed"
              className={inputClass}
              type="range"
              min={0}
              max={6}
              step={1}
              value={speedPresetIndex}
              disabled={controlsDisabled}
              onChange={(event) => onSpeedChange(Number(event.target.value))}
              onMouseUp={(event) => onSpeedCommit(Number((event.target as HTMLInputElement).value))}
              onTouchEnd={(event) => onSpeedCommit(Number((event.target as HTMLInputElement).value))}
              onKeyUp={(event) => onSpeedCommit(Number((event.target as HTMLInputElement).value))}
            />
          </div>
          <div className={`flex flex-col gap-1 ${lockedClass}`}>
            <span className="text-[11px] tracking-normal text-[var(--ink)]">Stretch {controlsDisabled ? '(Unavailable)' : ''}</span>
            <select
              aria-label="Time stretch mode"
              className={selectClass}
              value={timeStretchMode}
              disabled={controlsDisabled}
              onChange={(event) => onTimeStretchModeChange(event.target.value as TimeStretchMode)}
            >
              <option value="librosa">Librosa</option>
              <option value="native">Native</option>
            </select>
          </div>
        </div>
      ) : (
        <div className="grid gap-3">
          <div className={`flex flex-col gap-1 ${lockedClass}`}>
            <span className="text-[11px] tracking-normal text-[var(--ink)]">MIDI Format</span>
            <select
              className={selectClass}
              value={midiMode}
              disabled={controlsDisabled}
              onChange={(event) => onMidiModeChange(event.target.value as MidiMode)}
            >
              <option value="mpe">MPE</option>
              <option value="multitrack">Multi-Track MIDI</option>
              <option value="mono">Monophonic MIDI</option>
            </select>
          </div>
          <div className="flex items-end gap-2">
            <label className="flex flex-1 flex-col gap-1 text-[11px] tracking-normal text-[var(--ink)]">
              MIDI Output Device
              <select
                className={controlsDisabled ? `${selectClass} flex-1` : `${selectClass} flex-1`}
                value={midiOutputName}
                disabled={controlsDisabled}
                onChange={(event) => onMidiOutputChange(event.target.value)}
              >
                <option value="">Select a MIDI output...</option>
                {midiOutputs.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="h-[31px] rounded-md border border-[var(--panel-border)] px-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--muted)]"
              disabled={controlsDisabled}
              onClick={onMidiOutputsRefresh}
            >
              Refresh
            </button>
          </div>
          <label className="flex flex-col gap-1 text-[11px] tracking-normal text-[var(--ink)]">
            Pitch Bend Range
            <input
              type="text"
              inputMode="numeric"
              className="rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[11px] text-[var(--ink)]"
              value={midiPitchBendRange}
              onChange={(event) => onMidiPitchBendRangeChange(event.target.value)}
            />
          </label>
          <label className="flex flex-col gap-1 text-[11px] tracking-normal text-[var(--ink)]">
            Amplitude Mapping
            <select
              className={selectClass}
              value={midiAmplitudeMapping}
              disabled={controlsDisabled}
              onChange={(event) => onMidiAmplitudeMappingChange(event.target.value as MidiAmplitudeMapping)}
            >
              <option value="pressure">Pressure</option>
              <option value="cc74">CC74</option>
              <option value="cc1">CC1 (Mod Wheel)</option>
              <option value="velocity">Velocity</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-[11px] tracking-normal text-[var(--ink)]">
            Amplitude Curve
            <select
              className={selectClass}
              value={midiAmplitudeCurve}
              disabled={controlsDisabled}
              onChange={(event) => onMidiAmplitudeCurveChange(event.target.value as MidiAmplitudeCurve)}
            >
              <option value="linear">Linear</option>
              <option value="db">dB</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-[11px] tracking-normal text-[var(--ink)]">
            CC Update Rate
            <select
              className={selectClass}
              value={midiCcUpdateRateHz}
              disabled={controlsDisabled}
              onChange={(event) => onMidiCcUpdateRateChange(Number(event.target.value) as MidiCcUpdateRateHz)}
            >
              <option value={50}>50 Hz</option>
              <option value={100}>100 Hz</option>
              <option value={200}>200 Hz</option>
              <option value={400}>400 Hz</option>
              <option value={800}>800 Hz</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-[11px] tracking-normal text-[var(--ink)]">
            BPM
            <input
              type="text"
              inputMode="numeric"
              className="rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[11px] text-[var(--ink)]"
              value={midiBpm}
              onChange={(event) => onMidiBpmChange(event.target.value)}
            />
          </label>
        </div>
      )}
    </aside>
  )
}

function formatSpeedLabel(speed: number): string {
  if (speed === 0.125) return '1/8x'
  if (speed === 0.25) return '1/4x'
  if (speed === 0.5) return '1/2x'
  return `${speed}x`
}
