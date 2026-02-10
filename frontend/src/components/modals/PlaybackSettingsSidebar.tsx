type TimeStretchMode = 'native' | 'librosa'

export type PlaybackSettingsSidebarProps = {
  mixValue: number
  speedPresetIndex: number
  speedValue: number
  timeStretchMode: TimeStretchMode
  controlsDisabled: boolean
  onClose: () => void
  onMixChange: (value: number) => void
  onSpeedChange: (value: number) => void
  onTimeStretchModeChange: (mode: TimeStretchMode) => void
}

export function PlaybackSettingsSidebar({
  mixValue,
  speedPresetIndex,
  speedValue,
  timeStretchMode,
  controlsDisabled,
  onClose,
  onMixChange,
  onSpeedChange,
  onTimeStretchModeChange,
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
      {controlsDisabled ? (
        <div className="mb-3 rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--accent-warm)]">
          Some settings are unavailable during playback
        </div>
      ) : null}
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
    </aside>
  )
}

function formatSpeedLabel(speed: number): string {
  if (speed === 0.125) return '1/8x'
  if (speed === 0.25) return '1/4x'
  if (speed === 0.5) return '1/2x'
  return `${speed}x`
}
