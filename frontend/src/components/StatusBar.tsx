export type StatusBarProps = {
  statusLabel: string
  cursorLabel: string
  statusNote: string | null
  apiBadge: string
  spectrogramDim: number
  spectrogramDimEnabled: boolean
  onSpectrogramDimChange: (value: number) => void
  playbackError: string | null
  onOpenPlaybackSettings: () => void
}

export function StatusBar({
  statusLabel,
  cursorLabel,
  statusNote,
  apiBadge,
  spectrogramDim,
  spectrogramDimEnabled,
  onSpectrogramDimChange,
  playbackError,
  onOpenPlaybackSettings,
}: StatusBarProps) {
  return (
    <footer className="grid grid-cols-1 gap-2 px-1 py-1 text-[11px] tracking-normal text-[var(--ink)] lg:grid-cols-4">
      <div
        className={`flex items-center rounded-sm border px-3 py-2 ${
          playbackError ? 'border-red-400 bg-red-900/30 text-red-100' : 'border-[var(--panel-border)] bg-[var(--panel-strong)]'
        }`}
      >
        Status: {statusLabel}
        {playbackError ? (
          <span className="ml-2 font-mono text-[11px] normal-case tracking-normal">
            {playbackError}{' '}
            <button type="button" className="underline decoration-dotted underline-offset-2" onClick={onOpenPlaybackSettings}>
              Open MIDI Settings
            </button>
          </span>
        ) : statusNote ? (
          <span className="ml-2 font-mono text-[11px] normal-case tracking-normal">{statusNote}</span>
        ) : null}
      </div>
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 font-mono text-[11px] text-[var(--ink)]">
        {cursorLabel}
      </div>
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 font-mono text-[11px]">
        <span className="tracking-normal text-[var(--ink)]">{apiBadge}</span>
      </div>
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 font-mono text-[11px]">
        <label className="flex w-full items-center gap-2 tracking-normal text-[var(--ink)]">
          <span className="shrink-0">Spectrogram Dim {Math.round(spectrogramDim * 100)}%</span>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={Math.round(spectrogramDim * 100)}
            disabled={!spectrogramDimEnabled}
            onChange={(event) => onSpectrogramDimChange(Number(event.target.value) / 100)}
            className="w-full accent-[var(--accent-warm)] disabled:opacity-40"
            title="Adjust spectrogram dim level"
          />
        </label>
      </div>
    </footer>
  )
}
