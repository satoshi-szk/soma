export type StatusBarProps = {
  statusLabel: string
  cursorLabel: string
  statusNote: string | null
  apiBadge: string
  spectrogramDim: number
  spectrogramDimEnabled: boolean
  onSpectrogramDimChange: (value: number) => void
}

export function StatusBar({
  statusLabel,
  cursorLabel,
  statusNote,
  apiBadge,
  spectrogramDim,
  spectrogramDimEnabled,
  onSpectrogramDimChange,
}: StatusBarProps) {
  return (
    <footer className="grid grid-cols-1 gap-2 px-1 py-1 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)] lg:grid-cols-4">
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2">
        Status: {statusLabel}
        {statusNote ? <span className="ml-2 font-mono text-[10px] normal-case tracking-normal">{statusNote}</span> : null}
      </div>
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 font-mono text-[10px]">
        {cursorLabel}
      </div>
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 font-mono text-[9px]">
        <span className="uppercase tracking-[0.18em] text-[var(--muted)]">{apiBadge}</span>
      </div>
      <div className="flex items-center rounded-sm border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 font-mono text-[9px]">
        <label className="flex w-full items-center gap-2 uppercase tracking-[0.16em]">
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
