export type StatusBarProps = {
  statusLabel: string
  cursorLabel: string
  statusNote: string | null
  apiBadge: string
}

export function StatusBar({ statusLabel, cursorLabel, statusNote, apiBadge }: StatusBarProps) {
  return (
    <footer className="panel flex flex-col gap-2 rounded-none px-4 py-3 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between">
      <span>
        Status: {statusLabel}
        {statusNote ? <span className="ml-2 font-mono text-[10px] normal-case tracking-normal">{statusNote}</span> : null}
      </span>
      <span className="font-mono text-[10px]">{cursorLabel}</span>
      <span className="font-mono text-[9px]">
        Mem: 42MB <span className="ml-2 uppercase tracking-[0.18em] text-[var(--muted)]">{apiBadge}</span>
      </span>
    </footer>
  )
}
