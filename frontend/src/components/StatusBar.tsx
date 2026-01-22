export type StatusBarProps = {
  statusLabel: string
  statusNote: string | null
}

export function StatusBar({ statusLabel, statusNote }: StatusBarProps) {
  return (
    <footer className="panel flex flex-col gap-2 rounded-none px-4 py-3 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between">
      <span>Status: {statusLabel}</span>
      <span className="font-mono text-[10px]">{statusNote ? statusNote : 'T: 12.3s | F: 440Hz (A4) | A: -6dB'}</span>
      <span className="font-mono text-[10px]">Mem: 42MB</span>
    </footer>
  )
}
