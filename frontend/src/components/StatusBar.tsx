export type StatusBarProps = {
  statusLabel: string
}

export function StatusBar({ statusLabel }: StatusBarProps) {
  return (
    <footer className="panel flex flex-col gap-2 rounded-2xl px-4 py-3 text-[10px] uppercase tracking-[0.22em] text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between">
      <span>Status: {statusLabel}</span>
      <span className="font-mono text-[10px]">T: 12.3s | F: 440Hz (A4) | A: -6dB</span>
      <span className="font-mono text-[10px]">Mem: 42MB</span>
    </footer>
  )
}
