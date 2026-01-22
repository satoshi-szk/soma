import type { Partial } from '../app/types'

export type SelectionHudProps = {
  selected: Partial | null
  canMute: boolean
  canDelete: boolean
  onMute: () => void
  onDelete: () => void
}

export function SelectionHud({ selected, canMute, canDelete, onMute, onDelete }: SelectionHudProps) {
  return (
    <div className="panel rounded-none px-4 py-3 text-sm text-[var(--muted)]">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
        <span>Selection HUD</span>
        <span className="font-mono text-[10px]">{selected ? selected.id : 'None'}</span>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          className="rounded-none border border-[var(--panel-border)] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-strong)]"
          onClick={onMute}
          disabled={!canMute}
        >
          Mute
        </button>
        <button
          className="rounded-none border border-[var(--panel-border)] px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-warm)]"
          onClick={onDelete}
          disabled={!canDelete}
        >
          Delete
        </button>
        <span className="text-[10px] uppercase tracking-[0.2em] text-[var(--muted)]">
          {selected
            ? `T: ${selected.points[0]?.time.toFixed(2)}s | F: ${selected.points[0]?.freq.toFixed(1)}Hz`
            : 'No selection'}
        </span>
      </div>
    </div>
  )
}
