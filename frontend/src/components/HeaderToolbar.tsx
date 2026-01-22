import type { ToolId } from '../app/types'
import { MENU_SECTIONS, TOOL_KEYS, TOOL_LIST } from '../app/constants'

export type HeaderToolbarProps = {
  apiBadge: string
  menuOpen: boolean
  activeTool: ToolId
  isPlaying: boolean
  isLooping: boolean
  mixValue: number
  playbackTimeLabel: string
  playDisabled: boolean
  onMenuToggle: () => void
  onMenuAction: (label: string) => void
  onToolChange: (tool: ToolId) => void
  onStop: () => void
  onPlayToggle: () => void
  onLoopToggle: () => void
  onMixChange: (value: number) => void
  onExport: () => void
  menuRef: React.RefObject<HTMLDivElement | null>
}

export function HeaderToolbar({
  apiBadge,
  menuOpen,
  activeTool,
  isPlaying,
  isLooping,
  mixValue,
  playbackTimeLabel,
  playDisabled,
  onMenuToggle,
  onMenuAction,
  onToolChange,
  onStop,
  onPlayToggle,
  onLoopToggle,
  onMixChange,
  onExport,
  menuRef,
}: HeaderToolbarProps) {
  return (
    <header className="panel flex flex-col gap-4 rounded-none px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-center gap-4">
        <div className="relative" ref={menuRef}>
          <button
            className="rounded-none border border-[var(--panel-border)] bg-white px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]"
            onClick={onMenuToggle}
            aria-expanded={menuOpen}
          >
            Menu
          </button>
          {menuOpen ? (
            <div className="absolute left-0 top-[calc(100%+8px)] z-20 w-56 rounded-none border border-[var(--panel-border)] bg-white p-3 text-xs shadow-lg">
              {MENU_SECTIONS.map((section) => (
                <div key={section.label} className="border-b border-[var(--panel-border)] py-2 last:border-b-0">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">
                    {section.label}
                  </div>
                  <div className="mt-2 flex flex-col gap-1">
                    {section.items.map((item) => (
                      <button
                        key={item}
                        className="rounded-none px-2 py-1 text-left text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--ink)] hover:bg-[var(--panel-strong)]"
                        onClick={() => onMenuAction(item)}
                      >
                        {item}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : null}
        </div>
        <div>
          <div className="text-xl font-semibold tracking-[0.2em] text-[var(--ink)]">SOMA</div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-[var(--muted)]">Sonic Observation</div>
        </div>
        <span className="rounded-none border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
          {apiBadge}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2 rounded-none border border-[var(--panel-border)] bg-white px-3 py-2 text-sm font-semibold text-[var(--ink)]">
          <button
            onClick={onStop}
            className="rounded-none border border-transparent px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)] hover:border-[var(--panel-border)]"
          >
            Stop
          </button>
          <button
            onClick={onPlayToggle}
            className={`rounded-none px-4 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${
              playDisabled
                ? 'bg-[var(--panel-strong)] text-[var(--muted)]'
                : 'bg-[var(--accent)] text-white'
            }`}
            disabled={playDisabled}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={onLoopToggle}
            className={`rounded-none px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${
              isLooping
                ? 'bg-[var(--accent-warm)] text-white'
                : 'border border-[var(--panel-border)] text-[var(--muted)]'
            }`}
          >
            Loop
          </button>
        </div>
        <div className="flex flex-col">
          <span className="text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">
            Original {100 - mixValue}% / Resynth {mixValue}%
          </span>
          <input
            aria-label="Original and resynth mix"
            className="h-1 w-40 accent-[var(--accent)]"
            type="range"
            min={0}
            max={100}
            value={mixValue}
            onChange={(event) => onMixChange(Number(event.target.value))}
          />
        </div>
        <div className="rounded-none border border-[var(--panel-border)] bg-[var(--panel-strong)] px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]">
          {playbackTimeLabel}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="grid grid-cols-2 gap-2 rounded-none border border-[var(--panel-border)] bg-white p-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)] lg:grid-cols-4">
          {TOOL_LIST.map((tool) => (
            <button
              key={tool.id}
              onClick={() => onToolChange(tool.id)}
              className={`rounded-none px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] ${
                activeTool === tool.id
                  ? 'bg-[var(--accent)] text-white'
                  : 'text-[var(--muted)] hover:bg-[var(--panel-strong)]'
              }`}
            >
              {tool.label}
              <span className="ml-2 font-mono text-[9px] text-white/70">{TOOL_KEYS[tool.id]}</span>
            </button>
          ))}
        </div>
        <button
          className="rounded-none bg-[var(--accent-warm)] px-5 py-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-white"
          onClick={onExport}
        >
          Export
        </button>
      </div>
    </header>
  )
}
