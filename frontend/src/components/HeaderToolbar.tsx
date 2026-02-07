import type { ToolId } from '../app/types'
import { MENU_SECTIONS, TOOL_KEYS, TOOL_LIST } from '../app/constants'

export type HeaderToolbarProps = {
  menuOpen: boolean
  activeTool: ToolId
  isPlaying: boolean
  isProbePlaying: boolean
  mixValue: number
  speedValue: number
  playbackTimeLabel: string
  playDisabled: boolean
  onMenuToggle: () => void
  onMenuAction: (label: string) => void
  onToolChange: (tool: ToolId) => void
  onPlayStop: () => void
  onProbeToggle: () => void
  onRewind: () => void
  onMixChange: (value: number) => void
  onSpeedChange: (value: number) => void
  menuRef: React.RefObject<HTMLDivElement | null>
}

export function HeaderToolbar({
  menuOpen,
  activeTool,
  isPlaying,
  isProbePlaying,
  mixValue,
  speedValue,
  playbackTimeLabel,
  playDisabled,
  onMenuToggle,
  onMenuAction,
  onToolChange,
  onPlayStop,
  onProbeToggle,
  onRewind,
  onMixChange,
  onSpeedChange,
  menuRef,
}: HeaderToolbarProps) {
  return (
    <header className="panel flex flex-col gap-2 rounded-none px-2 py-2 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex items-center gap-2">
        <div className="relative" ref={menuRef}>
          <button
            className="flex h-9 w-10 flex-col items-center justify-center gap-1 rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] text-[var(--accent-strong)]"
            onClick={onMenuToggle}
            aria-expanded={menuOpen}
            aria-label="Menu"
          >
            <span className="sr-only">Menu</span>
            <span className="h-0.5 w-4 rounded-full bg-[var(--accent-strong)]" />
            <span className="h-0.5 w-4 rounded-full bg-[var(--accent-strong)]" />
            <span className="h-0.5 w-4 rounded-full bg-[var(--accent-strong)]" />
          </button>
          {menuOpen ? (
            <div className="absolute left-0 top-[calc(100%+8px)] z-20 w-56 rounded-md border border-[var(--panel-border)] bg-[var(--panel)] p-3 text-xs shadow-lg">
              {MENU_SECTIONS.map((section) => (
                <div key={section.label} className="border-b border-[var(--panel-border)] py-2 last:border-b-0">
                  <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--muted)]">
                    {section.label}
                  </div>
                  <div className="mt-2 flex flex-col gap-1">
                    {section.items.map((item) => (
                      <button
                        key={item}
                        className="rounded-md px-2 py-1 text-left text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--ink)] hover:bg-[var(--panel-strong)]"
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
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2 rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-3 py-2 text-sm font-semibold text-[var(--ink)]">
          <button
            onClick={onRewind}
            className={`flex h-8 w-8 items-center justify-center rounded-md ${
              isPlaying
                ? 'bg-[var(--panel-strong)] text-[var(--muted)]'
                : 'border border-[var(--panel-border)] text-[var(--muted)]'
            }`}
            disabled={isPlaying}
            aria-label="Rewind"
            title="Rewind"
          >
            <RewindIcon />
          </button>
          <button
            onClick={onPlayStop}
            className={`flex h-8 w-8 items-center justify-center rounded-md ${
              playDisabled && !isPlaying
                ? 'bg-[var(--panel-strong)] text-[var(--muted)]'
                : 'bg-[var(--accent)] text-white'
            }`}
            disabled={playDisabled && !isPlaying}
            aria-label={isPlaying ? 'Stop' : 'Play'}
            title={isPlaying ? 'Stop' : 'Play'}
          >
            {isPlaying ? <StopIcon /> : <PlayIcon />}
          </button>
          <button
            onClick={onProbeToggle}
            className={`rounded-md px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ${
              isProbePlaying
                ? 'bg-[var(--accent)] text-white'
                : isPlaying
                  ? 'bg-[var(--panel-strong)] text-[var(--muted)]'
                  : 'border border-[var(--panel-border)] text-[var(--muted)]'
            }`}
            disabled={isPlaying}
            aria-label="Harmonic Probe"
            title="Harmonic Probe (H)"
          >
            Probe
          </button>
        </div>
        <div className="flex flex-col">
          <span className="text-[11px] tracking-normal text-[var(--ink)]">
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
        <div className="flex flex-col">
          <span className="text-[11px] tracking-normal text-[var(--ink)]">Speed {(speedValue / 100).toFixed(2)}x</span>
          <input
            aria-label="Playback speed"
            className="h-1 w-40 accent-[var(--accent)]"
            type="range"
            min={12.5}
            max={800}
            step={0.5}
            value={speedValue}
            onChange={(event) => onSpeedChange(Number(event.target.value))}
          />
        </div>
        <div className="rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] px-4 py-2 text-[12px] font-semibold tracking-normal text-[var(--ink)]">
          {playbackTimeLabel}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="grid grid-cols-2 gap-2 rounded-md border border-[var(--panel-border)] bg-[var(--panel-strong)] p-2 text-[11px] font-semibold tracking-normal text-[var(--ink)] lg:grid-cols-4">
          {TOOL_LIST.map((tool) => (
            <button
              key={tool.id}
              onClick={() => onToolChange(tool.id)}
              className={`rounded-md px-3 py-2 text-[11px] font-semibold tracking-normal ${
                activeTool === tool.id
                  ? 'bg-[var(--accent)] text-white'
                  : 'text-[var(--muted)] hover:bg-[var(--panel-strong)]'
              }`}
              aria-label={tool.label}
              title={tool.label}
            >
              {tool.label}
              <span className={`ml-2 font-mono text-[9px] ${activeTool === tool.id ? 'text-white/70' : 'text-[var(--muted)]'}`}>
                {TOOL_KEYS[tool.id]}
              </span>
            </button>
          ))}
        </div>
      </div>
    </header>
  )
}

function RewindIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <rect x="1.1" y="2.1" width="1.1" height="9.8" rx="0.35" fill="currentColor" />
      <path d="M7.1 7 12.1 2.1v9.8L7.1 7Z" fill="currentColor" />
      <path d="M2.8 7 7.8 2.1v9.8L2.8 7Z" fill="currentColor" />
    </svg>
  )
}

function PlayIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <path d="M4 2.7 11 7 4 11.3V2.7Z" fill="currentColor" />
    </svg>
  )
}

function StopIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <rect x="3.1" y="3.1" width="7.8" height="7.8" rx="0.9" fill="currentColor" />
    </svg>
  )
}
