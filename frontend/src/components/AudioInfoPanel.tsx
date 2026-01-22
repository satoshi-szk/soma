import type { AudioInfo } from '../app/types'
import { formatDuration } from '../app/utils'

export type AudioInfoPanelProps = {
  audioInfo: AudioInfo | null
  analysisError: string | null
  statusNote: string | null
}

export function AudioInfoPanel({ audioInfo, analysisError, statusNote }: AudioInfoPanelProps) {
  return (
    <div className="panel rounded-none px-4 py-3 text-sm text-[var(--muted)]">
      <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--muted)]">Audio Info</div>
      <div className="mt-3 flex flex-col gap-2 text-[11px]">
        <span className="font-semibold text-[var(--ink)]">{audioInfo ? audioInfo.name : 'No file'}</span>
        <span className="font-mono text-[10px] text-[var(--muted)]">
          {audioInfo
            ? `${audioInfo.sample_rate} Hz | ${audioInfo.channels} ch | ${formatDuration(audioInfo.duration_sec)}`
            : 'Load a WAV file to analyze.'}
        </span>
        {audioInfo?.truncated ? (
          <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent-warm)]">
            Preview uses first 30 seconds.
          </span>
        ) : null}
        {analysisError ? (
          <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--accent-warm)]">
            {analysisError}
          </span>
        ) : null}
        {statusNote ? (
          <span className="text-[10px] uppercase tracking-[0.18em] text-[var(--muted)]">{statusNote}</span>
        ) : null}
      </div>
    </div>
  )
}
