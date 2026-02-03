import type { AnalysisSettings, Partial, PartialPoint, SpectrogramPreview, ToolId } from '../../app/types'

export type WorkspaceProps = {
  preview: SpectrogramPreview | null
  viewportPreviews: SpectrogramPreview[] | null
  settings: AnalysisSettings
  partials: Partial[]
  selectedIds: string[]
  selectedInfo: Partial | null
  activeTool: ToolId
  analysisState: 'idle' | 'analyzing' | 'error'
  isSnapping: boolean
  zoomX: number
  zoomY: number
  pan: { x: number; y: number }
  playbackPosition: number
  canEditPlayhead: boolean
  onZoomXChange: (zoom: number, targetPan?: { x: number; y: number }) => void
  onPanChange: (pan: { x: number; y: number }) => void
  onStageSizeChange: (size: { width: number; height: number }) => void
  onTraceCommit: (trace: Array<[number, number]>) => Promise<boolean>
  onEraseCommit: (trace: Array<[number, number]>) => void
  onSelectBoxCommit: (selection: { time_start: number; time_end: number; freq_start: number; freq_end: number }) => void
  onHitTestCommit: (point: { time: number; freq: number }) => void
  onUpdatePartial: (id: string, points: PartialPoint[]) => void
  onConnectPick: (point: { time: number; freq: number }) => void
  onOpenAudio: () => void
  onOpenAudioPath: (path: string) => void
  onOpenAudioFile: (file: File) => void
  onPlayheadChange: (time: number) => void
  allowDrop: boolean
  onCursorMove: (cursor: { time: number; freq: number; amp: number | null }) => void
  onPartialMute: () => void
  onPartialDelete: () => void
  onZoomInY: () => void
  onZoomOutY: () => void
}

