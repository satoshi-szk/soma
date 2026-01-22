import type { AnalysisSettings, ToolId } from './types'

export const TOOL_KEYS: Record<ToolId, string> = {
  select: 'V',
  trace: 'P',
  erase: 'E',
  connect: 'C',
}

export const TOOL_LIST: { id: ToolId; label: string }[] = [
  { id: 'select', label: 'Select' },
  { id: 'trace', label: 'Trace' },
  { id: 'erase', label: 'Erase' },
  { id: 'connect', label: 'Connect' },
]

export const DEFAULT_SETTINGS: AnalysisSettings = {
  freq_min: 20,
  freq_max: 20000,
  bins_per_octave: 48,
  time_resolution_ms: 10,
  preview_freq_max: 12000,
  preview_bins_per_octave: 12,
  color_map: 'magma',
  brightness: 0,
  contrast: 1,
}

export const MENU_SECTIONS = [
  {
    label: 'Project',
    items: ['New Project', 'Open Project...', 'Open Audio...', 'Save Project', 'Save As...'],
  },
  { label: 'Analysis', items: ['Analysis Settings...', 'Plugin Manager...'] },
  { label: 'View', items: ['Zoom In', 'Zoom Out', 'Reset View'] },
  { label: 'System', items: ['About SOMA', 'Quit'] },
]
