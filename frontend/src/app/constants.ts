import type { AnalysisSettings, ToolId } from "./types";

export const TOOL_KEYS: Record<ToolId, string> = {
  select: "V",
  trace: "P",
  erase: "E",
  connect: "C",
};

export const TOOL_LIST: { id: ToolId; label: string }[] = [
  { id: "select", label: "Select" },
  { id: "trace", label: "Trace" },
  { id: "erase", label: "Erase" },
  { id: "connect", label: "Connect" },
];

export const DEFAULT_SETTINGS: AnalysisSettings = {
  spectrogram: {
    freq_min: 20,
    freq_max: 20000,
    preview_freq_max: 12000,
    multires_blend_octaves: 1.0,
    gain: 1,
    min_db: -80,
    max_db: 0,
    gamma: 1,
  },
  snap: {
    freq_min: 20,
    freq_max: 20000,
    bins_per_octave: 96,
    time_resolution_ms: 10,
    wavelet_bandwidth: 8.0,
    wavelet_center_freq: 1.5,
  },
};

export const ZOOM_X_MIN_PX_PER_SEC = 0.05;
export const ZOOM_X_MAX_PX_PER_SEC = 10000;
export const ZOOM_X_STEP_RATIO = 2.0;
export const ZOOM_Y_MIN = 1;
export const ZOOM_Y_MAX = 10;

// UI レイアウト定数
export const RULER_HEIGHT = 28;
export const AUTOMATION_LANE_HEIGHT = 120;

export const MENU_SECTIONS = [
  {
    label: "Project",
    items: [
      "New Project",
      "Open Project...",
      "Save Project",
      "Save As...",
      "Export...",
    ],
  },
  { label: "Analysis", items: ["Analysis Settings...", "Plugin Manager..."] },
  { label: "View", items: ["Zoom In", "Zoom Out", "Reset View"] },
  { label: "System", items: ["About SOMA", "Quit"] },
];
