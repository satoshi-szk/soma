import { useEffect, useRef, useState } from 'react'
import { SelectionHud } from '../SelectionHud'
import { WorkspaceCanvas } from './internal/WorkspaceCanvas'
import { useDropAudio } from './internal/useDropAudio'
import { useWorkspaceController } from './internal/useWorkspaceController'
import type { WorkspaceProps } from './types'

export function Workspace(props: WorkspaceProps) {
  const { preview, selectedInfo, selectedIds, onOpenAudio, analysisState, allowDrop, isSnapping, onPartialMute, onPartialDelete, onZoomInY, onZoomOutY, onStageSizeChange } = props
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const { isDragActive, handleDragOver, handleDragEnter, handleDragLeave, handleDrop } = useDropAudio({
    containerRef,
    allowDrop,
    onOpenAudioPath: props.onOpenAudioPath,
    onOpenAudioFile: props.onOpenAudioFile,
  })
  useEffect(() => {
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setStageSize({ width, height })
        onStageSizeChange({ width, height })
      }
    })
    if (containerRef.current) observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [onStageSizeChange])
  const controller = useWorkspaceController(props, stageSize)
  const { tracePathD, committedTracePath, hudPosition, pxPerOctave } = controller

  return (
    <div
      ref={containerRef}
      className={`canvas-surface relative h-full min-h-[520px] rounded-none p-4 ${
        allowDrop && isDragActive ? 'ring-2 ring-[var(--accent)] ring-offset-2 ring-offset-[var(--canvas)]' : ''
      }`}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <WorkspaceCanvas controller={controller} />
      {tracePathD || committedTracePath ? (
        <svg
          className="pointer-events-none absolute inset-4"
          width={stageSize.width}
          height={stageSize.height}
          viewBox={`0 0 ${stageSize.width} ${stageSize.height}`}
        >
          {tracePathD ? <path d={tracePathD} fill="none" stroke="#7feeff" strokeWidth={3.75} /> : null}
          <path
            d={committedTracePath}
            fill="none"
            stroke={isSnapping ? '#7feeff' : 'rgba(245, 159, 139, 0.5)'}
            strokeWidth={isSnapping ? 3.75 : 1.25}
            className={isSnapping ? 'animate-pulse' : ''}
          />
        </svg>
      ) : null}
      {!preview ? (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center text-white/70">
          <div className="text-xs uppercase tracking-[0.3em]">No Audio Loaded</div>
          {allowDrop ? <div className="text-[10px] uppercase tracking-[0.24em] text-white/60">Drop a WAV file here</div> : null}
          <button
            className="rounded-md bg-white/90 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.2em] text-[var(--accent-strong)]"
            onClick={onOpenAudio}
          >
            Open Audio
          </button>
        </div>
      ) : null}
      {allowDrop && isDragActive ? (
        <div className="pointer-events-none absolute inset-3 rounded-md border border-dashed border-white/60" />
      ) : null}
      {analysisState === 'analyzing' ? (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 text-xs uppercase tracking-[0.24em] text-white">
          Analyzing...
        </div>
      ) : null}
      {selectedInfo ? (
        <SelectionHud
          selected={selectedInfo}
          canMute={selectedIds.length === 1}
          canDelete={selectedIds.length > 0}
          onMute={onPartialMute}
          onDelete={onPartialDelete}
          position={hudPosition}
        />
      ) : null}
      {preview ? (
        <div className="absolute right-6 top-1/2 -translate-y-1/2 flex flex-col gap-1">
          <button
            className="flex h-7 w-7 items-center justify-center rounded-sm bg-[rgba(12,18,30,0.85)] text-[var(--muted)] hover:bg-[rgba(20,28,45,0.95)] hover:text-white transition-colors"
            onClick={onZoomInY}
            title="Zoom in frequency"
          >
            <span className="text-sm font-bold">+</span>
          </button>
          <div className="flex h-5 items-center justify-center text-[9px] text-[var(--muted)] font-mono whitespace-nowrap">
            {pxPerOctave.toFixed(0)} px/oct
          </div>
          <button
            className="flex h-7 w-7 items-center justify-center rounded-sm bg-[rgba(12,18,30,0.85)] text-[var(--muted)] hover:bg-[rgba(20,28,45,0.95)] hover:text-white transition-colors"
            onClick={onZoomOutY}
            title="Zoom out frequency"
          >
            <span className="text-sm font-bold">−</span>
          </button>
        </div>
      ) : null}
    </div>
  )
}
