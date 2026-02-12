import { useState, useEffect, useLayoutEffect, useRef, useCallback, useMemo } from 'react'
import { ensurePreviewData } from '../app/previewData'
import { isPywebviewApiAvailable, pywebviewApi } from '../app/pywebviewApi'
import {
  ZOOM_X_MAX_PX_PER_SEC,
  ZOOM_X_MIN_PX_PER_SEC,
  ZOOM_X_STEP_RATIO,
  ZOOM_Y_MIN,
  ZOOM_Y_MAX,
  RULER_HEIGHT,
  AUTOMATION_LANE_HEIGHT,
} from '../app/constants'
import type { AnalysisSettings, SpectrogramPreview } from '../app/types'

type ViewportParams = {
  timeStart: number
  timeEnd: number
  freqMin: number
  freqMax: number
}

type ViewportCacheEntry = {
  preview: SpectrogramPreview
  quality: 'low' | 'high'
  receivedAt: number
  sourceDuration: number
  sourceFreqMin: number
  sourceFreqMax: number
}

type ReportError = (context: string, message: string) => void

const buildViewportParamsKey = (params: ViewportParams, settings: AnalysisSettings, pixelHeight: number): string =>
  [
    params.timeStart.toFixed(6),
    params.timeEnd.toFixed(6),
    params.freqMin.toFixed(3),
    params.freqMax.toFixed(3),
    settings.gain.toFixed(3),
    settings.min_db.toFixed(3),
    settings.max_db.toFixed(3),
    settings.gamma.toFixed(3),
    pixelHeight.toString(),
  ].join('|')

export function useViewport(preview: SpectrogramPreview | null, settings: AnalysisSettings, reportError: ReportError) {
  const [zoomXState, setZoomXState] = useState<number | null>(null)
  const [zoomY, setZoomY] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [stageSize, setStageSize] = useState({ width: 900, height: 420 })
  const [viewportCache, setViewportCache] = useState<ViewportCacheEntry[]>([])
  const interactionRef = useRef(false)
  const interactionTimerRef = useRef<number | null>(null)
  const requestedTileKeysRef = useRef<Set<string>>(new Set())
  const pendingParamsRef = useRef<ViewportParams[] | null>(null)
  const lastPreviewId = useRef<string | null>(null)

  const baseZoomX = useMemo(() => {
    if (!preview) return 1
    return stageSize.width / Math.max(preview.duration_sec, 1e-6)
  }, [preview, stageSize])

  const zoomX = zoomXState ?? baseZoomX

  // スペクトログラム領域の高さ（ルーラーとオートメーションレーンを除く）
  const spectrogramAreaHeight = useMemo(
    () => Math.max(1, stageSize.height - AUTOMATION_LANE_HEIGHT - RULER_HEIGHT),
    [stageSize.height]
  )

  // Y 軸の基準スケール（preview 高さをスペクトログラム領域へ対応付ける）
  const baseScaleY = useMemo(() => {
    if (!preview) return 1
    return spectrogramAreaHeight / preview.height
  }, [preview, spectrogramAreaHeight])

  // コンテンツがビューポート範囲内に収まるよう pan を制限する
  const clampPan = useCallback(
    (
      newPan: { x: number; y: number },
      currentZoomX: number,
      currentZoomY: number
    ): { x: number; y: number } => {
      if (!preview) return newPan

      const contentWidth = preview.duration_sec * currentZoomX
      const contentHeight = preview.height * baseScaleY * currentZoomY

      // コンテンツがビューポートより小さい場合は原点に固定する（余白を作らない）
      const clampedX =
        contentWidth <= stageSize.width
          ? 0
          : Math.min(0, Math.max(-(contentWidth - stageSize.width), newPan.x))

      const clampedY =
        contentHeight <= spectrogramAreaHeight
          ? 0
          : Math.min(0, Math.max(-(contentHeight - spectrogramAreaHeight), newPan.y))

      return { x: clampedX, y: clampedY }
    },
    [preview, baseScaleY, stageSize.width, spectrogramAreaHeight]
  )

  // 現在の preview ソースを表す安定した ID を生成する
  const currentPreviewId = useMemo(() => {
    if (!preview) return null
    return `${preview.duration_sec}-${preview.freq_min}-${preview.freq_max}`
  }, [preview])

  // preview ソースが変わったらスロットル状態をクリアする
  useLayoutEffect(() => {
    if (currentPreviewId !== lastPreviewId.current) {
      lastPreviewId.current = currentPreviewId
      requestedTileKeysRef.current = new Set()
      const clearTimer = window.setTimeout(() => {
        setViewportCache([])
      }, 0)
      return () => {
        window.clearTimeout(clearTimer)
      }
    }
  }, [currentPreviewId])

  const clampZoomX = useCallback(
    (value: number) => {
      const minZoomX = Math.max(ZOOM_X_MIN_PX_PER_SEC, baseZoomX)
      return Math.min(ZOOM_X_MAX_PX_PER_SEC, Math.max(minZoomX, value))
    },
    [baseZoomX]
  )

  const clampZoomY = useCallback(
    (value: number) => Math.min(ZOOM_Y_MAX, Math.max(ZOOM_Y_MIN, value)),
    []
  )

  // 常にクランプを適用する setPan ラッパー
  const setPanClamped = useCallback(
    (value: { x: number; y: number } | ((prev: { x: number; y: number }) => { x: number; y: number })) => {
      setPan((prev) => {
        const next = typeof value === 'function' ? value(prev) : value
        return clampPan(next, zoomX, zoomY)
      })
    },
    [clampPan, zoomX, zoomY]
  )

  const setZoomXClamped = useCallback(
    (value: number | ((prev: number) => number), targetPan?: { x: number; y: number }) => {
      setZoomXState((prev) => {
        const current = prev ?? baseZoomX
        const next = typeof value === 'function' ? value(current) : value
        const clamped = clampZoomX(next)
        // targetPan があればそれを使い、なければ現在の pan を再クランプする
        setPan((prevPan) => clampPan(targetPan ?? prevPan, clamped, zoomY))
        return clamped
      })
    },
    [baseZoomX, clampZoomX, clampPan, zoomY]
  )

  const viewportPreviews = useMemo(() => {
    if (!preview || (zoomX <= baseZoomX && zoomY <= 1)) return null
    const currentDuration = preview.duration_sec
    const currentFreqMin = preview.freq_min
    const currentFreqMax = preview.freq_max

    // 現在の preview ソースに一致するキャッシュだけを残す
    const validEntries = viewportCache.filter(
      (entry) =>
        entry.sourceDuration === currentDuration &&
        entry.sourceFreqMin === currentFreqMin &&
        entry.sourceFreqMax === currentFreqMax
    )

    // 古いエントリが最初に見つかった時だけログを出す
    // （毎レンダーでフィルタされるため、ログスパムを避ける）

    return validEntries
      .slice()
      .sort((a, b) => a.receivedAt - b.receivedAt)
      .map((entry) => entry.preview)
  }, [preview, zoomX, zoomY, viewportCache, baseZoomX])

  const upsertViewportPreview = useCallback(
    (resolved: SpectrogramPreview, quality: 'low' | 'high') => {
      if (!preview) return
      const entry: ViewportCacheEntry = {
        preview: resolved,
        quality,
        receivedAt: Date.now(),
        sourceDuration: preview.duration_sec,
        sourceFreqMin: preview.freq_min,
        sourceFreqMax: preview.freq_max,
      }
      setViewportCache((prev) => {
        const hasHigh = prev.some(
          (item) =>
            item.quality === 'high' &&
            Math.abs(item.preview.time_start - resolved.time_start) < 0.001 &&
            Math.abs(item.preview.time_end - resolved.time_end) < 0.001 &&
            Math.abs(item.preview.freq_min - resolved.freq_min) < 0.1 &&
            Math.abs(item.preview.freq_max - resolved.freq_max) < 0.1
        )
        if (quality === 'low' && hasHigh) {
          return prev
        }
        const filtered =
          quality === 'high'
            ? prev.filter(
                (item) =>
                  !(
                    Math.abs(item.preview.time_start - resolved.time_start) < 0.001 &&
                    Math.abs(item.preview.time_end - resolved.time_end) < 0.001 &&
                    Math.abs(item.preview.freq_min - resolved.freq_min) < 0.1 &&
                    Math.abs(item.preview.freq_max - resolved.freq_max) < 0.1
                  )
              )
            : prev
        const nextItems = [...filtered, entry]
        const limit = 8
        if (nextItems.length <= limit) return nextItems
        return nextItems.slice(nextItems.length - limit)
      })
    },
    [preview]
  )

  const sendViewportRequest = useCallback(
    async (paramsList: ViewportParams[]) => {
      const api = isPywebviewApiAvailable() ? pywebviewApi : null
      if (!api) return

      const requestTasks: Promise<void>[] = []
      const tileHeight = Math.round(spectrogramAreaHeight)
      for (const params of paramsList) {
        const key = buildViewportParamsKey(params, settings, tileHeight)
        if (requestedTileKeysRef.current.has(key)) continue
        requestedTileKeysRef.current.add(key)
        requestTasks.push(
          (async () => {
            try {
              const result = await api.request_spectrogram_tile({
                time_start: params.timeStart,
                time_end: params.timeEnd,
                freq_min: params.freqMin,
                freq_max: params.freqMax,
                width: 1024,
                height: tileHeight,
                gain: settings.gain,
                min_db: settings.min_db,
                max_db: settings.max_db,
                gamma: settings.gamma,
              })

              if (result.status === 'error') {
                requestedTileKeysRef.current.delete(key)
                reportError('Viewport', result.message ?? 'Failed to request viewport preview')
                return
              }
              const resolved = await ensurePreviewData(result.preview)
              if (!resolved.image_path) {
                requestedTileKeysRef.current.delete(key)
                return
              }
              upsertViewportPreview(resolved, result.quality === 'high' ? 'high' : 'low')
            } catch {
              requestedTileKeysRef.current.delete(key)
            }
          })()
        )
      }
      if (requestTasks.length > 0) {
        await Promise.allSettled(requestTasks)
      }
    },
    [reportError, spectrogramAreaHeight, settings, upsertViewportPreview]
  )

  // zoom/pan 変更時に viewport preview を要求する（操作停止後）
  useEffect(() => {
    if (!preview || (zoomX <= baseZoomX && zoomY <= 1)) {
      requestedTileKeysRef.current = new Set()
      return
    }

    const duration = preview.duration_sec
    const freqMin = preview.freq_min
    const freqMax = preview.freq_max

    // 可視ビューポートを計算する
    const visibleTimeStart = Math.max(0, -pan.x / zoomX)
    const visibleTimeEnd = Math.min(duration, visibleTimeStart + stageSize.width / zoomX)
    const logMin = Math.log(freqMin)
    const logMax = Math.log(freqMax)
    const visibleFreqMax = Math.exp(
      logMax - Math.max(0, -pan.y / (spectrogramAreaHeight * zoomY)) * (logMax - logMin)
    )
    const visibleFreqMin = Math.exp(
      logMax -
        Math.min(1, (spectrogramAreaHeight - pan.y) / (spectrogramAreaHeight * zoomY)) * (logMax - logMin)
    )

    const clampedFreqMin = Math.max(freqMin, visibleFreqMin)
    const clampedFreqMax = Math.min(freqMax, visibleFreqMax)
    const viewportDuration = Math.max(1e-6, stageSize.width / zoomX)
    const overscanStart = Math.max(0, visibleTimeStart - viewportDuration)
    const overscanEnd = Math.min(duration, visibleTimeEnd + viewportDuration)
    const tileDuration = Math.max(1e-6, 1024 / zoomX)
    const startTileIndex = Math.floor(overscanStart / tileDuration)
    const endTileIndex = Math.floor(Math.max(overscanStart, overscanEnd - 1e-9) / tileDuration)
    const tiles: ViewportParams[] = []
    for (let tileIndex = startTileIndex; tileIndex <= endTileIndex; tileIndex += 1) {
      const tileStart = Math.max(0, tileIndex * tileDuration)
      const tileEnd = Math.min(duration, tileStart + tileDuration)
      if (tileEnd - tileStart < 1e-6) continue
      tiles.push({
        timeStart: tileStart,
        timeEnd: tileEnd,
        freqMin: clampedFreqMin,
        freqMax: clampedFreqMax,
      })
    }
    pendingParamsRef.current = tiles

    interactionRef.current = true
    if (interactionTimerRef.current) {
      window.clearTimeout(interactionTimerRef.current)
    }
    interactionTimerRef.current = window.setTimeout(() => {
      interactionRef.current = false
      const params = pendingParamsRef.current
      if (params && params.length > 0) {
        void sendViewportRequest(params)
      }
    }, 200)

    return () => {
      if (interactionTimerRef.current) {
        window.clearTimeout(interactionTimerRef.current)
      }
    }
  }, [zoomX, zoomY, pan, stageSize.width, spectrogramAreaHeight, preview, baseZoomX, sendViewportRequest])

  const zoomInX = useCallback(() => {
    setZoomXClamped((value) => value * ZOOM_X_STEP_RATIO)
  }, [setZoomXClamped])

  const zoomOutX = useCallback(() => {
    setZoomXClamped((value) => value / ZOOM_X_STEP_RATIO)
  }, [setZoomXClamped])

  const zoomInY = useCallback(() => {
    if (!preview) return
    const oldZoom = zoomY
    const newZoom = clampZoomY(oldZoom + 1.0)
    if (newZoom === oldZoom) return

    // ビューポート中央の周波数を維持する
    const centerY = spectrogramAreaHeight / 2
    const oldContentY = (centerY - pan.y) / (baseScaleY * oldZoom)
    const newPanY = centerY - oldContentY * baseScaleY * newZoom

    setZoomY(newZoom)
    setPan((prev) => clampPan({ x: prev.x, y: newPanY }, zoomX, newZoom))
  }, [preview, zoomY, zoomX, pan.y, baseScaleY, spectrogramAreaHeight, clampZoomY, clampPan])

  const zoomOutY = useCallback(() => {
    if (!preview) return
    const oldZoom = zoomY
    const newZoom = clampZoomY(oldZoom - 1.0)
    if (newZoom === oldZoom) return

    // ビューポート中央の周波数を維持する
    const centerY = spectrogramAreaHeight / 2
    const oldContentY = (centerY - pan.y) / (baseScaleY * oldZoom)
    const newPanY = centerY - oldContentY * baseScaleY * newZoom

    setZoomY(newZoom)
    setPan((prev) => clampPan({ x: prev.x, y: newPanY }, zoomX, newZoom))
  }, [preview, zoomY, zoomX, pan.y, baseScaleY, spectrogramAreaHeight, clampZoomY, clampPan])

  const resetView = useCallback(() => {
    setZoomXState(null)
    setZoomY(ZOOM_Y_MIN)
    setPan({ x: 0, y: 0 })
    setViewportCache([])
  }, [])

  return {
    zoomX,
    zoomY,
    pan,
    stageSize,
    viewportPreviews,
    setZoomX: setZoomXClamped,
    setPan: setPanClamped,
    setStageSize,
    zoomInX,
    zoomOutX,
    zoomInY,
    zoomOutY,
    resetView,
  }
}
