import { useCallback, useEffect, useState } from 'react'
import type { DragEvent as ReactDragEvent } from 'react'
import type { RefObject } from 'react'

type Params = {
  containerRef: RefObject<HTMLDivElement | null>
  allowDrop: boolean
  onOpenAudioPath: (path: string) => void
  onOpenAudioFile: (file: File) => void
}

export function useDropAudio({ containerRef, allowDrop, onOpenAudioPath, onOpenAudioFile }: Params) {
  const [isDragActive, setIsDragActive] = useState(false)

  const resolveDroppedPath = useCallback((dataTransfer: DataTransfer | null) => {
    const directFile = dataTransfer?.files?.[0]
    if (directFile) {
      const candidate = (directFile as File & { path?: string }).path
      if (candidate) return candidate
    }

    const items = dataTransfer?.items
    if (items) {
      for (const item of Array.from(items)) {
        if (item.kind !== 'file') continue
        const file = item.getAsFile()
        const candidate = (file as (File & { path?: string }) | null)?.path
        if (candidate) return candidate
      }
    }

    const uriList = dataTransfer?.getData('text/uri-list')
    if (uriList) {
      const entry = uriList
        .split('\n')
        .map((line) => line.trim())
        .find((line) => line && !line.startsWith('#'))
      if (entry) {
        try {
          const url = new URL(entry)
          if (url.protocol === 'file:') {
            return decodeURIComponent(url.pathname)
          }
        } catch {
          // 無効な URI は無視する。
        }
      }
    }

    return null
  }, [])

  const isEventInsideContainer = useCallback((event: Event) => {
    const container = containerRef.current
    if (!container) return false
    const withPath = event as Event & { composedPath?: () => EventTarget[] }
    if (typeof withPath.composedPath === 'function') {
      return withPath.composedPath().includes(container)
    }
    return container.contains(event.target as Node)
  }, [containerRef])

  const handleDragOver = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      event.dataTransfer.dropEffect = 'copy'
    },
    [allowDrop]
  )

  const handleDragEnter = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      setIsDragActive(true)
    },
    [allowDrop]
  )

  const handleDragLeave = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      setIsDragActive(false)
    },
    [allowDrop]
  )

  const handleDrop = useCallback(
    (event: ReactDragEvent<HTMLDivElement>) => {
      if (!allowDrop) return
      event.preventDefault()
      setIsDragActive(false)
      const filePath = resolveDroppedPath(event.dataTransfer)
      if (filePath) {
        onOpenAudioPath(filePath)
        return
      }
      const fallbackFile = event.dataTransfer?.files?.[0] ?? null
      if (!fallbackFile) return
      onOpenAudioFile(fallbackFile)
    },
    [allowDrop, onOpenAudioPath, onOpenAudioFile, resolveDroppedPath]
  )

  useEffect(() => {
    if (!allowDrop) return

    const handleWindowDragOver = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'copy'
      }
    }

    const handleWindowDragEnter = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      setIsDragActive(true)
    }

    const handleWindowDragLeave = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      setIsDragActive(false)
    }

    const handleWindowDrop = (event: globalThis.DragEvent) => {
      if (!isEventInsideContainer(event)) return
      event.preventDefault()
      setIsDragActive(false)
      const filePath = resolveDroppedPath(event.dataTransfer)
      if (filePath) {
        onOpenAudioPath(filePath)
        return
      }
      const fallbackFile = event.dataTransfer?.files?.[0] ?? null
      if (!fallbackFile) return
      onOpenAudioFile(fallbackFile)
    }

    window.addEventListener('dragover', handleWindowDragOver, true)
    window.addEventListener('dragenter', handleWindowDragEnter, true)
    window.addEventListener('dragleave', handleWindowDragLeave, true)
    window.addEventListener('drop', handleWindowDrop, true)

    return () => {
      window.removeEventListener('dragover', handleWindowDragOver, true)
      window.removeEventListener('dragenter', handleWindowDragEnter, true)
      window.removeEventListener('dragleave', handleWindowDragLeave, true)
      window.removeEventListener('drop', handleWindowDrop, true)
    }
  }, [allowDrop, isEventInsideContainer, resolveDroppedPath, onOpenAudioPath, onOpenAudioFile])

  return {
    isDragActive,
    handleDragOver,
    handleDragEnter,
    handleDragLeave,
    handleDrop,
  }
}
