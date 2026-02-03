import { useEffect, useCallback } from 'react'
import type { ToolId } from '../app/types'

type Options = {
  onToolChange: (tool: ToolId) => void
  onUndo: () => void
  onRedo: () => void
  onPlayToggle: () => void
  onSave: () => void | Promise<void>
}

export function useKeyboardShortcuts({ onToolChange, onUndo, onRedo, onPlayToggle, onSave }: Options) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target?.isContentEditable) return
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      const key = event.key.toLowerCase()
      if (key === 'v') onToolChange('select')
      if (key === 'p') onToolChange('trace')
      if (key === 'e') onToolChange('erase')
      if (key === 'c') onToolChange('connect')
      if (event.code === 'Space') {
        event.preventDefault()
        onPlayToggle()
      }
      if (key === 'z' && (event.metaKey || event.ctrlKey) && !event.shiftKey) {
        event.preventDefault()
        onUndo()
      }
      if (key === 'z' && event.shiftKey && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        onRedo()
      }
      if (key === 's' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        void onSave()
      }
    },
    [onToolChange, onUndo, onRedo, onPlayToggle, onSave]
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}
