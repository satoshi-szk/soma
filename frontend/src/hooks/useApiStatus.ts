import { useEffect, useState } from 'react'
import type { ApiStatus } from '../app/types'

export const useApiStatus = () => {
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')

  useEffect(() => {
    let alive = true

    const checkApi = async () => {
      const api = window.pywebview?.api
      if (!api?.health) {
        if (alive) setApiStatus('disconnected')
        return
      }
      try {
        const result = await api.health()
        if (alive) {
          setApiStatus(result?.status === 'ok' ? 'connected' : 'disconnected')
        }
      } catch {
        if (alive) setApiStatus('disconnected')
      }
    }

    void checkApi()
    const handleReady = () => {
      void checkApi()
    }
    window.addEventListener('pywebviewready', handleReady)
    return () => {
      alive = false
      window.removeEventListener('pywebviewready', handleReady)
    }
  }, [])

  return apiStatus
}
