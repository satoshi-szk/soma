export {}

declare global {
  interface Window {
    pywebview?: {
      api?: {
        health: () => Promise<{ status: string }>
        open_audio: () => Promise<{
          status: 'ok' | 'cancelled' | 'error'
          message?: string
          audio?: {
            path: string
            name: string
            sample_rate: number
            duration_sec: number
            channels: number
            truncated: boolean
          }
          preview?: {
            width: number
            height: number
            data: number[]
          }
        }>
      }
    }
  }
}
