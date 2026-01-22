export {}

declare global {
  interface Window {
    pywebview?: {
      api?: {
        health: () => Promise<{ status: string }>
        status: () => Promise<{
          status: 'ok'
          is_playing: boolean
          is_resynthesizing: boolean
          position: number
        }>
        analysis_status: () => Promise<{
          status: 'ok'
          state: 'idle' | 'processing' | 'ready' | 'error'
          preview?: {
            width: number
            height: number
            data: number[]
            freq_min: number
            freq_max: number
            duration_sec: number
          }
          message?: string
        }>
        frontend_log: (level: string, message: string) => Promise<{ status: 'ok' }>
        open_audio: () => Promise<any>
        new_project: () => Promise<any>
        open_project: () => Promise<any>
        save_project: () => Promise<any>
        save_project_as: () => Promise<any>
        update_settings: (payload: any) => Promise<any>
        trace_partial: (payload: any) => Promise<any>
        erase_partial: (payload: any) => Promise<any>
        update_partial: (payload: any) => Promise<any>
        merge_partials: (payload: any) => Promise<any>
        delete_partials: (payload: any) => Promise<any>
        toggle_mute: (payload: any) => Promise<any>
        hit_test: (payload: any) => Promise<any>
        select_in_box: (payload: any) => Promise<any>
        undo: () => Promise<any>
        redo: () => Promise<any>
        play: (payload: any) => Promise<any>
        pause: () => Promise<any>
        stop: () => Promise<any>
        playback_state: () => Promise<any>
        export_mpe: (payload: any) => Promise<any>
        export_audio: (payload: any) => Promise<any>
      }
    }
  }
}
