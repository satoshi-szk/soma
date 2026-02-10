export {}

declare global {
  interface Window {
    pywebview?: {
      api?: {
        health: () => Promise<unknown>
        status: () => Promise<unknown>
        frontend_log: (level: string, message: string) => Promise<unknown>
        open_audio: () => Promise<unknown>
        open_audio_path: (payload: unknown) => Promise<unknown>
        open_audio_data: (payload: unknown) => Promise<unknown>
        new_project: () => Promise<unknown>
        open_project: () => Promise<unknown>
        save_project: () => Promise<unknown>
        save_project_as: () => Promise<unknown>
        update_settings: (payload: unknown) => Promise<unknown>
        trace_partial: (payload: unknown) => Promise<unknown>
        erase_partial: (payload: unknown) => Promise<unknown>
        update_partial: (payload: unknown) => Promise<unknown>
        merge_partials: (payload: unknown) => Promise<unknown>
        delete_partials: (payload: unknown) => Promise<unknown>
        toggle_mute: (payload: unknown) => Promise<unknown>
        hit_test: (payload: unknown) => Promise<unknown>
        select_in_box: (payload: unknown) => Promise<unknown>
        undo: () => Promise<unknown>
        redo: () => Promise<unknown>
        play: (payload: unknown) => Promise<unknown>
        start_harmonic_probe: (payload: unknown) => Promise<unknown>
        update_harmonic_probe: (payload: unknown) => Promise<unknown>
        stop_harmonic_probe: () => Promise<unknown>
        pause: () => Promise<unknown>
        stop: (payload?: unknown) => Promise<unknown>
        set_master_volume: (payload: unknown) => Promise<unknown>
        update_playback_mix: (payload: unknown) => Promise<unknown>
        list_midi_outputs: () => Promise<unknown>
        update_playback_settings: (payload: unknown) => Promise<unknown>
        playback_state: () => Promise<unknown>
        export_mpe: (payload: unknown) => Promise<unknown>
        export_multitrack_midi: (payload: unknown) => Promise<unknown>
        export_monophonic_midi: (payload: unknown) => Promise<unknown>
        export_audio: (payload: unknown) => Promise<unknown>
        request_viewport_preview: (payload: unknown) => Promise<unknown>
      }
    }
  }
}
