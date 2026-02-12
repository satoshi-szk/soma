from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from soma.midi_player import MidiPlayer
from soma.models import AnalysisSettings, AudioInfo, SourceInfo, SpectrogramPreview
from soma.partial_store import PartialStore
from soma.synth import AudioPlayer, Synthesizer


class ProjectSession:
    def __init__(self, event_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.audio_info: AudioInfo | None = None
        self.audio_data: np.ndarray | None = None
        self.settings = AnalysisSettings()
        self.preview: SpectrogramPreview | None = None
        self._amp_reference: float | None = None
        self._stft_amp_reference: float | None = None
        self._snap_amp_reference: float | None = None
        self.preview_state = "idle"
        self.preview_error: str | None = None
        self.store = PartialStore()
        self.project_path: Path | None = None
        self.source_info: SourceInfo | None = None
        self.synth = Synthesizer(sample_rate=44100, duration_sec=0.0)
        self.player = AudioPlayer()
        self.midi_player = MidiPlayer()
        self._lock = threading.Lock()
        self._is_resynthesizing = False
        self._logger = logging.getLogger(__name__)
        self._event_sink = event_sink

        self._active_snap_request_id: str | None = None
        self._active_snap_trace: list[tuple[float, float]] | None = None
        self._queued_snaps: list[tuple[str, list[tuple[float, float]]]] = []

        self._playback_mode: str | None = None
        self._playback_output_mode = "audio"
        self._playback_speed_ratio = 1.0
        self._is_preparing_playback = False
        self._pending_playback_position_sec = 0.0
        self._playback_prepare_request_id = 0
        self._playback_content_revision = 0
        self._playback_cache_key: tuple[int, float, float, str] | None = None
        self._playback_cache_buffer: np.ndarray | None = None
        self._stretched_original_cache_key: tuple[int, float, str] | None = None
        self._stretched_original_cache_buffer: np.ndarray | None = None
        self._time_scaled_resynth_cache_key: tuple[int, float, int] | None = None
        self._time_scaled_resynth_cache_buffer: np.ndarray | None = None

        self._master_volume = 1.0
        self.player.set_master_volume(self._master_volume)
        self._last_mix_ratio = 0.55
        self._last_speed_ratio = 1.0
        self._last_time_stretch_mode = "librosa"
        self._midi_mode = "mpe"
        self._midi_output_name = ""
        self._midi_pitch_bend_range = 48
        self._midi_amplitude_mapping = "cc74"
        self._midi_amplitude_curve = "linear"
        self._midi_cc_update_rate_hz = 400
        self._midi_bpm = 120.0
        self._spectrogram_renderer: Any = None

    def emit(self, payload: dict[str, Any]) -> None:
        if self._event_sink is None:
            return
        try:
            self._event_sink(payload)
        except Exception:  # pragma: no cover
            self._logger.exception("event_sink failed")

    def reset_for_new_project(self) -> None:
        self._close_spectrogram_renderer()
        self.audio_info = None
        self.audio_data = None
        self.settings = AnalysisSettings()
        self.preview = None
        self._amp_reference = None
        self._stft_amp_reference = None
        self._snap_amp_reference = None
        self.preview_state = "idle"
        self.preview_error = None
        self.store = PartialStore()
        self.project_path = None
        self.source_info = None
        self.synth.reset(sample_rate=44100, duration_sec=0.0)
        self._active_snap_request_id = None
        self._active_snap_trace = None
        self._queued_snaps = []
        self._playback_mode = None
        self._playback_output_mode = "audio"
        self._playback_speed_ratio = 1.0
        self._is_preparing_playback = False
        self._pending_playback_position_sec = 0.0
        self._playback_prepare_request_id += 1
        self._last_mix_ratio = 0.55
        self._last_speed_ratio = 1.0
        self._last_time_stretch_mode = "librosa"
        self._midi_mode = "mpe"
        self._midi_output_name = ""
        self._midi_pitch_bend_range = 48
        self._midi_amplitude_mapping = "cc74"
        self._midi_amplitude_curve = "linear"
        self._midi_cc_update_rate_hz = 400
        self._midi_bpm = 120.0
        self._master_volume = 1.0
        self.player.set_master_volume(self._master_volume)

    def apply_loaded_audio(
        self,
        *,
        info: AudioInfo,
        audio: np.ndarray,
        source_info: SourceInfo,
        initial_mix_buffer: np.ndarray,
    ) -> None:
        self._close_spectrogram_renderer()
        self.audio_info = info
        self.audio_data = audio
        self.source_info = source_info
        self.preview = None
        self._amp_reference = None
        self._stft_amp_reference = None
        self._snap_amp_reference = None
        self.preview_state = "idle"
        self.preview_error = None
        self.synth.reset(sample_rate=info.sample_rate, duration_sec=info.duration_sec)
        self.player.load(initial_mix_buffer, info.sample_rate)
        self.midi_player.stop()
        self._playback_mode = None

    def _close_spectrogram_renderer(self) -> None:
        renderer = self._spectrogram_renderer
        if renderer is None:
            return
        try:
            renderer.close()
        except Exception:
            self._logger.exception("failed to close spectrogram renderer")
        self._spectrogram_renderer = None
