from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from soma.models import Partial, PartialPoint


class PartialStore:
    def __init__(self, chunk_size_sec: float = 0.1) -> None:
        self.partials: dict[str, Partial] = {}
        self.chunk_size_sec = chunk_size_sec
        self.index: dict[int, set[str]] = defaultdict(set)

    def add(self, partial: Partial) -> None:
        self.partials[partial.id] = partial
        self._index_partial(partial)

    def update(self, partial: Partial) -> None:
        self.remove(partial.id)
        self.add(partial)

    def remove(self, partial_id: str) -> None:
        if partial_id not in self.partials:
            return
        partial = self.partials.pop(partial_id)
        self._remove_from_index(partial)

    def get(self, partial_id: str) -> Partial | None:
        return self.partials.get(partial_id)

    def all(self) -> list[Partial]:
        return list(self.partials.values())

    def hit_test(
        self,
        time_sec: float,
        freq_hz: float,
        freq_min: float,
        freq_max: float,
        tolerance: float = 0.05,
    ) -> str | None:
        chunk = self._chunk_id(time_sec)
        candidates = set(self.index.get(chunk, set()))
        candidates.update(self.index.get(chunk - 1, set()))
        candidates.update(self.index.get(chunk + 1, set()))
        if not candidates:
            return None

        best_id = None
        best_distance = float("inf")
        for partial_id in candidates:
            partial = self.partials.get(partial_id)
            if not partial or partial.is_muted:
                continue
            distance = _distance_to_partial(time_sec, freq_hz, partial.points, freq_min, freq_max)
            if distance < best_distance:
                best_distance = distance
                best_id = partial_id

        return best_id if best_distance <= tolerance else None

    def select_in_box(
        self,
        time_start: float,
        time_end: float,
        freq_start: float,
        freq_end: float,
    ) -> list[str]:
        start_chunk = self._chunk_id(min(time_start, time_end))
        end_chunk = self._chunk_id(max(time_start, time_end))
        candidate_ids: set[str] = set()
        for chunk in range(start_chunk, end_chunk + 1):
            candidate_ids.update(self.index.get(chunk, set()))
        if not candidate_ids:
            return []

        t_min = min(time_start, time_end)
        t_max = max(time_start, time_end)
        f_min = min(freq_start, freq_end)
        f_max = max(freq_start, freq_end)
        selected: list[str] = []
        for partial_id in candidate_ids:
            partial = self.partials.get(partial_id)
            if not partial:
                continue
            for point in partial.points:
                if t_min <= point.time <= t_max and f_min <= point.freq <= f_max:
                    selected.append(partial_id)
                    break
        return selected

    def rebuild(self) -> None:
        self.index.clear()
        for partial in self.partials.values():
            self._index_partial(partial)

    def _index_partial(self, partial: Partial) -> None:
        for chunk_id in self._chunk_ids_for_points(partial.points):
            self.index[chunk_id].add(partial.id)

    def _remove_from_index(self, partial: Partial) -> None:
        for chunk_id in self._chunk_ids_for_points(partial.points):
            bucket = self.index.get(chunk_id)
            if not bucket:
                continue
            bucket.discard(partial.id)
            if not bucket:
                self.index.pop(chunk_id, None)

    def _chunk_ids_for_points(self, points: Iterable[PartialPoint]) -> set[int]:
        ids = set()
        for point in points:
            ids.add(self._chunk_id(point.time))
        return ids

    def _chunk_id(self, time_sec: float) -> int:
        return int(time_sec / self.chunk_size_sec)


def _distance_to_partial(
    time_sec: float,
    freq_hz: float,
    points: list[PartialPoint],
    freq_min: float,
    freq_max: float,
) -> float:
    if not points:
        return float("inf")

    times = np.array([point.time for point in points], dtype=np.float32)
    freqs = np.array([point.freq for point in points], dtype=np.float32)
    log_min = np.log10(max(freq_min, 1.0))
    log_max = np.log10(max(freq_max, freq_min + 1.0))
    log_freqs = (np.log10(np.clip(freqs, freq_min, freq_max)) - log_min) / max(1e-6, log_max - log_min)
    target_time = time_sec
    target_freq = (np.log10(np.clip(freq_hz, freq_min, freq_max)) - log_min) / max(1e-6, log_max - log_min)

    if times.size == 1:
        return float(np.hypot(times[0] - target_time, log_freqs[0] - target_freq))

    min_distance = float("inf")
    for idx in range(len(points) - 1):
        t0, t1 = times[idx], times[idx + 1]
        f0, f1 = log_freqs[idx], log_freqs[idx + 1]
        dt = t1 - t0
        df = f1 - f0
        if abs(dt) < 1e-6 and abs(df) < 1e-6:
            distance = np.hypot(target_time - t0, target_freq - f0)
        else:
            u = ((target_time - t0) * dt + (target_freq - f0) * df) / (dt * dt + df * df)
            u = float(np.clip(u, 0.0, 1.0))
            proj_time = t0 + u * dt
            proj_freq = f0 + u * df
            distance = np.hypot(target_time - proj_time, target_freq - proj_freq)
        min_distance = min(min_distance, float(distance))

    return min_distance
