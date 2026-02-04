from __future__ import annotations

import numpy as np


def peak_normalize_buffer(buffer: np.ndarray, target_peak: float = 0.99) -> np.ndarray:
    if buffer.size == 0:
        return buffer.astype(np.float32)

    peak = float(np.max(np.abs(buffer)))
    if not np.isfinite(peak) or peak <= 0.0:
        return buffer.astype(np.float32)

    normalized = buffer.astype(np.float32) * (target_peak / peak)
    return np.asarray(np.clip(normalized, -1.0, 1.0), dtype=np.float32)
