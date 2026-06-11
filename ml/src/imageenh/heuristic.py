"""Python mirror of the heuristic baseline (web/src/lib/analyze/heuristic.ts),
used only by evaluate.py to compare the ML brain against the same baseline
the browser ships."""

from __future__ import annotations

import math

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _damp(factor: float, power: float) -> float:
    return math.exp(math.log(factor) * power)


def heuristic_factors(thumb_u8: np.ndarray) -> tuple[float, float, float]:
    """thumb_u8: (h, w, 3) uint8 → (brightness, contrast, saturation)."""
    x = thumb_u8.astype(np.float64) / 255.0
    luma = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    mean_luma = float(luma.mean())
    chroma = np.abs(x - luma[..., None]).mean(axis=-1)
    mean_chroma = float(chroma.mean())

    hist, _ = np.histogram(np.minimum((luma * 255 + 0.5).astype(int), 255), bins=256, range=(0, 256))
    cum = np.cumsum(hist)
    n = luma.size

    def percentile(q: float) -> float:
        idx = int(np.searchsorted(cum, q * n))
        return min(idx, 255) / 255.0

    p5, p95 = percentile(0.05), percentile(0.95)

    brightness = _clamp(_damp(_clamp(0.45 / max(mean_luma, 0.02), 0.5, 2), 0.7), 0.5, 2)
    spread = max((p95 - p5) * brightness, 0.05)
    contrast = _clamp(_damp(_clamp(0.8 / spread, 0.5, 2), 0.7), 0.6, 1.8)
    saturation = _clamp(_damp(_clamp(0.09 / max(mean_chroma, 0.005), 0.4, 2.5), 0.6), 0.6, 1.9)
    return brightness, contrast, saturation
