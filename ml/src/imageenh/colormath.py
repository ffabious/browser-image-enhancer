"""Canonical color math — the Python twin of web/src/lib/apply/enhance.ts.

This module implements docs/color-math.md exactly. Any change here must be
mirrored in the TS implementation and the golden fixtures regenerated.
"""

from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)
LN25 = math.log(2.5)

# Rec.601 luma coefficients.
LUMA_R, LUMA_G, LUMA_B = 0.299, 0.587, 0.114


def factors_from_output(o: np.ndarray) -> tuple[float, float, float]:
    """Map model output o ∈ [-1, 1]^3 to (brightness, contrast, saturation)."""
    o = np.asarray(o, dtype=np.float64)
    return (
        float(math.exp(o[0] * LN2)),
        float(math.exp(o[1] * LN2)),
        float(math.exp(o[2] * LN25)),
    )


def output_from_factors(beta: float, gamma: float, sigma: float) -> np.ndarray:
    """Inverse of factors_from_output (log-space parameterization)."""
    return np.array(
        [math.log(beta) / LN2, math.log(gamma) / LN2, math.log(sigma) / LN25],
        dtype=np.float32,
    )


def apply_float(rgb: np.ndarray, beta: float, gamma: float, sigma: float) -> np.ndarray:
    """Apply brightness -> contrast -> saturation to float RGB in [0, 1].

    No clamping is performed (single final clamp happens at quantization);
    used both by apply_u8 and (via TF re-implementation) the training loss.
    rgb: float32 array of shape (..., 3).
    """
    rgb = rgb.astype(np.float32, copy=False)
    x2 = (rgb * np.float32(beta) - np.float32(0.5)) * np.float32(gamma) + np.float32(0.5)
    luma = (
        np.float32(LUMA_R) * x2[..., 0:1]
        + np.float32(LUMA_G) * x2[..., 1:2]
        + np.float32(LUMA_B) * x2[..., 2:3]
    )
    return luma + (x2 - luma) * np.float32(sigma)


def quantize_u8(x: np.ndarray) -> np.ndarray:
    """Final clamp to [0,1] and round-half-up quantization to uint8.

    np.round is banker's rounding and must not be used (see docs/color-math.md).
    """
    return np.floor(np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def apply_u8(rgb: np.ndarray, beta: float, gamma: float, sigma: float) -> np.ndarray:
    """Apply enhancement to a uint8 RGB image (alpha, if any, is the caller's
    concern — this function takes shape (..., 3))."""
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 input, got {rgb.dtype}")
    x = rgb.astype(np.float32) / np.float32(255.0)
    return quantize_u8(apply_float(x, beta, gamma, sigma))


def apply_rgba_u8(rgba: np.ndarray, beta: float, gamma: float, sigma: float) -> np.ndarray:
    """Apply enhancement to a uint8 RGBA image, leaving alpha untouched."""
    out = rgba.copy()
    out[..., :3] = apply_u8(rgba[..., :3], beta, gamma, sigma)
    return out
