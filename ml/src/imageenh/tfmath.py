"""TensorFlow mirror of the canonical color math (docs/color-math.md), used by
the data pipeline (degradation) and the training loss (differentiable apply).

The uint8 quantization path that must match TS lives in colormath.py; here we
work in float and only quantize when faking an 8-bit input image.
"""

from __future__ import annotations

import math

import tensorflow as tf

LN2 = math.log(2.0)
LN25 = math.log(2.5)
LUMA = tf.constant([0.299, 0.587, 0.114])


def factors_from_output(o: tf.Tensor) -> tf.Tensor:
    """o: (batch, 3) in [-1, 1] → (batch, 3) multiplicative factors."""
    scale = tf.constant([LN2, LN2, LN25])
    return tf.exp(o * scale)


def output_from_factors(factors: tf.Tensor) -> tf.Tensor:
    scale = tf.constant([LN2, LN2, LN25])
    return tf.math.log(factors) / scale


def apply(images: tf.Tensor, factors: tf.Tensor) -> tf.Tensor:
    """Brightness → contrast → saturation, unclamped.

    images: (batch, h, w, 3) float in [0, 1]; factors: (batch, 3).
    """
    beta = factors[:, 0][:, None, None, None]
    gamma = factors[:, 1][:, None, None, None]
    sigma = factors[:, 2][:, None, None, None]
    x2 = (images * beta - 0.5) * gamma + 0.5
    luma = tf.reduce_sum(x2 * LUMA, axis=-1, keepdims=True)
    return luma + (x2 - luma) * sigma


def soft_clamp(x: tf.Tensor, leak: float = 0.02) -> tf.Tensor:
    """clip(x, 0, 1) with a small linear leak outside so gradients survive."""
    clipped = tf.clip_by_value(x, 0.0, 1.0)
    return clipped + leak * (x - clipped)


def quantize_like_u8(x: tf.Tensor) -> tf.Tensor:
    """Snap a degraded image to the 8-bit grid (inputs in the browser are
    8-bit). Not differentiable — only used on the no-gradient degrade path."""
    return tf.round(tf.clip_by_value(x, 0.0, 1.0) * 255.0) / 255.0
