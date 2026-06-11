"""Self-supervised dataset: well-exposed thumbnails + synthetic degradation.

Thumbnails are precomputed once (256×256 aspect-ignoring squash, like the
browser's preprocessing but slightly larger to allow crop jitter) and cached
as a .npy next to the image directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from . import tfmath

THUMB = 256
CROP = 224

# Degradation factor ranges (log-uniform), per docs/spec.md §5.
BETA_RANGE = (0.45, 2.0)
GAMMA_RANGE = (0.5, 1.8)
SIGMA_RANGE = (0.3, 1.8)
PER_PARAM_IDENTITY_P = 0.2
FULL_IDENTITY_P = 0.1
NOISE_STD_MAX = 0.01
JPEG_P = 0.5

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def precompute_thumbs(image_dir: Path, cache: Path | None = None) -> np.ndarray:
    """Squash every image in image_dir to THUMB×THUMB uint8; cache as .npy."""
    cache = cache or image_dir.with_suffix(".thumbs.npy")
    if cache.exists():
        return np.load(cache)
    files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"no images found in {image_dir}")
    thumbs = np.empty((len(files), THUMB, THUMB, 3), dtype=np.uint8)
    for i, path in enumerate(files):
        with Image.open(path) as im:
            thumbs[i] = np.asarray(
                im.convert("RGB").resize((THUMB, THUMB), Image.Resampling.BILINEAR)
            )
        if (i + 1) % 50 == 0:
            print(f"  thumbs: {i + 1}/{len(files)}")
    np.save(cache, thumbs)
    return thumbs


def _sample_factors() -> tf.Tensor:
    """One degradation factor triple, with identity dropout."""

    def one(lo: float, hi: float) -> tf.Tensor:
        f = tf.exp(tf.random.uniform([], np.log(lo), np.log(hi)))
        return tf.where(tf.random.uniform([]) < PER_PARAM_IDENTITY_P, 1.0, f)

    factors = tf.stack([one(*BETA_RANGE), one(*GAMMA_RANGE), one(*SIGMA_RANGE)])
    return tf.where(tf.random.uniform([]) < FULL_IDENTITY_P, tf.ones(3), factors)


def _degrade(original: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """original: (h, w, 3) float [0,1] → (degraded, original, target_o)."""
    factors = _sample_factors()
    degraded = tfmath.apply(original[None], factors[None])[0]

    # Sensor-ish noise, then snap to the 8-bit grid like a real input file.
    noise_std = tf.random.uniform([], 0.0, NOISE_STD_MAX)
    degraded = degraded + tf.random.normal(tf.shape(degraded), stddev=noise_std)
    degraded = tfmath.quantize_like_u8(degraded)

    # JPEG round-trip on half the samples (robustness to compression).
    if tf.random.uniform([]) < JPEG_P:
        q = tf.random.uniform([], 70, 96, dtype=tf.int32)
        degraded = tf.image.adjust_jpeg_quality(degraded, q)

    # Aux target: log-inverse of the degradation, clipped to the model range.
    target_o = tf.clip_by_value(-tfmath.output_from_factors(factors[None])[0], -1.0, 1.0)
    return degraded, original, target_o


def _augment(thumb: tf.Tensor) -> tf.Tensor:
    thumb = tf.image.random_flip_left_right(thumb)
    thumb = tf.image.random_crop(thumb, (CROP, CROP, 3))
    return tf.cast(thumb, tf.float32) / 255.0


def make_dataset(thumbs: np.ndarray, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(thumbs)
    if shuffle:
        ds = ds.shuffle(len(thumbs), reshuffle_each_iteration=True).repeat()
    ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_degrade, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
