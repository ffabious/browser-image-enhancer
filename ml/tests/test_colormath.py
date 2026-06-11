import numpy as np
import pytest

from imageenh import colormath


@pytest.fixture
def rgba():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64, 4), dtype=np.uint8)


def test_identity_is_exact(rgba):
    out = colormath.apply_rgba_u8(rgba, 1.0, 1.0, 1.0)
    np.testing.assert_array_equal(out, rgba)


def test_alpha_untouched(rgba):
    out = colormath.apply_rgba_u8(rgba, 1.7, 0.6, 2.0)
    np.testing.assert_array_equal(out[..., 3], rgba[..., 3])


def test_brightness_monotonic():
    gray = np.full((8, 8, 3), 128, dtype=np.uint8)
    brighter = colormath.apply_u8(gray, 1.5, 1.0, 1.0)
    darker = colormath.apply_u8(gray, 0.7, 1.0, 1.0)
    assert brighter.mean() > 128 > darker.mean()


def test_contrast_pivot_fixed():
    # Pixels at exactly 0.5 are unchanged by any contrast factor.
    mid = np.full((4, 4, 3), 128, dtype=np.uint8)  # 128/255 ≈ 0.502
    for gamma in (0.5, 1.0, 2.0):
        out = colormath.apply_u8(mid, 1.0, gamma, 1.0)
        assert np.abs(out.astype(int) - 128).max() <= 1


def test_zero_saturation_is_grayscale(rgba):
    out = colormath.apply_u8(rgba[..., :3], 1.0, 1.0, 1e-6)
    assert np.abs(out[..., 0].astype(int) - out[..., 1].astype(int)).max() <= 1
    assert np.abs(out[..., 1].astype(int) - out[..., 2].astype(int)).max() <= 1


def test_output_clamped(rgba):
    out = colormath.apply_u8(rgba[..., :3], 2.0, 2.0, 2.5)
    assert out.dtype == np.uint8


def test_factor_roundtrip():
    o = np.array([0.3, -0.7, 0.9], dtype=np.float32)
    beta, gamma, sigma = colormath.factors_from_output(o)
    np.testing.assert_allclose(
        colormath.output_from_factors(beta, gamma, sigma), o, atol=1e-6
    )
    assert 0.5 <= beta <= 2.0 and 0.5 <= gamma <= 2.0 and 0.4 <= sigma <= 2.5


def test_degrade_then_inverse_params_recovers_unclipped():
    # For mid-range pixels (no clipping), applying inverse factors in reverse
    # is approximately identity for brightness-only changes.
    rng = np.random.default_rng(1)
    img = rng.integers(100, 156, size=(16, 16, 3), dtype=np.uint8)
    degraded = colormath.apply_u8(img, 0.8, 1.0, 1.0)
    restored = colormath.apply_u8(degraded, 1.25, 1.0, 1.0)
    assert np.abs(restored.astype(int) - img.astype(int)).max() <= 2
