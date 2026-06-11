"""The TF training math must agree with the canonical NumPy implementation
(which in turn is pinned to the TS runtime by golden fixtures)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from imageenh import colormath, tfmath  # noqa: E402


def test_apply_matches_colormath():
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 1, size=(2, 16, 16, 3)).astype(np.float32)
    factors = np.array([[1.4, 0.7, 1.8], [0.6, 1.5, 0.5]], dtype=np.float32)

    got = tfmath.apply(tf.constant(img), tf.constant(factors)).numpy()
    for i in range(2):
        want = colormath.apply_float(img[i], *factors[i])
        np.testing.assert_allclose(got[i], want, atol=1e-5)


def test_factor_mapping_matches():
    o = np.array([[0.3, -0.7, 0.9]], dtype=np.float32)
    got = tfmath.factors_from_output(tf.constant(o)).numpy()[0]
    want = colormath.factors_from_output(o[0])
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_output_from_factors_roundtrip():
    f = np.array([[1.3, 0.8, 2.0]], dtype=np.float32)
    o = tfmath.output_from_factors(tf.constant(f))
    back = tfmath.factors_from_output(o).numpy()
    np.testing.assert_allclose(back, f, rtol=1e-5)


def test_soft_clamp_passes_through_in_range():
    x = tf.constant([[0.0, 0.5, 1.0, 1.5, -0.5]])
    y = tfmath.soft_clamp(x).numpy()[0]
    np.testing.assert_allclose(y[:3], [0.0, 0.5, 1.0], atol=1e-7)
    assert y[3] == pytest.approx(1.0 + 0.02 * 0.5)
    assert y[4] == pytest.approx(-0.02 * 0.5)
