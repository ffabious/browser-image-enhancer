"""Export golden fixtures that pin the Python/TS contract.

Usage:
    uv run python -m imageenh.export_golden colormath
    uv run python -m imageenh.export_golden nn        # requires a trained model

Fixtures are written to web/tests/golden/ and committed, so CI never needs
Python to run the TS test suite.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

from . import colormath

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = REPO_ROOT / "web" / "tests" / "golden"


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def export_colormath() -> Path:
    rng = np.random.default_rng(42)
    cases = []

    factor_sets: list[tuple[float, float, float]] = [
        (1.0, 1.0, 1.0),  # identity
        (0.5, 2.0, 2.5),  # range extremes
        (2.0, 0.5, 0.4),  # opposite extremes
    ]
    factor_sets += [
        tuple(np.exp(rng.uniform(-1, 1, 3) * [colormath.LN2, colormath.LN2, colormath.LN25]))
        for _ in range(5)
    ]

    for beta, gamma, sigma in factor_sets:
        rgba = rng.integers(0, 256, size=(32, 32, 4), dtype=np.uint8)
        cases.append(
            {
                "width": 32,
                "height": 32,
                "factors": [float(beta), float(gamma), float(sigma)],
                "input": _b64(rgba),
                "expected": _b64(colormath.apply_rgba_u8(rgba, beta, gamma, sigma)),
            }
        )

    # A ramp covering every 8-bit value in every channel position.
    ramp = np.zeros((16, 16, 4), dtype=np.uint8)
    vals = np.arange(256, dtype=np.uint8).reshape(16, 16)
    ramp[..., 0] = vals
    ramp[..., 1] = vals[::-1, :]
    ramp[..., 2] = vals[:, ::-1]
    ramp[..., 3] = 255
    for beta, gamma, sigma in [(1.3, 1.2, 1.5), (0.7, 0.9, 0.6)]:
        cases.append(
            {
                "width": 16,
                "height": 16,
                "factors": [beta, gamma, sigma],
                "input": _b64(ramp),
                "expected": _b64(colormath.apply_rgba_u8(ramp, beta, gamma, sigma)),
            }
        )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLDEN_DIR / "colormath.json"
    out.write_text(json.dumps({"tolerance": 1, "cases": cases}))
    return out


def export_nn() -> Path:
    """Per-layer golden vectors for the hand-rolled TS engine, generated from
    the exported weights. Requires the trained model at ml/runs/model.keras."""
    import keras  # deferred: heavy import, train extra

    model_path = REPO_ROOT / "ml" / "runs" / "model.keras"
    model = keras.saving.load_model(model_path)

    # The browser runs on fp16-rounded weights (export_weights.py); golden
    # outputs must be computed with the same rounding or they can never match.
    for w in model.weights:
        w.assign(w.numpy().astype(np.float16).astype(np.float32))

    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, size=(1, 224, 224, 3)).astype(np.float32)

    layers = [
        l
        for l in model.layers
        if l.weights or isinstance(l, keras.layers.GlobalAveragePooling2D)
    ]
    probe = keras.Model(model.inputs, [l.output for l in layers])
    outputs = probe(x)

    fixture = {
        "input": _b64(x),
        "layers": [
            {
                "name": layer.name,
                "shape": list(np.asarray(out).shape[1:]),
                "output": _b64(np.asarray(out).astype(np.float32)),
            }
            for layer, out in zip(layers, outputs)
        ],
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLDEN_DIR / "nn.json"
    out_path.write_text(json.dumps(fixture))
    return out_path


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "colormath"
    if which == "colormath":
        print(f"wrote {export_colormath()}")
    elif which == "nn":
        print(f"wrote {export_nn()}")
    else:
        raise SystemExit(f"unknown fixture set: {which}")


if __name__ == "__main__":
    main()
