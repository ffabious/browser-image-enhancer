"""Export the trained Keras model to the browser format:
web/public/model/weights.bin (little-endian fp16) + manifest.json.

    uv run --extra train python -m imageenh.export_weights [runs/model.keras]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import keras
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "web" / "public" / "model"


def export(model: keras.Model, out_dir: Path = OUT_DIR) -> None:
    layers: list[dict] = []
    blobs: list[np.ndarray] = []
    offset = 0

    def push(arr: np.ndarray) -> int:
        nonlocal offset
        blobs.append(arr.astype(np.float16).ravel())
        start = offset
        offset += arr.size
        return start

    gap_emitted = False
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            kernel, bias = (w.numpy() for w in layer.weights)
            kh, kw, in_c, out_c = kernel.shape
            assert kh == kw, "square kernels only"
            layers.append(
                {
                    "type": "conv",
                    "kernel": int(kh),
                    "stride": int(layer.strides[0]),
                    "inC": int(in_c),
                    "outC": int(out_c),
                    "activation": layer.activation.__name__ if layer.activation else "none",
                    "wOffset": push(kernel),
                    "bOffset": push(bias),
                }
            )
        elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
            layers.append({"type": "gap"})
            gap_emitted = True
        elif isinstance(layer, keras.layers.Dense):
            assert gap_emitted, "dense before gap is not supported by the TS engine"
            kernel, bias = (w.numpy() for w in layer.weights)
            in_dim, units = kernel.shape
            layers.append(
                {
                    "type": "dense",
                    "inDim": int(in_dim),
                    "units": int(units),
                    "activation": layer.activation.__name__ if layer.activation else "none",
                    "wOffset": push(kernel),
                    "bOffset": push(bias),
                }
            )

    manifest = {
        "version": 1,
        "inputSize": int(model.input_shape[1]),
        "totalElements": int(offset),
        "layers": layers,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    weights = np.concatenate(blobs)
    assert weights.dtype == np.float16
    (out_dir / "weights.bin").write_bytes(weights.tobytes())
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    kb = weights.nbytes / 1024
    print(f"wrote {out_dir / 'weights.bin'} ({kb:.0f} KiB, {offset} params)")
    print(f"wrote {out_dir / 'manifest.json'}")


def main() -> None:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "ml" / "runs" / "model.keras"
    export(keras.saving.load_model(model_path))


if __name__ == "__main__":
    main()
