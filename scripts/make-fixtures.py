"""Generate e2e test fixtures into web/e2e/fixtures/generated/ (gitignored).

    ml/.venv/bin/python scripts/make-fixtures.py

The 15 Mpx stress images are synthetic but non-trivial (gradients + texture)
so JPEG/PNG encode sizes and enhance timings are realistic.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from imageenh import colormath  # noqa: E402

OUT = ROOT / "web" / "e2e" / "fixtures" / "generated"


def synthetic(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    img = np.stack(
        [
            0.35 + 0.4 * np.sin(xx * 7 + yy * 3),
            0.45 + 0.3 * np.cos(xx * 5 - yy * 6),
            0.5 + 0.35 * np.sin((xx + yy) * 4),
        ],
        axis=-1,
    )
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return colormath.quantize_u8(np.clip(0.5 + (img - 0.5) * 0.9, 0, 1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    small = synthetic(640, 480, 1)
    dark_small = colormath.apply_u8(small, 0.6, 0.9, 0.8)
    Image.fromarray(dark_small).save(OUT / "small.jpg", quality=88)
    Image.fromarray(dark_small).save(OUT / "small.png")
    Image.fromarray(dark_small).save(OUT / "small.bmp")

    big = synthetic(5000, 3000, 2)  # exactly 15 Mpx
    dark_big = colormath.apply_u8(big, 0.65, 0.85, 0.85)
    Image.fromarray(dark_big).save(OUT / "big.jpg", quality=85)
    Image.fromarray(dark_big).save(OUT / "big.png")

    for f in sorted(OUT.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
