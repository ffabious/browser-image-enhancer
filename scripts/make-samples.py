"""Generate the demo sample images (synthetic 'photos' degraded with the
canonical color math) into web/public/samples/.

Run: ml/.venv/bin/python scripts/make-samples.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from imageenh import colormath  # noqa: E402

OUT = ROOT / "web" / "public" / "samples"
W, H = 1280, 853


def base_image(seed: int) -> np.ndarray:
    """A plausible synthetic landscape: sky gradient, sun, hills, noise."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]

    img = np.zeros((H, W, 3), dtype=np.float32)
    # Sky: blue → warm near horizon.
    horizon = 0.55
    sky = np.clip(1 - y / horizon, 0, 1)
    img[..., 0] = 0.45 + 0.35 * (1 - sky)
    img[..., 1] = 0.55 + 0.25 * (1 - sky) * 0.6
    img[..., 2] = 0.95 - 0.45 * (1 - sky)

    # Sun glow.
    sun = np.exp(-(((x - 0.72) * 2.2) ** 2 + ((y - 0.30) * 3.2) ** 2) * 18)
    img += sun[..., None] * np.array([0.9, 0.75, 0.4], dtype=np.float32)

    # Rolling hills (two layers).
    for amp, base, color in [
        (0.06, 0.62, (0.18, 0.42, 0.22)),
        (0.05, 0.78, (0.10, 0.28, 0.13)),
    ]:
        ridge = base + amp * np.sin(x * 9 + rng.uniform(0, 6)) + amp * 0.5 * np.sin(x * 23)
        mask = (y > ridge).astype(np.float32)
        shade = 1 - 0.25 * np.sin(x * 40)  # texture
        for c in range(3):
            img[..., c] = img[..., c] * (1 - mask) + color[c] * shade * mask

    img += rng.normal(0, 0.008, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def save_degraded(name: str, img: np.ndarray, beta: float, gamma: float, sigma: float) -> None:
    u8 = colormath.quantize_u8(img)
    degraded = colormath.apply_u8(u8, beta, gamma, sigma)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(degraded).save(OUT / name, quality=88)
    print(f"wrote {OUT / name}")


def main() -> None:
    save_degraded("dark.jpg", base_image(1), 0.55, 0.95, 0.9)
    save_degraded("hazy.jpg", base_image(2), 1.05, 0.55, 0.85)
    save_degraded("dull.jpg", base_image(3), 0.95, 0.9, 0.45)


if __name__ == "__main__":
    main()
