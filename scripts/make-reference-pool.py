"""Build the reference image pool (brief step 3) from DIV2K validation images.

Each pool entry is an (original, degraded) pair: the original is the quality
target, the degraded image is what the system receives. Degradations use the
canonical color math with documented factors, so parameter recovery can be
measured exactly. Images are downscaled to ≤1280 px to keep the repo small.

Run after scripts/fetch-dataset.sh:
    ml/.venv/bin/python scripts/make-reference-pool.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from imageenh import colormath  # noqa: E402

SRC = ROOT / "ml" / "data" / "DIV2K_valid_HR"
OUT = ROOT / "reference-images"
MAX_SIDE = 1280

# (DIV2K id, category, β_d, γ_d, σ_d)
POOL = [
    ("0801", "underexposed", 0.55, 1.0, 0.95),
    ("0802", "overexposed", 1.7, 1.0, 1.0),
    ("0803", "low-contrast", 1.0, 0.55, 1.0),
    ("0804", "hazy", 1.15, 0.6, 0.85),
    ("0805", "desaturated", 1.0, 1.0, 0.45),
    ("0806", "oversaturated", 1.0, 1.0, 1.7),
    ("0807", "night", 0.45, 0.85, 0.8),
    ("0808", "mixed-dark-flat", 0.7, 0.75, 0.85),
    ("0812", "mixed-dark-dull", 0.6, 0.9, 0.6),
    ("0813", "mixed-bright-flat", 1.4, 0.7, 1.0),
    ("0809", "perfect", 1.0, 1.0, 1.0),
    ("0810", "perfect", 1.0, 1.0, 1.0),
    ("0811", "perfect", 1.0, 1.0, 1.0),
]


def load_resized(div2k_id: str) -> np.ndarray:
    with Image.open(SRC / f"{div2k_id}.png") as im:
        im = im.convert("RGB")
        scale = MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize(
                (round(im.width * scale), round(im.height * scale)),
                Image.Resampling.BILINEAR,
            )
        return np.asarray(im)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    entries = []
    for div2k_id, category, beta, gamma, sigma in POOL:
        original = load_resized(div2k_id)
        degraded = colormath.apply_u8(original, beta, gamma, sigma)
        orig_name = f"{div2k_id}-original.jpg"
        deg_name = f"{div2k_id}-{category}.jpg"
        Image.fromarray(original).save(OUT / orig_name, quality=95)
        Image.fromarray(degraded).save(OUT / deg_name, quality=90)
        entries.append(
            {
                "id": div2k_id,
                "category": category,
                "original": orig_name,
                "degraded": deg_name,
                "factors": [beta, gamma, sigma],
            }
        )
        print(f"{div2k_id} {category}")
    (OUT / "manifest.json").write_text(json.dumps({"entries": entries}, indent=2))
    print(f"wrote {len(entries)} pairs to {OUT}")


if __name__ == "__main__":
    main()
