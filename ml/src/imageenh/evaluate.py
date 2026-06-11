"""Quality evaluation over the reference pool (brief step 6).

Compares three pipelines on every (original, degraded) pair:
  none      — the degraded image as-is (lower bound)
  heuristic — the browser's baseline brain
  ml        — the trained CNN (with fp16-rounded weights, as deployed)

Metrics: PSNR / SSIM vs the original, parameter MAE in log space, and
identity safety (how much the 'perfect' images are altered). Writes a
Markdown report to docs/eval-report.md.

    uv run --extra train --extra eval python -m imageenh.evaluate
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from . import colormath
from .heuristic import heuristic_factors

REPO_ROOT = Path(__file__).resolve().parents[3]
THUMB = 224


def browser_thumb(img_u8: np.ndarray) -> np.ndarray:
    """224×224 aspect-ignoring squash — same preprocessing as the browser."""
    return np.asarray(
        Image.fromarray(img_u8).resize((THUMB, THUMB), Image.Resampling.BILINEAR)
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=2))


def load_ml_predictor(model_path: Path):
    import keras

    model = keras.saving.load_model(model_path)
    for w in model.weights:  # deployed weights are fp16-rounded
        w.assign(w.numpy().astype(np.float16).astype(np.float32))

    def predict(thumb_u8: np.ndarray) -> tuple[float, float, float]:
        x = thumb_u8.astype(np.float32)[None] / 255.0
        o = np.asarray(model(x))[0]
        return colormath.factors_from_output(o)

    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=REPO_ROOT / "reference-images")
    ap.add_argument("--model", type=Path, default=REPO_ROOT / "ml" / "runs" / "model.keras")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs" / "eval-report.md")
    args = ap.parse_args()

    manifest = json.loads((args.pool / "manifest.json").read_text())
    predict_ml = load_ml_predictor(args.model)

    brains = {
        "none": lambda thumb: (1.0, 1.0, 1.0),
        "heuristic": heuristic_factors,
        "ml": predict_ml,
    }

    rows = []
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for entry in manifest["entries"]:
        original = np.asarray(Image.open(args.pool / entry["original"]).convert("RGB"))
        degraded = np.asarray(Image.open(args.pool / entry["degraded"]).convert("RGB"))
        thumb = browser_thumb(degraded)
        true_o = colormath.output_from_factors(*entry["factors"])
        is_perfect = entry["category"] == "perfect"

        row = {"id": entry["id"], "category": entry["category"]}
        for name, brain in brains.items():
            factors = brain(thumb)
            restored = colormath.apply_u8(degraded, *factors)
            row[f"{name}_psnr"] = psnr(restored, original)
            row[f"{name}_ssim"] = ssim(restored, original)
            agg[name]["psnr"].append(row[f"{name}_psnr"])
            agg[name]["ssim"].append(row[f"{name}_ssim"])
            if name != "none":
                # Correction should invert the degradation: predicted o vs -true o.
                pred_o = colormath.output_from_factors(*factors)
                mae = float(np.mean(np.abs(pred_o - np.clip(-true_o, -1, 1))))
                agg[name]["param_mae"].append(mae)
                if is_perfect:
                    delta = float(
                        np.mean(np.abs(restored.astype(int) - original.astype(int)))
                    )
                    agg[name]["identity_delta"].append(delta)
        rows.append(row)
        print(
            f"{entry['id']:>5} {entry['category']:<18} "
            f"none {row['none_psnr']:5.1f} dB | heur {row['heuristic_psnr']:5.1f} dB | "
            f"ml {row['ml_psnr']:5.1f} dB"
        )

    lines = [
        "# Отчёт об оценке качества",
        "",
        "Метрики по эталонному пулу (`reference-images/`): восстановление",
        "деградированных изображений тремя вариантами «мозга». PSNR/SSIM считаются",
        "относительно исходного (неиспорченного) изображения.",
        "",
        "| ID | Категория | PSNR без обработки | PSNR эвристика | PSNR ML | SSIM ML |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['none_psnr']:.1f} | "
            f"{r['heuristic_psnr']:.1f} | {r['ml_psnr']:.1f} | {r['ml_ssim']:.3f} |"
        )
    lines += ["", "## Сводка", "", "| Метрика | без обработки | эвристика | ML |", "|---|---|---|---|"]
    for metric in ("psnr", "ssim"):
        vals = [np.mean(agg[n][metric]) for n in ("none", "heuristic", "ml")]
        fmt = "{:.2f}" if metric == "psnr" else "{:.4f}"
        lines.append(
            f"| средний {metric.upper()} | " + " | ".join(fmt.format(v) for v in vals) + " |"
        )
    lines.append(
        f"| MAE параметров (лог-пространство) | — | "
        f"{np.mean(agg['heuristic']['param_mae']):.3f} | {np.mean(agg['ml']['param_mae']):.3f} |"
    )
    lines.append(
        f"| Identity safety, средний |Δ| на «perfect», уровней | — | "
        f"{np.mean(agg['heuristic']['identity_delta']):.2f} | "
        f"{np.mean(agg['ml']['identity_delta']):.2f} |"
    )
    lines.append("")
    args.report.write_text("\n".join(lines))
    print(f"\nreport written to {args.report}")


if __name__ == "__main__":
    main()
