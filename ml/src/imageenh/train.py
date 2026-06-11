"""Training loop (custom tf.GradientTape loop — no Keras fit magic).

    uv run --extra train python -m imageenh.train --data-dir data/DIV2K_valid_HR \
        --epochs 40 --out runs/

Loss: image-space L1 between apply(degraded, predicted) and the original
(soft clamp for gradient flow) + aux parameter MSE decayed to 0 over the
first half of training.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import dataset, tfmath
from .model import build_model

AUX_WEIGHT = 0.1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    thumbs = dataset.precompute_thumbs(args.data_dir)
    n_val = min(max(1, int(len(thumbs) * args.val_frac)), len(thumbs) - 1)
    val_thumbs, train_thumbs = thumbs[:n_val], thumbs[n_val:]
    print(f"train images: {len(train_thumbs)}, val images: {n_val}")

    train_ds = dataset.make_dataset(train_thumbs, args.batch_size)
    model = build_model(args.width_mult)
    model.summary()

    total_steps = args.epochs * args.steps_per_epoch
    schedule = tf.keras.optimizers.schedules.CosineDecay(args.lr, total_steps)
    optimizer = tf.keras.optimizers.Adam(schedule)
    aux_decay_steps = total_steps / 2

    @tf.function
    def train_step(degraded, original, target_o, step):
        with tf.GradientTape() as tape:
            o = model(degraded, training=True)
            factors = tfmath.factors_from_output(o)
            restored = tfmath.soft_clamp(tfmath.apply(degraded, factors))
            img_l1 = tf.reduce_mean(tf.abs(restored - original))
            aux = tf.reduce_mean(tf.square(o - target_o))
            aux_w = AUX_WEIGHT * tf.maximum(0.0, 1.0 - tf.cast(step, tf.float32) / aux_decay_steps)
            loss = img_l1 + aux_w * aux
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, img_l1, aux

    def evaluate() -> tuple[float, float]:
        """Validation: image L1 after restoration + parameter MAE (log space)."""
        val_ds = dataset.make_dataset(val_thumbs, args.batch_size, shuffle=False)
        l1s, maes = [], []
        for degraded, original, target_o in val_ds.take(max(1, 2048 // args.batch_size)):
            o = model(degraded, training=False)
            restored = tf.clip_by_value(tfmath.apply(degraded, tfmath.factors_from_output(o)), 0, 1)
            l1s.append(float(tf.reduce_mean(tf.abs(restored - original))))
            maes.append(float(tf.reduce_mean(tf.abs(o - target_o))))
        return float(np.mean(l1s)), float(np.mean(maes))

    args.out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    step = 0
    it = iter(train_ds)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(args.steps_per_epoch):
            degraded, original, target_o = next(it)
            loss, img_l1, aux = train_step(degraded, original, target_o, tf.constant(step))
            losses.append([float(loss), float(img_l1), float(aux)])
            step += 1
        mean_loss, mean_l1, mean_aux = np.mean(losses, axis=0)
        val_l1, val_mae = evaluate()
        marker = ""
        if val_l1 < best_val:
            best_val = val_l1
            model.save(args.out / "model.keras")
            marker = "  ← saved"
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {mean_loss:.4f}  imgL1 {mean_l1:.4f}  "
            f"aux {mean_aux:.4f}  val-imgL1 {val_l1:.4f}  val-paramMAE {val_mae:.4f}  "
            f"({time.time() - t0:.0f}s){marker}"
        )

    print(f"best val image-L1: {best_val:.4f}; model saved to {args.out / 'model.keras'}")


if __name__ == "__main__":
    main()
