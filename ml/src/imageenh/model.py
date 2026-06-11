"""CNN architecture (see docs/spec.md §5): 5× strided conv → GAP → FC."""

from __future__ import annotations

import keras

WIDTHS = (16, 32, 64, 96, 128)
INPUT_SIZE = 224


def build_model(width_mult: float = 1.0, input_size: int = INPUT_SIZE) -> keras.Model:
    inputs = keras.Input(shape=(input_size, input_size, 3), name="thumb")
    x = inputs
    for i, w in enumerate(WIDTHS):
        x = keras.layers.Conv2D(
            max(8, int(w * width_mult)),
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"conv{i + 1}",
        )(x)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(64, activation="relu", name="fc1")(x)
    outputs = keras.layers.Dense(3, activation="tanh", name="fc2")(x)
    return keras.Model(inputs, outputs, name="enhancer")
