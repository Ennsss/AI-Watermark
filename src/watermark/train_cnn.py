"""Training helper for the optional CNN-assisted extraction branch."""

from __future__ import annotations

import numpy as np

from watermark.models.cnn_decoder import build_cnn_decoder


def train_cnn_decoder(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    val_inputs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
):
    """Train the shallow CNN decoder on LH2/HL2 inputs and raw bit targets."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "CNN training requires TensorFlow. Install the optional ml dependencies."
        ) from exc

    output_bits = int(train_targets.shape[1])
    model = build_cnn_decoder(
        input_shape=tuple(train_inputs.shape[1:]),
        output_bits=output_bits,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_inputs is not None else "loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )
    ]

    validation_data = None
    if val_inputs is not None and val_targets is not None:
        validation_data = (val_inputs, val_targets)

    history = model.fit(
        train_inputs,
        train_targets,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
    return model, history

