"""Shallow baseline CNN decoder for transform-domain extraction."""

from __future__ import annotations


def build_cnn_decoder(
    input_shape: tuple[int, int, int] = (128, 128, 2),
    output_bits: int = 128,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.30,
):
    """Build the paper-aligned baseline CNN decoder.

    TensorFlow is imported lazily so the classical pipeline can run without ML
    dependencies. Install the optional ``ml`` extra to train or run this model.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "CNN extraction requires TensorFlow. Install the optional ml "
            "dependencies before training or running the CNN decoder."
        ) from exc

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(output_bits, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model

