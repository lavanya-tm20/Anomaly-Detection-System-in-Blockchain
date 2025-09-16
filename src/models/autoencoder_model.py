from __future__ import annotations
import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class AutoencoderDetector:
    """
    Simple fully-connected autoencoder for anomaly detection.
    - Train on normal data only (label=0) to reconstruct features.
    - Use reconstruction error threshold to flag anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (64, 32),
        learning_rate: float = 1e-3,
        random_state: int | None = 42,
    ) -> None:
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        if random_state is not None:
            tf.keras.utils.set_random_seed(random_state)
        self.model = self._build_model()

    def _build_model(self) -> keras.Model:
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        for h in self.hidden_dims:
            x = layers.Dense(h, activation="relu")(x)
        encoded = layers.Dense(self.encoding_dim, activation="relu")(x)
        x = layers.Dense(self.hidden_dims[-1], activation="relu")(encoded)
        for h in reversed(self.hidden_dims[:-1]):
            x = layers.Dense(h, activation="relu")(x)
        outputs = layers.Dense(self.input_dim, activation="linear")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mse")
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        epochs: int = 30,
        batch_size: int = 128,
        validation_split: float = 0.1,
        verbose: int = 0,
    ) -> None:
        if y is not None:
            X_train = X[y == 0]
        else:
            X_train = X
        self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=verbose,
        )

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X, verbose=0)
        errors = np.mean(np.square(X - preds), axis=1)
        return errors

    def predict(self, X: np.ndarray, threshold: float | None = None, contamination: float = 0.02) -> np.ndarray:
        errors = self.reconstruction_error(X)
        if threshold is None:
            # Set threshold by contamination quantile (higher error => anomaly)
            threshold = float(np.quantile(errors, 1.0 - contamination))
        return (errors >= threshold).astype(int)

