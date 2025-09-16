from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    Wrapper for scikit-learn Isolation Forest anomaly detector.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        random_state: int | None = 42,
        max_samples: str | int = "auto",
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            max_samples=max_samples,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # scikit returns 1 for inliers, -1 for outliers -> convert to 0/1
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # Lower scores are more anomalous in Isolation Forest
        return -self.model.score_samples(X)

