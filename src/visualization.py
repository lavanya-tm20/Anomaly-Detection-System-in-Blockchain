from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_anomalies_scatter(
    X: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = "Anomalies (PCA 2D)",
) -> None:
    """
    Projects features to 2D using PCA and plots anomalies vs normals.
    """
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    normals = y_pred == 0
    anomalies = y_pred == 1

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[normals, 0], X2[normals, 1], s=12, c="#4CAF50", alpha=0.6, label="Normal")
    plt.scatter(X2[anomalies, 0], X2[anomalies, 1], s=18, c="#F44336", alpha=0.8, label="Anomaly")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

