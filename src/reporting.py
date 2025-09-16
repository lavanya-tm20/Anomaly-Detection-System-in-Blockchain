from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def generate_text_report(
    df_raw: pd.DataFrame,
    y_pred: np.ndarray,
    metrics_iso: Dict[str, float],
    metrics_ae: Dict[str, float],
    output_path: str,
    top_k: int = 20,
) -> None:
    """
    Write a simple text report with metrics and top-K anomalies by model.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Anomaly Detection Report\n")
        f.write("========================\n\n")
        f.write("Metrics\n")
        f.write("-------\n")
        f.write(f"Isolation Forest: precision={metrics_iso['precision']:.4f}, recall={metrics_iso['recall']:.4f}, f1={metrics_iso['f1']:.4f}\n")
        f.write(f"Autoencoder     : precision={metrics_ae['precision']:.4f}, recall={metrics_ae['recall']:.4f}, f1={metrics_ae['f1']:.4f}\n\n")

        # Top anomalies according to combined prediction (1s). This is illustrative; real
        # ranking should use model-specific anomaly scores.
        f.write(f"Top {top_k} predicted anomalies (raw rows)\n")
        f.write("----------------------------------------\n")
        anomalous_idx = np.where(y_pred == 1)[0][:top_k]
        if len(anomalous_idx) == 0:
            f.write("No anomalies were predicted.\n")
        else:
            subset = df_raw.iloc[anomalous_idx][["sender", "receiver", "amount", "timestamp", "gas_fee"]]
            for i, (_, row) in enumerate(subset.iterrows(), start=1):
                f.write(
                    f"{i:02d}. sender={row['sender']} receiver={row['receiver']} amount={row['amount']:.6f} gas_fee={row['gas_fee']:.6f} timestamp={row['timestamp']}\n"
                )

