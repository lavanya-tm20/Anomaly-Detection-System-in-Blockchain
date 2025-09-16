import os
import argparse
import numpy as np
import pandas as pd

from src.data import simulate_blockchain_transactions
from src.preprocessing import clean_transactions, build_feature_matrix, engineer_features
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.evaluation import evaluate_predictions
from src.visualization import plot_anomalies_scatter
from src.reporting import generate_text_report


def ensure_dirs() -> None:
    os.makedirs("outputs", exist_ok=True)


def run_workflow(
    num_tx: int,
    anomaly_ratio: float,
    random_state: int,
    iso_contamination: float,
    ae_contamination: float,
) -> None:
    # 1) Data
    df_raw, y_true = simulate_blockchain_transactions(
        num_transactions=num_tx,
        anomaly_ratio=anomaly_ratio,
        random_state=random_state,
    )

    # 2) Preprocessing
    df_clean = clean_transactions(df_raw)

    # Align labels with cleaned dataframe
    keep_mask = (df_raw["amount"] > 0) & (df_raw["gas_fee"] > 0)
    kept_indices = np.where(keep_mask.values)[0]
    y_true_clean = y_true[kept_indices]

    X, artifacts = build_feature_matrix(df_clean)

    # 3) Models
    iso = IsolationForestDetector(contamination=iso_contamination, random_state=random_state)
    iso.fit(X)
    y_pred_iso = iso.predict(X)

    ae = AutoencoderDetector(input_dim=X.shape[1])
    ae.fit(X, y=y_true_clean)
    y_pred_ae = ae.predict(X, contamination=ae_contamination)

    # 4) Evaluation
    metrics_iso = evaluate_predictions(y_true_clean, y_pred_iso)
    metrics_ae = evaluate_predictions(y_true_clean, y_pred_ae)

    # Combine predictions for reporting/visual (union)
    y_pred_union = ((y_pred_iso == 1) | (y_pred_ae == 1)).astype(int)

    # 5) Output
    ensure_dirs()
    plot_path = os.path.join("outputs", "anomalies_scatter.png")
    report_path = os.path.join("outputs", "anomaly_report.txt")
    csv_path = os.path.join("outputs", "transactions_with_labels.csv")

    plot_anomalies_scatter(X, y_pred_union, plot_path)

    # Print metrics to console
    print("Isolation Forest Metrics:", metrics_iso)
    print("Autoencoder Metrics:", metrics_ae)

    # Generate text report
    generate_text_report(df_clean, y_pred_union, metrics_iso, metrics_ae, report_path)
    print(f"Saved scatter plot to: {plot_path}")
    print(f"Saved report to     : {report_path}")

    # Save CSV with raw + engineered features + labels
    features_df = engineer_features(df_clean)
    combined = pd.concat([df_clean.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    combined["label_union"] = y_pred_union
    combined["label_isoforest"] = y_pred_iso
    combined["label_autoencoder"] = y_pred_ae
    # Add simple text label for readability
    combined["label_text"] = np.where(combined["label_union"] == 1, "anomaly", "normal")

    # Attach metrics as metadata rows at the top? Better: save separate JSON if needed.
    combined.to_csv(csv_path, index=False)
    print(f"Saved labeled transactions to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blockchain Transaction Anomaly Detection")
    parser.add_argument("--num_tx", type=int, default=5000, help="Number of simulated transactions")
    parser.add_argument("--anomaly_ratio", type=float, default=0.02, help="Ratio of anomalies in data")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--iso_contamination", type=float, default=0.02, help="Isolation Forest contamination")
    parser.add_argument("--ae_contamination", type=float, default=0.02, help="Autoencoder anomaly fraction for threshold")

    args = parser.parse_args()
    run_workflow(
        num_tx=args.num_tx,
        anomaly_ratio=args.anomaly_ratio,
        random_state=args.random_state,
        iso_contamination=args.iso_contamination,
        ae_contamination=args.ae_contamination,
    )

