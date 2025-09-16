# Anomaly Detection in Blockchain Transactions

This project simulates blockchain-like transactions and detects anomalies using Isolation Forest (scikit-learn) and an Autoencoder (TensorFlow/Keras). It includes preprocessing, evaluation metrics, visualization, and a simple text-based report.

## Workflow

```
[Data Collection]
       |
       v
[Preprocessing]
       |
       v
[Feature Extraction]
       |
       v
[Anomaly Detection]
 (Isolation Forest &
     Autoencoder)
       |
       v
   [Reporting]
 (metrics, plot, text)
```

## Quickstart

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the full workflow:
```bash
python main.py --num_tx 5000 --anomaly_ratio 0.03 --random_state 42
```

Outputs are saved under `outputs/`:
- `anomalies_scatter.png`: PCA scatter with anomalies highlighted
- `anomaly_report.txt`: Text report with metrics and top anomalies

## Project Structure

```
.
├── main.py
├── requirements.txt
├── README.md
└── src
    ├── __init__.py
    ├── data.py
    ├── preprocessing.py
    ├── evaluation.py
    ├── visualization.py
    ├── reporting.py
    └── models
        ├── __init__.py
        ├── isolation_forest_model.py
        └── autoencoder_model.py
```

## Notes
- The Autoencoder is trained only on normal transactions and then used to flag anomalies based on reconstruction error.
- Isolation Forest is an unsupervised baseline trained on all features.
- Precision/Recall/F1 are computed using the simulated ground truth labels.

