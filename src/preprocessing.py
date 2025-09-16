from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning: drop NA, remove non-positive amounts/gas, ensure dtypes.
    """
    df = df.copy()
    df = df.dropna()
    # Keep only positive amounts and gas fees
    df = df[(df["amount"] > 0) & (df["gas_fee"] > 0)]
    # Ensure integer types for addresses and timestamp
    df["sender"] = df["sender"].astype(int)
    df["receiver"] = df["receiver"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature extraction from raw fields. Keeps numeric features only.
    - amount
    - gas_fee
    - is_self_transfer
    - address_interaction_delta (|sender - receiver|)
    - hour_of_day (cyclical encoding: sin/cos)
    - day_of_week (cyclical encoding: sin/cos)
    - amount_to_gas_ratio
    """
    df = df.copy()

    # Self transfer flag
    df["is_self_transfer"] = (df["sender"] == df["receiver"]).astype(int)

    # Address distance as a proxy for topology feature
    df["address_interaction_delta"] = (df["sender"] - df["receiver"]).abs()

    # Timestamps to datetime
    dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    hour = dt.dt.hour.values
    dow = dt.dt.dayofweek.values  # 0=Mon

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Ratios
    df["amount_to_gas_ratio"] = df["amount"] / (df["gas_fee"] + 1e-8)

    # Keep selected feature columns
    feature_cols = [
        "amount",
        "gas_fee",
        "amount_to_gas_ratio",
        "is_self_transfer",
        "address_interaction_delta",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]

    return df[feature_cols]


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Convert engineered features to normalized matrix X and return scaler state.
    Returns:
      X: np.ndarray shape (n_samples, n_features)
      artifacts: { 'scaler': StandardScaler, 'feature_names': list[str] }
    """
    features = engineer_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values.astype(float))
    artifacts = {
        "scaler": scaler,
        "feature_names": list(features.columns),
    }
    return X, artifacts

