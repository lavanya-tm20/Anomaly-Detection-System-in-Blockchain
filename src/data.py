import numpy as np
import pandas as pd
from typing import Tuple


def simulate_blockchain_transactions(
    num_transactions: int = 5000,
    anomaly_ratio: float = 0.02,
    random_state: int | None = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate blockchain-like transactions with a fraction of anomalies.

    Columns: sender, receiver, amount, timestamp, gas_fee
    Returns: (DataFrame, labels) where labels=1 for anomaly, 0 for normal
    """
    rng = np.random.default_rng(random_state)

    num_anomalies = int(num_transactions * anomaly_ratio)
    num_normals = num_transactions - num_anomalies

    # Simulate user addresses as integers (could be hashed/encoded in practice)
    num_users = max(1000, int(num_transactions * 0.2))
    senders_normal = rng.integers(0, num_users, size=num_normals)
    receivers_normal = (senders_normal + rng.integers(1, 50, size=num_normals)) % num_users

    # Normal amounts: log-normal distribution
    amount_normal = rng.lognormal(mean=1.5, sigma=0.5, size=num_normals)
    # Normal gas fees correlate weakly with amount
    gas_normal = np.clip(amount_normal * rng.normal(0.01, 0.002, size=num_normals), 0.0001, None)

    # Timestamps over a period (e.g., 30 days), in seconds
    start_ts = pd.Timestamp("2024-01-01").value // 10**9
    end_ts = pd.Timestamp("2024-01-31").value // 10**9
    timestamp_normal = rng.integers(start_ts, end_ts, size=num_normals)

    # Assemble normal df
    df_normal = pd.DataFrame(
        {
            "sender": senders_normal,
            "receiver": receivers_normal,
            "amount": amount_normal,
            "timestamp": timestamp_normal,
            "gas_fee": gas_normal,
        }
    )

    # Anomalies: unusual amounts, self-transfers, bursts, extreme gas, rare addresses
    senders_anom = rng.integers(0, num_users, size=num_anomalies)
    # Some anomalies send to themselves (self-transfer) or rare receivers
    receivers_anom = np.where(
        rng.random(size=num_anomalies) < 0.4,
        senders_anom,
        rng.integers(num_users, num_users + 200, size=num_anomalies),  # rare addresses
    )

    # Extreme amounts (very high or very low)
    amount_anom = rng.choice([
        rng.lognormal(mean=4.0, sigma=0.7, size=num_anomalies),  # very high
        rng.lognormal(mean=-1.0, sigma=0.3, size=num_anomalies), # very low
    ])

    # Gas anomalies: disproportionately high or near-zero
    gas_anom = np.where(
        rng.random(size=num_anomalies) < 0.5,
        np.clip(amount_anom * rng.normal(0.1, 0.02, size=num_anomalies), 0.0001, None),
        np.clip(amount_anom * rng.normal(0.001, 0.0002, size=num_anomalies), 0.00005, None),
    )

    # Timestamp anomalies: bursts (same second) or out-of-range
    ts_choices = rng.integers(end_ts, end_ts + 7 * 24 * 3600, size=num_anomalies)
    timestamp_anom = np.where(rng.random(size=num_anomalies) < 0.5, ts_choices, start_ts - rng.integers(1, 7 * 24 * 3600, size=num_anomalies))

    df_anom = pd.DataFrame(
        {
            "sender": senders_anom,
            "receiver": receivers_anom,
            "amount": amount_anom,
            "timestamp": timestamp_anom,
            "gas_fee": gas_anom,
        }
    )

    df = pd.concat([df_normal, df_anom], ignore_index=True)
    labels = np.array([0] * num_normals + [1] * num_anomalies)

    # Shuffle to mix normals and anomalies
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)
    labels = labels[perm]

    return df, labels

