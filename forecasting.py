"""Pure forecasting helpers shared by the Streamlit app and unit tests.

Keeping these functions free of Streamlit and TensorFlow makes the most important
data-alignment rules cheap to test.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["ret1", "vol_change", "sma_10_dist", "volatility_10"]
_TICKER_PATTERN = re.compile(r"^[A-Z0-9^][A-Z0-9.^=_-]{0,24}$")


def normalize_ticker(value: str) -> str:
    """Return a Yahoo Finance-style ticker or raise a user-facing error."""
    ticker = value.strip().upper()
    if not ticker:
        raise ValueError("Enter a ticker symbol.")
    if not _TICKER_PATTERN.fullmatch(ticker):
        raise ValueError(
            "Ticker contains unsupported characters. Use letters, numbers, '.', '-', '^', '=' or '_'."
        )
    return ticker


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create model features and the next-session return target."""
    required = {"Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Market data is missing: {', '.join(sorted(missing))}.")

    feat = df[["Close", "Volume"]].copy()
    feat["ret1"] = feat["Close"].pct_change(fill_method=None)
    feat["vol_change"] = feat["Volume"].pct_change(fill_method=None)
    sma_10 = feat["Close"].rolling(10).mean()
    feat["sma_10_dist"] = (feat["Close"] - sma_10) / sma_10
    feat["volatility_10"] = feat["ret1"].rolling(10).std()
    feat["target_ret_next"] = feat["ret1"].shift(-1)
    clean = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat, clean


def build_sequences(
    features: np.ndarray, targets: np.ndarray, time_step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build windows aligned to the return immediately after each window.

    ``targets[row]`` is the next-session return associated with ``features[row]``.
    Therefore a window ending at ``row`` must use that same row's target.  Keeping
    this rule here prevents the subtle one-session gap that can otherwise creep
    into LSTM training code.
    """
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("targets must be 1-D and match the feature rows")
    if time_step < 1 or len(features) <= time_step:
        raise ValueError("not enough rows for the requested lookback window")

    windows = [features[end - time_step : end] for end in range(time_step, len(features) + 1)]
    aligned_targets = [targets[end - 1] for end in range(time_step, len(features) + 1)]
    return np.asarray(windows, dtype=np.float32), np.asarray(aligned_targets, dtype=np.float32)


def data_fingerprint(data: pd.DataFrame) -> str:
    """Return a stable cache key that changes whenever training data changes."""
    relevant = data[FEATURE_COLUMNS + ["target_ret_next"]]
    row_hashes = pd.util.hash_pandas_object(relevant, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def display_currency(ticker: str) -> str:
    """Best-effort currency marker for the ticker examples supported by the UI."""
    return "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"


def format_volume(value: float) -> str:
    """Format trading volume without assuming an Indian or US unit system."""
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def per_step_log_error(predicted_price: float, actual_price: float, steps: int) -> float:
    """Return a compoundable per-step error for one resolved forecast."""
    if predicted_price <= 0 or actual_price <= 0 or steps < 1:
        raise ValueError("prices and step count must be positive")
    return float(np.log(actual_price / predicted_price) / steps)


def apply_log_bias(
    predicted_return: float, bias_per_step: float, max_abs_return: float = 0.20
) -> float:
    """Apply feedback in log-return space and bound pathological model outputs."""
    safe_return = float(np.clip(predicted_return, -0.95, max_abs_return))
    corrected = float(np.expm1(np.log1p(safe_return) + bias_per_step))
    return float(np.clip(corrected, -max_abs_return, max_abs_return))
