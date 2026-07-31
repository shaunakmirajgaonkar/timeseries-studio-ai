"""
anomaly_detection.py
---------------------
Local anomaly detection for time series: rolling z-score, IQR, and
residual-based detection using a smoothed baseline. No external services.
"""

import numpy as np
import pandas as pd


class AnomalyDetector:
    def __init__(self, method: str = "zscore", window: int = 14, threshold: float = 3.0):
        self.method = method
        self.window = window
        self.threshold = threshold

    def detect(self, series: pd.DataFrame) -> pd.DataFrame:
        df = series.copy()
        if self.method == "zscore":
            df = self._zscore_method(df)
        elif self.method == "iqr":
            df = self._iqr_method(df)
        else:
            df = self._residual_method(df)
        return df

    def _zscore_method(self, df: pd.DataFrame) -> pd.DataFrame:
        roll_mean = df["y"].rolling(self.window, min_periods=1, center=True).mean()
        roll_std = df["y"].rolling(self.window, min_periods=1, center=True).std().replace(0, np.nan)
        z = (df["y"] - roll_mean) / roll_std
        df["anomaly_score"] = z.abs().fillna(0)
        df["is_anomaly"] = df["anomaly_score"] > self.threshold
        df["baseline"] = roll_mean
        return df

    def _iqr_method(self, df: pd.DataFrame) -> pd.DataFrame:
        q1, q3 = df["y"].quantile(0.25), df["y"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df["is_anomaly"] = (df["y"] < lower) | (df["y"] > upper)
        df["anomaly_score"] = (df["y"] - df["y"].median()).abs() / (iqr + 1e-8)
        df["baseline"] = df["y"].rolling(self.window, min_periods=1, center=True).mean()
        return df

    def _residual_method(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = df["y"].rolling(self.window, min_periods=1, center=True).mean()
        residual = df["y"] - trend
        std = residual.std() + 1e-8
        df["anomaly_score"] = (residual / std).abs()
        df["is_anomaly"] = df["anomaly_score"] > self.threshold
        df["baseline"] = trend
        return df
