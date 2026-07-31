"""
feature_engineering.py
-----------------------
Local feature engineering for time series: lags, rolling stats, calendar
features, and windowed sequences for deep learning models.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


class FeatureEngineer:
    def __init__(self, lags: List[int] = None, rolling_windows: List[int] = None):
        self.lags = lags or [1, 2, 3, 7, 14]
        self.rolling_windows = rolling_windows or [3, 7, 14]

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["dayofweek"] = df["ds"].dt.dayofweek
        df["day"] = df["ds"].dt.day
        df["month"] = df["ds"].dt.month
        df["quarter"] = df["ds"].dt.quarter
        df["year"] = df["ds"].dt.year
        df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for lag in self.lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        for window in self.rolling_windows:
            df[f"roll_mean_{window}"] = df["y"].shift(1).rolling(window).mean()
            df[f"roll_std_{window}"] = df["y"].shift(1).rolling(window).std()
        return df

    def build_supervised(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.add_calendar_features(df)
        out = self.add_lag_features(out)
        out = out.dropna().reset_index(drop=True)
        return out

    def feature_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = {"ds", "y"}
        return [c for c in df.columns if c not in exclude]

    def make_sequences(self, series: np.ndarray, window: int, horizon: int = 1
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """For LSTM/TFT-style models: sliding windows of length `window` -> next `horizon` steps."""
        X, y = [], []
        for i in range(len(series) - window - horizon + 1):
            X.append(series[i:i + window])
            y.append(series[i + window:i + window + horizon])
        return np.array(X), np.array(y)
