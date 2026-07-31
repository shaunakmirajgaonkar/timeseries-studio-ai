"""
data_loader.py
--------------
Local data ingestion + preprocessing for TimeSeries Studio AI.
Reads CSV/Excel/Parquet from disk — no network or cloud calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SeriesConfig:
    date_col: str
    target_col: str
    freq: str = "D"          # D, W, M, H, etc.
    group_col: Optional[str] = None  # for multi-series datasets


class DataLoader:
    def load(self, filepath: str) -> pd.DataFrame:
        path = Path(filepath)
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return df

    def prepare_series(self, df: pd.DataFrame, config: SeriesConfig) -> pd.DataFrame:
        """Returns a clean 2-column dataframe: ds (datetime), y (float), resampled to freq."""
        work = df[[config.date_col, config.target_col]].copy()
        work.columns = ["ds", "y"]
        work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
        work = work.dropna(subset=["ds"])
        work = work.sort_values("ds")
        work = work.set_index("ds").resample(config.freq).mean()
        work["y"] = work["y"].interpolate(method="linear").ffill().bfill()
        work = work.reset_index()
        return work

    def train_test_split(self, series: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if horizon >= len(series):
            horizon = max(1, len(series) // 5)
        train = series.iloc[:-horizon].reset_index(drop=True)
        test = series.iloc[-horizon:].reset_index(drop=True)
        return train, test

    def detect_frequency(self, series: pd.DataFrame) -> str:
        """Best-effort inference of the sampling frequency from the ds column."""
        if len(series) < 3:
            return "D"
        diffs = series["ds"].diff().dropna()
        median_seconds = diffs.dt.total_seconds().median()
        if median_seconds <= 3600:
            return "H"
        elif median_seconds <= 86400:
            return "D"
        elif median_seconds <= 7 * 86400:
            return "W"
        else:
            return "M"
