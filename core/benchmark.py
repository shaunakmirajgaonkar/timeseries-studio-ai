"""
benchmark.py
------------
Automatically trains and benchmarks multiple forecasting models on a
held-out test split, scores them with standard error metrics, and selects
the best-performing model. Everything runs locally — no external services.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .models import MODEL_REGISTRY
from .data_loader import DataLoader


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


@dataclass
class ModelResult:
    name: str
    model: object
    forecast: pd.DataFrame
    metrics: Dict[str, float]
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    results: List[ModelResult] = field(default_factory=list)
    best_model_name: Optional[str] = None

    def leaderboard(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            if r.error:
                rows.append({"model": r.name, "RMSE": None, "MAE": None, "MAPE": None, "status": f"failed: {r.error}"})
            else:
                rows.append({"model": r.name, "RMSE": r.metrics["rmse"],
                             "MAE": r.metrics["mae"], "MAPE": r.metrics["mape"], "status": "ok"})
        df = pd.DataFrame(rows)
        return df.sort_values("RMSE", na_position="last").reset_index(drop=True)


class ModelBenchmark:
    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or list(MODEL_REGISTRY.keys())
        self.loader = DataLoader()

    def run(self, series: pd.DataFrame, horizon: int, progress_callback=None) -> BenchmarkReport:
        train, test = self.loader.train_test_split(series, horizon)
        report = BenchmarkReport()

        for i, name in enumerate(self.model_names):
            if progress_callback:
                progress_callback(i, len(self.model_names), name)
            try:
                model_cls = MODEL_REGISTRY[name]
                model = model_cls()
                model.fit(train)
                forecast = model.predict(len(test))
                y_true = test["y"].values
                y_pred = forecast["yhat"].values[:len(y_true)]

                metrics = {
                    "rmse": rmse(y_true, y_pred),
                    "mae": mae(y_true, y_pred),
                    "mape": mape(y_true, y_pred),
                }
                report.results.append(ModelResult(name, model, forecast, metrics))
            except Exception as e:
                report.results.append(ModelResult(name, None, None, {}, error=str(e)))

        valid = [r for r in report.results if not r.error]
        if valid:
            best = min(valid, key=lambda r: r.metrics["rmse"])
            report.best_model_name = best.name
        return report

    def refit_best_on_full_data(self, series: pd.DataFrame, model_name: str, horizon: int):
        """Refit the chosen model on the FULL series (train+test) to forecast truly into the future."""
        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls()
        model.fit(series)
        forecast = model.predict(horizon)
        return model, forecast
