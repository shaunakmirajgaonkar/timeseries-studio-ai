"""
explainability.py
------------------
Local explainability layer: STL-style decomposition (trend/seasonality/
residual) plus feature-importance surfacing for tree-based models.
No external API calls — uses statsmodels + the model's own attributes.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict


class ForecastExplainer:
    def decompose(self, series: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
        from statsmodels.tsa.seasonal import STL

        s = series.set_index("ds")["y"]
        if period is None:
            period = self._infer_period(series)

        try:
            stl = STL(s, period=period, robust=True)
            result = stl.fit()
            out = series.copy()
            out["trend"] = result.trend.values
            out["seasonal"] = result.seasonal.values
            out["residual"] = result.resid.values
            return out
        except Exception:
            # fall back to a simple rolling decomposition if STL fails (e.g. too few points)
            out = series.copy()
            out["trend"] = s.rolling(max(2, period), min_periods=1, center=True).mean().values
            out["seasonal"] = (s.values - out["trend"].values)
            out["residual"] = 0.0
            return out

    def _infer_period(self, series: pd.DataFrame) -> int:
        n = len(series)
        diffs = series["ds"].diff().dropna()
        if diffs.empty:
            return min(7, max(2, n // 2))
        median_days = diffs.dt.total_seconds().median() / 86400
        if median_days < 1.5:
            return 7           # daily data -> weekly seasonality
        elif median_days < 10:
            return 4           # weekly data -> ~monthly seasonality
        else:
            return 12          # monthly data -> yearly seasonality

    def top_drivers(self, feature_importance: Optional[Dict], top_n: int = 8) -> pd.DataFrame:
        if not feature_importance:
            return pd.DataFrame(columns=["feature", "importance"])
        items = list(feature_importance.items())[:top_n]
        return pd.DataFrame(items, columns=["feature", "importance"])

    def summarize_drivers_text(self, feature_importance: Optional[Dict]) -> str:
        if not feature_importance:
            return "This model type doesn't expose per-feature importances directly."
        top = self.top_drivers(feature_importance, top_n=3)
        if top.empty:
            return "No driver information available."
        parts = [f"{row.feature} ({row.importance:.2%})" for row in top.itertuples()]
        return "Top prediction drivers: " + ", ".join(parts)
