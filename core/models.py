"""
models.py
---------
Unified local forecasting model zoo: ARIMA, Prophet, XGBoost, and an LSTM
(PyTorch). All models run 100% on-device — no cloud training or inference
API calls anywhere in this file.

Each model implements the same interface:
    fit(train_df)              -> self
    predict(horizon)            -> pd.DataFrame[ds, yhat, yhat_lower, yhat_upper]
    feature_importance()        -> dict | None   (for explainability)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional, Dict

from .feature_engineering import FeatureEngineer


# --------------------------------------------------------------- ARIMA -----
class ARIMAModel:
    name = "ARIMA"

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.model = None
        self.result = None
        self.last_date = None
        self.freq = None

    def fit(self, train_df: pd.DataFrame):
        from statsmodels.tsa.arima.model import ARIMA
        self.freq = pd.infer_freq(train_df["ds"]) or "D"
        self.last_date = train_df["ds"].iloc[-1]
        self.model = ARIMA(train_df["y"].values, order=self.order)
        self.result = self.model.fit()
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        forecast = self.result.get_forecast(steps=horizon)
        mean = forecast.predicted_mean
        ci = forecast.conf_int(alpha=0.2)
        future_dates = pd.date_range(self.last_date, periods=horizon + 1, freq=self.freq)[1:]
        return pd.DataFrame({
            "ds": future_dates,
            "yhat": mean,
            "yhat_lower": ci[:, 0] if hasattr(ci, "__getitem__") else ci.iloc[:, 0].values,
            "yhat_upper": ci[:, 1] if hasattr(ci, "__getitem__") else ci.iloc[:, 1].values,
        })

    def feature_importance(self) -> Optional[Dict]:
        return None


# ------------------------------------------------------------- PROPHET -----
class ProphetModel:
    name = "Prophet"

    def __init__(self, yearly_seasonality="auto", weekly_seasonality="auto"):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None
        self.freq = None

    def fit(self, train_df: pd.DataFrame):
        from prophet import Prophet
        self.freq = pd.infer_freq(train_df["ds"]) or "D"
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            interval_width=0.8,
        )
        self.model.fit(train_df[["ds", "y"]])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq)
        forecast = self.model.predict(future)
        tail = forecast.tail(horizon)
        return pd.DataFrame({
            "ds": tail["ds"].values,
            "yhat": tail["yhat"].values,
            "yhat_lower": tail["yhat_lower"].values,
            "yhat_upper": tail["yhat_upper"].values,
        })

    def feature_importance(self) -> Optional[Dict]:
        """Prophet's decomposed components act as a form of explainability."""
        if self.model is None:
            return None
        return {"components": "Use model.plot_components() equivalent — trend/seasonality decomposition available"}


# ------------------------------------------------------------- XGBOOST -----
class XGBoostModel:
    name = "XGBoost"

    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.fe = FeatureEngineer()
        self.feature_cols = None
        self.last_known = None
        self.freq = None

    def fit(self, train_df: pd.DataFrame):
        import xgboost as xgb
        self.freq = pd.infer_freq(train_df["ds"]) or "D"
        supervised = self.fe.build_supervised(train_df)
        self.feature_cols = self.fe.feature_columns(supervised)
        X, y = supervised[self.feature_cols], supervised["y"]
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, objective="reg:squarederror",
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.last_known = train_df.copy()
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        history = self.last_known.copy()
        preds = []
        future_dates = pd.date_range(history["ds"].iloc[-1], periods=horizon + 1, freq=self.freq)[1:]

        for date in future_dates:
            supervised_full = self._build_features_incremental(history, date)
            X_next = supervised_full[self.feature_cols].iloc[[-1]]
            yhat = float(self.model.predict(X_next)[0])
            preds.append(yhat)
            history = pd.concat([history, pd.DataFrame({"ds": [date], "y": [yhat]})], ignore_index=True)

        std = history["y"].std()
        return pd.DataFrame({
            "ds": future_dates,
            "yhat": preds,
            "yhat_lower": np.array(preds) - 1.28 * std,
            "yhat_upper": np.array(preds) + 1.28 * std,
        })

    def _build_features_incremental(self, history: pd.DataFrame, next_date) -> pd.DataFrame:
        extended = pd.concat([history, pd.DataFrame({"ds": [next_date], "y": [np.nan]})], ignore_index=True)
        feat = self.fe.add_calendar_features(extended)
        feat = self.fe.add_lag_features(feat)
        return feat.reset_index(drop=True).ffill()

    def feature_importance(self) -> Optional[Dict]:
        if self.model is None:
            return None
        importances = self.model.feature_importances_
        return dict(sorted(zip(self.feature_cols, importances.tolist()),
                            key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------- LSTM -----
class LSTMModel:
    """A compact single-layer LSTM — deep learning option that still trains
    in seconds on CPU for typical business time series (hundreds-thousands
    of points)."""
    name = "LSTM"

    def __init__(self, window=14, hidden_size=32, epochs=60, lr=0.01):
        self.window = window
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.mean = None
        self.std = None
        self.last_window = None
        self.last_date = None
        self.freq = None

    def _build_torch_model(self):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return Net(self.hidden_size)

    def fit(self, train_df: pd.DataFrame):
        import torch
        import torch.nn as nn
        from .feature_engineering import FeatureEngineer

        self.freq = pd.infer_freq(train_df["ds"]) or "D"
        self.last_date = train_df["ds"].iloc[-1]
        y = train_df["y"].values.astype(np.float32)
        self.mean, self.std = y.mean(), (y.std() + 1e-8)
        y_norm = (y - self.mean) / self.std

        fe = FeatureEngineer()
        X, Y = fe.make_sequences(y_norm, self.window, horizon=1)
        if len(X) < 5:
            # not enough data for the requested window — shrink it
            self.window = max(2, len(y_norm) // 3)
            X, Y = fe.make_sequences(y_norm, self.window, horizon=1)

        X_t = torch.tensor(X).unsqueeze(-1)
        Y_t = torch.tensor(Y)

        self.model = self._build_torch_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = loss_fn(pred, Y_t)
            loss.backward()
            optimizer.step()

        self.last_window = y_norm[-self.window:]
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        import torch
        self.model.eval()
        window = list(self.last_window)
        preds_norm = []
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(np.array(window[-self.window:], dtype=np.float32)).view(1, self.window, 1)
                yhat = self.model(x).item()
                preds_norm.append(yhat)
                window.append(yhat)

        preds = np.array(preds_norm) * self.std + self.mean
        future_dates = pd.date_range(self.last_date, periods=horizon + 1, freq=self.freq)[1:]
        resid_std = self.std * 0.3
        return pd.DataFrame({
            "ds": future_dates,
            "yhat": preds,
            "yhat_lower": preds - 1.28 * resid_std,
            "yhat_upper": preds + 1.28 * resid_std,
        })

    def feature_importance(self) -> Optional[Dict]:
        return None


MODEL_REGISTRY = {
    "ARIMA": ARIMAModel,
    "Prophet": ProphetModel,
    "XGBoost": XGBoostModel,
    "LSTM": LSTMModel,
}
