# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Planned
- Temporal Fusion Transformer (TFT) model option
- Multi-series / multi-SKU batch forecasting
- Export forecasts and reports to CSV/PDF from the dashboard
- Scheduled monitoring mode (drift detection between forecast and actuals)

## [0.1.0] - 2026-07-31
### Added
- Initial release: local data loader (CSV/Excel/Parquet), feature engineering module
- Forecasting models: ARIMA, Prophet, XGBoost, PyTorch LSTM under a unified interface
- Automatic model benchmarking and best-model selection (`benchmark.py`)
- Anomaly detection module (z-score, IQR, residual methods)
- Explainability module: STL decomposition + XGBoost feature importance
- Streamlit dashboard with light theme: Forecast, Model Leaderboard, Anomaly Detection, and Explainability tabs
- Headless CLI (`forecast_cli.py`) for batch/cron use

### Fixed
- Plotly `update_layout` duplicate-kwarg conflicts (`legend`, `yaxis`) across multiple chart calls in the dashboard
