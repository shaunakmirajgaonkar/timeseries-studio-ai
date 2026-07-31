# 📈 TimeSeries Studio AI

**Local Forecasting Intelligence Platform** — Statistical + Machine Learning + Deep Learning, 100% offline.

TimeSeries Studio AI benchmarks multiple forecasting models (ARIMA, Prophet, XGBoost, LSTM) on your data, automatically selects the best performer, detects anomalies, and explains prediction drivers — all running entirely on your own machine. No API keys, no cloud calls, no data leaves your device.

![status](https://img.shields.io/badge/status-active-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## ✨ Features

- **Multi-model benchmarking** — ARIMA, Prophet, XGBoost, and PyTorch LSTM trained and scored side by side
- **Automatic model selection** — picks the best model by RMSE on a held-out test window, then refits on full data for the real forecast
- **Anomaly detection** — z-score, IQR, and residual-based methods with adjustable sensitivity
- **Explainability** — STL trend/seasonality/residual decomposition plus XGBoost feature importance
- **Interactive dashboard** — Streamlit + Plotly, light professional theme, zero setup beyond `pip install`
- **Headless CLI** — run the full pipeline from the command line for batch jobs or cron schedules

## 🖥️ Screenshots

See [`doc/screenshots`](doc/screenshots) for dashboard walkthroughs of the Forecast, Leaderboard, Anomaly Detection, and Explainability tabs.

## 🚀 Quick Start

```bash
git clone https://github.com/shaunakmirajgaonkar/timeseries-studio-ai.git
cd timeseries-studio-ai
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run ui/dashboard.py
```

See [`run instructions.md`](run%20instructions.md) for detailed setup, troubleshooting, and CLI usage.

## 🏗️ Architecture

```
timeseries_studio/
├── core/
│   ├── data_loader.py         # CSV/Excel/Parquet ingestion + resampling
│   ├── feature_engineering.py # lags, rolling stats, calendar features
│   ├── models.py              # ARIMA / Prophet / XGBoost / LSTM wrappers
│   ├── benchmark.py           # trains + scores + selects best model
│   ├── anomaly_detection.py   # zscore / IQR / residual anomaly flags
│   └── explainability.py      # STL decomposition + feature importance
├── ui/
│   └── dashboard.py           # Streamlit dashboard
├── forecast_cli.py            # headless CLI
├── doc/                       # additional documentation & screenshots
└── requirements.txt
```

## 🧪 Tech Stack

| Layer | Library |
|---|---|
| Data loading | pandas |
| Statistical model | statsmodels (ARIMA) |
| ML/DL forecasting | Prophet, XGBoost, PyTorch (LSTM) |
| Anomaly detection | rolling z-score / IQR / residual methods |
| Explainability | STL decomposition, XGBoost feature importance |
| Dashboard UI | Streamlit + Plotly |

## 🗺️ Roadmap

See [`CHANGELOG.md`](CHANGELOG.md) for release history and [`doc/roadmap.md`](doc/roadmap.md) for planned features (Temporal Fusion Transformer, multi-series support).

## 🤝 Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community standards.

## 🔒 Security

To report a vulnerability, see [`SECURITY.md`](SECURITY.md).

## 📄 License

MIT — see [`LICENSE`](LICENSE).

## 🙏 Acknowledgments

See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md).
