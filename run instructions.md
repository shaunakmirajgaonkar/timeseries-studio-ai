# Run Instructions

Detailed setup, run, and troubleshooting guide for TimeSeries Studio AI.

## 1. Prerequisites

- Python 3.10+ (verify with `python3 --version`)
- pip
- ~500MB free disk space for dependencies (PyTorch and Prophet are the largest)

## 2. Clone and set up a virtual environment

```bash
git clone https://github.com/shaunakmirajgaonkar/timeseries-studio-ai.git
cd timeseries-studio-ai
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

Confirm the venv is active — your prompt should show `(venv)` at the start,
and `which python` / `which pip` should point inside the `venv/` folder.

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on Prophet**: Prophet can be slow to build on some systems (it pulls
> in `cmdstanpy`). If installation hangs or fails, remove the `prophet` line
> from `requirements.txt` and deselect it in the dashboard's model list —
> ARIMA, XGBoost, and LSTM will run fine without it.

## 4. Run the dashboard

```bash
streamlit run ui/dashboard.py
```

This opens `http://localhost:8501` in your browser. In the sidebar:

1. Upload a CSV/Excel/Parquet file (a `sample_sales.csv` is provided for testing)
2. Select the date column, target column, and frequency, then **"Load & Prepare Series"**
3. Pick a forecast horizon and which models to benchmark
4. Click **"🚀 Run Benchmark"**
5. Explore the **Forecast**, **Model Leaderboard**, **Anomaly Detection**, and **Explainability** tabs

## 5. Run headless (CLI)

```bash
python forecast_cli.py --file data/sample_sales.csv --date-col date \
    --target-col sales --horizon 30 --freq D --out forecast_output.csv
```

Options:
| Flag | Description | Default |
|---|---|---|
| `--file` | Path to input CSV/Excel/Parquet | required |
| `--date-col` | Name of the date column | required |
| `--target-col` | Name of the numeric target column | required |
| `--freq` | Resampling frequency (D/W/M/H) | `D` |
| `--horizon` | Steps to forecast | `30` |
| `--models` | Space-separated model list | `ARIMA Prophet XGBoost LSTM` |
| `--out` | Output CSV path | `forecast_output.csv` |

## 6. Common issues

**`ModuleNotFoundError: No module named 'X'`**
The venv isn't active, or `pip install -r requirements.txt` didn't fully
complete. Re-activate the venv and re-run the install.

**`command not found: python`** (macOS/Linux)
Use `python3` outside an activated venv. Inside an activated venv, `python`
resolves correctly.

**`externally-managed-environment` error from pip**
You're installing outside a virtual environment on a Homebrew-managed Python.
Always `source venv/bin/activate` first — don't use `--break-system-packages`
as a substitute for the venv.

**spaCy/other model download errors** (only relevant if using this project
alongside GraphMind AI)
Not applicable to TimeSeries Studio AI — no spaCy dependency here.

**Plotly `update_layout` TypeErrors**
Fixed as of v0.1.0 — make sure you're on the latest `ui/dashboard.py`.

## 7. Updating

```bash
git pull
pip install -r requirements.txt --upgrade
```
