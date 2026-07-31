"""
forecast_cli.py
----------------
Headless CLI for TimeSeries Studio AI — load data, benchmark models,
and export the best forecast to CSV, all locally.

Usage:
    python forecast_cli.py --file data/sales.csv --date-col date \\
        --target-col sales --horizon 30 --freq D
"""

import argparse
from core.data_loader import DataLoader, SeriesConfig
from core.benchmark import ModelBenchmark


def main():
    parser = argparse.ArgumentParser(description="TimeSeries Studio AI — local forecasting CLI")
    parser.add_argument("--file", required=True)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--freq", default="D")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=["ARIMA", "Prophet", "XGBoost", "LSTM"])
    parser.add_argument("--out", default="forecast_output.csv")
    args = parser.parse_args()

    loader = DataLoader()
    raw = loader.load(args.file)
    config = SeriesConfig(date_col=args.date_col, target_col=args.target_col, freq=args.freq)
    series = loader.prepare_series(raw, config)
    print(f"Prepared series: {len(series)} points from {series['ds'].min()} to {series['ds'].max()}")

    benchmark = ModelBenchmark(args.models)

    def progress(i, total, name):
        print(f"[{i+1}/{total}] Training {name}...")

    report = benchmark.run(series, args.horizon, progress_callback=progress)
    print("\nLeaderboard:")
    print(report.leaderboard().to_string(index=False))

    if report.best_model_name:
        print(f"\nBest model: {report.best_model_name}. Refitting on full data...")
        _, forecast = benchmark.refit_best_on_full_data(series, report.best_model_name, args.horizon)
        forecast.to_csv(args.out, index=False)
        print(f"Forecast saved to {args.out}")
    else:
        print("No model trained successfully.")


if __name__ == "__main__":
    main()
