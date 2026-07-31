# Roadmap

Planned direction for TimeSeries Studio AI. Not a commitment or timeline —
just where the project is headed.

## Near-term
- [ ] Export forecasts and leaderboard results to CSV/PDF directly from the dashboard
- [ ] Persist benchmark results across sessions (currently in-memory via `st.session_state`)
- [ ] Configurable train/test split ratio in the UI (currently derived from horizon)
- [ ] Better LSTM defaults — auto-tune window size based on detected seasonality period

## Mid-term
- [ ] Temporal Fusion Transformer (TFT) model option via `pytorch-forecasting`,
      for multi-horizon forecasting with attention-based explainability
- [ ] Multi-series / multi-SKU batch forecasting — extend `SeriesConfig` with
      a `group_col` and loop the benchmark per group
- [ ] Global models (e.g. XGBoost with a `series_id` feature) for shared
      learning across many related series

## Long-term
- [ ] Scheduled monitoring mode: re-run forecasts on a cadence, diff against
      actuals, and alert on drift
- [ ] Optional local LLM narrative summaries of forecast changes (via Ollama,
      following the same optional-dependency pattern as the GraphMind AI
      sister project)
- [ ] Plugin system for custom models beyond the built-in `MODEL_REGISTRY`

## Non-goals
- Cloud hosting or SaaS mode — this project is intentionally local-first
- Requiring an API key for any core feature
