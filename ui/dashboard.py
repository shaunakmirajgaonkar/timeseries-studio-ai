"""
dashboard.py
------------
TimeSeries Studio AI — professional local forecasting dashboard.
Run with:  streamlit run ui/dashboard.py

100% local: ARIMA, Prophet, XGBoost, LSTM all train on-device. No cloud
APIs, no external inference calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile

from core.data_loader import DataLoader, SeriesConfig
from core.benchmark import ModelBenchmark
from core.anomaly_detection import AnomalyDetector
from core.explainability import ForecastExplainer

# ---------------------------------------------------------------- CONFIG ---
st.set_page_config(
    page_title="TimeSeries Studio AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#3B6EF5"
ACCENT = "#00B8A9"
WARN = "#F0932B"
DANGER = "#E84356"
BG_LIGHT = "#F7F8FC"
CARD_BG = "#FFFFFF"
BORDER = "#E4E7F0"
TEXT_DARK = "#1E2130"
TEXT_MUTED = "#6B7086"

CUSTOM_CSS = f"""
<style>
.stApp {{ background-color: {BG_LIGHT}; }}
section[data-testid="stSidebar"] {{
    background-color: {CARD_BG};
    border-right: 1px solid {BORDER};
}}
h1, h2, h3, h4 {{ color: {TEXT_DARK}; font-family: 'Segoe UI', sans-serif; }}
p, span, label, div {{ color: {TEXT_DARK}; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED} !important; }}

.metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(30,33,48,0.05);
}}
.metric-value {{ font-size: 28px; font-weight: 700; color: {PRIMARY}; }}
.metric-label {{ font-size: 13px; color: {TEXT_MUTED}; letter-spacing: 0.5px; text-transform: uppercase; }}

.stButton>button {{
    background: linear-gradient(135deg, {PRIMARY}, #6C93F8);
    color: white; border: none; border-radius: 8px; padding: 8px 20px; font-weight: 600;
    box-shadow: 0 2px 6px rgba(59,110,245,0.25);
}}
.stButton>button:hover {{ opacity: 0.92; }}

.badge-best {{
    display:inline-block; padding: 4px 12px; border-radius: 999px;
    background: rgba(0,184,169,0.12); color: {ACCENT}; font-size: 12px; font-weight: 700;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background-color: transparent; color: {TEXT_MUTED}; font-weight: 600;
    border-radius: 8px 8px 0 0; padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important; border-bottom: 2px solid {PRIMARY} !important;
}}

.stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input {{
    background-color: {CARD_BG} !important; border: 1px solid {BORDER} !important;
    color: {TEXT_DARK} !important; border-radius: 8px !important;
}}

[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
hr {{ border-color: {BORDER}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(color=TEXT_DARK, family="Segoe UI"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)


# ---------------------------------------------------------------- STATE ----
if "series" not in st.session_state:
    st.session_state.series = None
if "benchmark_report" not in st.session_state:
    st.session_state.benchmark_report = None
if "chosen_model" not in st.session_state:
    st.session_state.chosen_model = None
if "final_forecast" not in st.session_state:
    st.session_state.final_forecast = None

loader = DataLoader()


# ---------------------------------------------------------------- HEADER ---
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("## 📈 TimeSeries Studio AI")
    st.caption("Local Forecasting Intelligence Platform — Statistical + ML + Deep Learning, 100% offline")
with col_status:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Engine</div>'
        f'<div class="metric-value" style="font-size:18px;">Local ✅</div></div>',
        unsafe_allow_html=True,
    )
st.divider()


# ---------------------------------------------------------------- SIDEBAR --
st.sidebar.markdown("### 📂 Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV / Excel / Parquet", type=["csv", "xlsx", "xls", "parquet"])

if uploaded is not None:
    suffix = "." + uploaded.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    raw_df = loader.load(tmp_path)

    st.sidebar.markdown("### ⚙️ Series Configuration")
    date_col = st.sidebar.selectbox("Date column", raw_df.columns.tolist())
    target_col = st.sidebar.selectbox(
        "Target (value) column",
        [c for c in raw_df.columns if c != date_col],
    )
    freq = st.sidebar.selectbox("Frequency", ["D", "W", "M", "H"], index=0)

    if st.sidebar.button("Load & Prepare Series"):
        config = SeriesConfig(date_col=date_col, target_col=target_col, freq=freq)
        series = loader.prepare_series(raw_df, config)
        st.session_state.series = series
        st.session_state.benchmark_report = None
        st.session_state.final_forecast = None
        st.sidebar.success(f"Series prepared: {len(series)} points ✅")

st.sidebar.divider()
st.sidebar.markdown("### 🔮 Forecast Settings")
horizon = st.sidebar.slider("Forecast horizon (steps)", 7, 180, 30)
models_to_run = st.sidebar.multiselect(
    "Models to benchmark",
    ["ARIMA", "Prophet", "XGBoost", "LSTM"],
    default=["ARIMA", "Prophet", "XGBoost", "LSTM"],
)

run_benchmark = st.sidebar.button("🚀 Run Benchmark")

st.sidebar.divider()
st.sidebar.markdown("### 🚨 Anomaly Detection")
anomaly_method = st.sidebar.selectbox("Method", ["zscore", "iqr", "residual"])
anomaly_threshold = st.sidebar.slider("Sensitivity (z-threshold)", 1.5, 5.0, 3.0, 0.1)


# ---------------------------------------------------------------- MAIN -----
if st.session_state.series is None:
    st.info("👋 Upload a dataset in the sidebar to get started — CSV, Excel, or Parquet with a date column and a numeric target column.")
    st.stop()

series = st.session_state.series

m1, m2, m3, m4 = st.columns(4)
for col, (label, value) in zip(
    [m1, m2, m3, m4],
    [
        ("Data Points", len(series)),
        ("Start", series["ds"].min().strftime("%Y-%m-%d")),
        ("End", series["ds"].max().strftime("%Y-%m-%d")),
        ("Mean Value", f"{series['y'].mean():,.2f}"),
    ],
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value" style="font-size:20px;">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )
st.write("")

if run_benchmark and models_to_run:
    progress_bar = st.progress(0, text="Starting benchmark...")

    def progress_cb(i, total, name):
        progress_bar.progress((i + 1) / total, text=f"Training {name}...")

    benchmark = ModelBenchmark(models_to_run)
    report = benchmark.run(series, horizon, progress_callback=progress_cb)
    st.session_state.benchmark_report = report
    st.session_state.chosen_model = report.best_model_name
    progress_bar.empty()

    if report.best_model_name:
        with st.spinner(f"Refitting best model ({report.best_model_name}) on full data for future forecast..."):
            _, final_forecast = benchmark.refit_best_on_full_data(series, report.best_model_name, horizon)
        st.session_state.final_forecast = final_forecast
        st.success(f"Benchmark complete — best model: **{report.best_model_name}** ✅")


tab_forecast, tab_leaderboard, tab_anomaly, tab_explain = st.tabs(
    ["📈 Forecast", "🏆 Model Leaderboard", "🚨 Anomaly Detection", "🔍 Explainability"]
)

# ---- Forecast tab ----
with tab_forecast:
    st.markdown("#### Historical Data + Future Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["ds"], y=series["y"], mode="lines",
                              name="Historical", line=dict(color=TEXT_DARK, width=1.5)))

    if st.session_state.final_forecast is not None:
        fc = st.session_state.final_forecast
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], mode="lines",
                                  name=f"Forecast ({st.session_state.chosen_model})",
                                  line=dict(color=PRIMARY, width=2.5)))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc["ds"], fc["ds"][::-1]]),
            y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(59,110,245,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="80% Confidence Interval",
            showlegend=True,
        ))
        st.markdown(f'<span class="badge-best">🏆 Best model: {st.session_state.chosen_model}</span>', unsafe_allow_html=True)
    else:
        st.caption("Run the benchmark from the sidebar to generate a forecast.")

    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=480)
    fig.update_layout(legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)

# ---- Leaderboard tab ----
with tab_leaderboard:
    st.markdown("#### Model Performance Comparison")
    if st.session_state.benchmark_report is None:
        st.info("Run the benchmark to compare ARIMA, Prophet, XGBoost, and LSTM on your data.")
    else:
        report = st.session_state.benchmark_report
        lb = report.leaderboard()
        st.dataframe(lb, use_container_width=True, hide_index=True)

        valid_results = [r for r in report.results if not r.error]
        if valid_results:
            fig2 = go.Figure()
            for r in valid_results:
                color = ACCENT if r.name == report.best_model_name else "#B7BCD0"
                fig2.add_trace(go.Bar(x=[r.name], y=[r.metrics["rmse"]], name=r.name,
                                       marker_color=color, showlegend=False))
            fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=350,
                                title="RMSE by Model (lower is better)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("##### Test-window fit (held-out horizon)")
        fig3 = go.Figure()
        _, test = loader.train_test_split(series, horizon)
        fig3.add_trace(go.Scatter(x=test["ds"], y=test["y"], mode="lines+markers",
                                   name="Actual", line=dict(color=TEXT_DARK, width=2)))
        palette = [PRIMARY, ACCENT, WARN, DANGER]
        for i, r in enumerate(valid_results):
            fig3.add_trace(go.Scatter(x=r.forecast["ds"], y=r.forecast["yhat"], mode="lines",
                                       name=r.name, line=dict(color=palette[i % len(palette)], width=1.5, dash="dot")))
        fig3.update_layout(**PLOTLY_TEMPLATE["layout"], height=400)
        st.plotly_chart(fig3, use_container_width=True)

# ---- Anomaly tab ----
with tab_anomaly:
    st.markdown("#### Anomaly Detection")
    detector = AnomalyDetector(method=anomaly_method, threshold=anomaly_threshold)
    result = detector.detect(series)
    n_anomalies = int(result["is_anomaly"].sum())

    c1, c2 = st.columns([1, 3])
    c1.markdown(
        f'<div class="metric-card"><div class="metric-value" style="color:{DANGER}">{n_anomalies}</div>'
        f'<div class="metric-label">Anomalies Found</div></div>', unsafe_allow_html=True
    )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=result["ds"], y=result["y"], mode="lines",
                               name="Value", line=dict(color=TEXT_DARK, width=1.5)))
    fig4.add_trace(go.Scatter(x=result["ds"], y=result["baseline"], mode="lines",
                               name="Baseline", line=dict(color=PRIMARY, width=1, dash="dash")))
    anomalies = result[result["is_anomaly"]]
    fig4.add_trace(go.Scatter(x=anomalies["ds"], y=anomalies["y"], mode="markers",
                               name="Anomaly", marker=dict(color=DANGER, size=9, symbol="x")))
    fig4.update_layout(**PLOTLY_TEMPLATE["layout"], height=450)
    st.plotly_chart(fig4, use_container_width=True)

    if n_anomalies > 0:
        st.markdown("##### Flagged points")
        st.dataframe(
            anomalies[["ds", "y", "anomaly_score"]].sort_values("anomaly_score", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ---- Explainability tab ----
with tab_explain:
    st.markdown("#### Trend, Seasonality & Prediction Drivers")
    explainer = ForecastExplainer()
    decomposed = explainer.decompose(series)

    fig5 = make_subplots(rows=3, cols=1, shared_xaxes=True,
                          subplot_titles=("Trend", "Seasonality", "Residual"))
    fig5.add_trace(go.Scatter(x=decomposed["ds"], y=decomposed["trend"],
                               line=dict(color=PRIMARY)), row=1, col=1)
    fig5.add_trace(go.Scatter(x=decomposed["ds"], y=decomposed["seasonal"],
                               line=dict(color=ACCENT)), row=2, col=1)
    fig5.add_trace(go.Scatter(x=decomposed["ds"], y=decomposed["residual"],
                               line=dict(color=TEXT_MUTED)), row=3, col=1)
    fig5.update_layout(**PLOTLY_TEMPLATE["layout"], height=550, showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("##### Feature Importance (tree-based models only)")
    if st.session_state.benchmark_report is not None:
        xgb_result = next((r for r in st.session_state.benchmark_report.results
                            if r.name == "XGBoost" and not r.error), None)
        if xgb_result:
            importance = xgb_result.model.feature_importance()
            drivers_df = explainer.top_drivers(importance, top_n=10)
            fig6 = go.Figure(go.Bar(
                x=drivers_df["importance"], y=drivers_df["feature"],
                orientation="h", marker_color=PRIMARY,
            ))
            fig6.update_layout(**PLOTLY_TEMPLATE["layout"], height=400)
            fig6.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig6, use_container_width=True)
            st.caption(explainer.summarize_drivers_text(importance))
        else:
            st.info("Run the benchmark with XGBoost included to see feature importance.")
    else:
        st.info("Run the benchmark first to unlock feature-importance driver analysis.")

st.divider()
st.caption("TimeSeries Studio AI · Runs fully on-device · ARIMA · Prophet · XGBoost · PyTorch LSTM · Zero cloud calls")
