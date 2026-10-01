"""
visualize.py
────────────
Build an interactive Plotly HTML dashboard with:
  • Candlestick Chart  (OHLCV + Moving Averages + Bollinger Bands)
  • Prediction Curves  (LR / LSTM / ARIMA)
  • LSTM Training Loss
  • Model Metrics Table
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from config import TICKER, FORECAST_DAYS, OUTPUT_HTML, CURRENCY


# ─── Color palette ───────────────────────────────────────────────────────────
COLORS = {
    "bg":       "#0d1117",
    "paper":    "#161b22",
    "grid":     "#30363d",
    "text":     "#e6edf3",
    "actual":   "#58a6ff",
    "lr":       "#f78166",
    "lstm":     "#7ee787",
    "arima":    "#ffa657",
    "candle_up":"#3fb950",
    "candle_dn":"#f85149",
    "ma10":     "#d2a8ff",
    "ma30":     "#79c0ff",
    "ma50":     "#ffa657",
    "bb":       "#388bfd",
    "volume":   "#21262d",
}

_LAYOUT_DEFAULTS = dict(
    plot_bgcolor  = COLORS["bg"],
    paper_bgcolor = COLORS["paper"],
    font          = dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["grid"]),
    margin        = dict(l=50, r=30, t=60, b=40),
)


def _axis(title="", **kw):
    return dict(title=title, gridcolor=COLORS["grid"],
                zerolinecolor=COLORS["grid"], showgrid=True, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Candlestick Chart
# ─────────────────────────────────────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
        subplot_titles=(f"{TICKER} – Candlestick + Indicators", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=df["Close"].squeeze(),
        increasing_line_color=COLORS["candle_up"],
        decreasing_line_color=COLORS["candle_dn"],
        name="OHLCV",
    ), row=1, col=1)

    # Moving Averages
    for col, color, name in [
        ("MA_10", COLORS["ma10"], "MA 10"),
        ("MA_30", COLORS["ma30"], "MA 30"),
        ("MA_50", COLORS["ma50"], "MA 50"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col].squeeze(),
                line=dict(color=color, width=1.2),
                name=name, opacity=0.85,
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_up" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_up"].squeeze(),
            line=dict(color=COLORS["bb"], width=0.8, dash="dot"),
            name="BB Upper", opacity=0.6,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lo"].squeeze(),
            fill="tonexty",
            fillcolor="rgba(56,139,253,0.08)",
            line=dict(color=COLORS["bb"], width=0.8, dash="dot"),
            name="BB Lower", opacity=0.6,
        ), row=1, col=1)

    # Volume bars
    volume = df["Volume"].squeeze()
    fig.add_trace(go.Bar(
        x=df.index, y=volume,
        marker_color=COLORS["volume"],
        name="Volume", opacity=0.9,
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis2=_axis("Date"),
        yaxis=_axis(f"Price ({CURRENCY})"),
        yaxis2=_axis("Volume"),
        title=dict(text=f"<b>{TICKER} Candlestick Chart</b>",
                   font=dict(size=18), x=0.5),
        height=680,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Prediction Curves
# ─────────────────────────────────────────────────────────────────────────────

def prediction_chart(df: pd.DataFrame, lr_res, lstm_res, arima_res) -> go.Figure:
    # Build future date index
    last_date    = df.index[-1]
    future_dates = pd.bdate_range(last_date, periods=FORECAST_DAYS + 1)[1:]

    fig = go.Figure()

    # ── Actual close price
    close = df["Close"].squeeze()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        line=dict(color=COLORS["actual"], width=2),
        name="Actual Close",
    ))

    # ── Linear Regression – historical test prediction
    fig.add_trace(go.Scatter(
        x=lr_res["test_idx"], y=lr_res["test_pred"],
        line=dict(color=COLORS["lr"], width=1.5, dash="dash"),
        name="LR – Test Pred",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lr_res["future_pred"],
        line=dict(color=COLORS["lr"], width=2),
        name="LR – Forecast",
    ))

    # ── LSTM – historical test prediction
    fig.add_trace(go.Scatter(
        x=lstm_res["test_idx"], y=lstm_res["test_pred"],
        line=dict(color=COLORS["lstm"], width=1.5, dash="dash"),
        name="LSTM – Test Pred",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=lstm_res["future_pred"],
        line=dict(color=COLORS["lstm"], width=2),
        name="LSTM – Forecast",
    ))

    # ── ARIMA – historical test prediction
    fig.add_trace(go.Scatter(
        x=arima_res["test_idx"], y=arima_res["test_pred"],
        line=dict(color=COLORS["arima"], width=1.5, dash="dash"),
        name="ARIMA – Test Pred",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=arima_res["future_pred"],
        line=dict(color=COLORS["arima"], width=2),
        name="ARIMA – Forecast",
    ))

    # ARIMA confidence band
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(arima_res["conf_hi"]) + list(arima_res["conf_lo"][::-1]),
        fill="toself",
        fillcolor="rgba(255,166,87,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="ARIMA 95% CI",
        showlegend=True,
    ))

    # Vertical divider: actual ↔ forecast
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dot", line_color="#8b949e", line_width=1,
        annotation_text="Forecast ->",
        annotation_position="top right",
        annotation_font=dict(color="#8b949e", size=11),
    )

    fig.update_layout(
        title=dict(text=f"<b>{TICKER} – Model Predictions & {FORECAST_DAYS}-Day Forecast</b>",
                   font=dict(size=18), x=0.5),
        xaxis=_axis("Date"),
        yaxis=_axis(f"Price ({CURRENCY})"),
        height=520,
        hovermode="x unified",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM Training Loss
# ─────────────────────────────────────────────────────────────────────────────

def loss_chart(lstm_res) -> go.Figure:
    history = lstm_res["history"]
    epochs  = list(range(1, len(history["loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=history["loss"],
        line=dict(color=COLORS["lstm"], width=2),
        name="Train Loss",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history["val_loss"],
        line=dict(color=COLORS["arima"], width=2, dash="dash"),
        name="Val Loss",
    ))
    fig.update_layout(
        title=dict(text="<b>LSTM Training & Validation Loss</b>",
                   font=dict(size=16), x=0.5),
        xaxis=_axis("Epoch"),
        yaxis=_axis("MSE Loss"),
        height=380,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metrics Table
# ─────────────────────────────────────────────────────────────────────────────

def metrics_table(lr_res, lstm_res, arima_res) -> go.Figure:
    models  = ["Linear Regression", "LSTM", "ARIMA"]
    results = [lr_res["metrics"], lstm_res["metrics"], arima_res["metrics"]]

    rows = {
        "Model": models,
        "MAE":   [f"{r['MAE']:.4f}"  for r in results],
        "RMSE":  [f"{r['RMSE']:.4f}" for r in results],
        "R²":    [f"{r['R2']:.4f}"   for r in results],
    }

    fill_colors = ["#1f2937", "#1a2332", "#0d1117"]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Model</b>", "<b>MAE</b>", "<b>RMSE</b>", "<b>R²</b>"],
            fill_color="#388bfd",
            font=dict(color="white", size=13),
            align="center",
            height=36,
        ),
        cells=dict(
            values=[rows[c] for c in ["Model", "MAE", "RMSE", "R²"]],
            fill_color=[fill_colors] * 4,
            font=dict(color=COLORS["text"], size=12),
            align=["left", "center", "center", "center"],
            height=32,
        ),
    ))
    fig.update_layout(
        title=dict(text="<b>Model Evaluation Metrics (Test Set)</b>",
                   font=dict(size=16), x=0.5),
        paper_bgcolor=COLORS["paper"],
        font=dict(color=COLORS["text"]),
        height=220,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Combine into a single HTML report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(df, lr_res, lstm_res, arima_res):
    print("\n[VIZ] Building interactive HTML report …")

    fig_candle  = candlestick_chart(df)
    fig_pred    = prediction_chart(df, lr_res, lstm_res, arima_res)
    fig_loss    = loss_chart(lstm_res)
    fig_metrics = metrics_table(lr_res, lstm_res, arima_res)

    # Render each figure to HTML div
    to_div = lambda f: pio.to_html(f, full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{TICKER} – AI Stock Trend Prediction</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
  }}

  /* ── Hero Header ── */
  .hero {{
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border-bottom: 1px solid #30363d;
    padding: 40px 24px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 260px;
    background: radial-gradient(ellipse, rgba(56,139,253,0.15) 0%, transparent 70%);
    pointer-events: none;
  }}
  .badge {{
    display: inline-block;
    background: rgba(56,139,253,0.15);
    border: 1px solid rgba(56,139,253,0.4);
    color: #58a6ff;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 18px;
  }}
  .hero h1 {{
    font-size: clamp(24px, 4vw, 42px);
    font-weight: 700;
    background: linear-gradient(135deg, #e6edf3, #58a6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
  }}
  .hero p {{
    color: #8b949e;
    font-size: 14px;
    max-width: 600px;
    margin: 0 auto;
  }}

  /* ── Stat cards ── */
  .stats-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    justify-content: center;
    padding: 28px 24px 0;
    max-width: 1200px;
    margin: 0 auto;
  }}
  .stat-card {{
    flex: 1 1 200px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color .25s, transform .25s;
  }}
  .stat-card:hover {{
    border-color: #58a6ff;
    transform: translateY(-3px);
  }}
  .stat-card .label {{
    font-size: 11px;
    color: #8b949e;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }}
  .stat-card .value {{
    font-size: 22px;
    font-weight: 700;
    color: #e6edf3;
  }}
  .stat-card .value.blue {{ color: #58a6ff; }}
  .stat-card .value.green {{ color: #7ee787; }}
  .stat-card .value.orange {{ color: #ffa657; }}

  /* ── Pill tags ── */
  .tags {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    padding: 20px 24px 0;
  }}
  .tag {{
    font-size: 12px;
    padding: 4px 12px;
    border-radius: 999px;
    border: 1px solid;
  }}
  .tag.algo {{ border-color: #388bfd44; color: #58a6ff; background: #388bfd15; }}
  .tag.tech  {{ border-color: #3fb95044; color: #7ee787; background: #3fb95015; }}

  /* ── Section ── */
  .section {{
    max-width: 1280px;
    margin: 36px auto;
    padding: 0 24px;
  }}
  .section-title {{
    font-size: 13px;
    font-weight: 600;
    color: #58a6ff;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #30363d;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .section-title::before {{
    content: '';
    display: inline-block;
    width: 4px; height: 16px;
    background: #388bfd;
    border-radius: 2px;
  }}
  .chart-box {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    overflow: hidden;
    padding: 12px;
  }}

  /* side-by-side grid */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    color: #484f58;
    font-size: 12px;
    padding: 24px;
    border-top: 1px solid #21262d;
    margin-top: 40px;
  }}
  footer span {{ color: #8b949e; }}
</style>
</head>
<body>

<!-- ── HERO ── -->
<header class="hero">
  <div class="badge">AI / ML  •  Time Series Analysis</div>
  <h1>Stock Market Trend Prediction</h1>
  <p>Historical data → Feature Engineering → Multi-model prediction &amp; {FORECAST_DAYS}-day forecast for <strong>{TICKER}</strong></p>
</header>

<!-- ── TAG PILLS ── -->
<div class="tags">
  <span class="tag algo">Linear Regression</span>
  <span class="tag algo">LSTM (Deep Learning)</span>
  <span class="tag algo">ARIMA</span>
  <span class="tag tech">Time Series Analysis</span>
  <span class="tag tech">Feature Engineering</span>
  <span class="tag tech">Candlestick Chart</span>
  <span class="tag tech">Prediction Curves</span>
</div>

<!-- ── STAT CARDS ── -->
<div class="stats-row" id="stat-cards">
  <div class="stat-card">
    <div class="label">Ticker</div>
    <div class="value blue">{TICKER}</div>
  </div>
  <div class="stat-card">
    <div class="label">Forecast Horizon</div>
    <div class="value orange">{FORECAST_DAYS} Days</div>
  </div>
  <div class="stat-card">
    <div class="label">Models Trained</div>
    <div class="value green">3</div>
  </div>
  <div class="stat-card">
    <div class="label">Features Engineered</div>
    <div class="value">10+</div>
  </div>
</div>

<!-- ── CANDLESTICK ── -->
<div class="section">
  <div class="section-title">Candlestick Chart + Technical Indicators</div>
  <div class="chart-box">{to_div(fig_candle)}</div>
</div>

<!-- ── PREDICTIONS ── -->
<div class="section">
  <div class="section-title">Model Predictions &amp; Forecast</div>
  <div class="chart-box">{to_div(fig_pred)}</div>
</div>

<!-- ── METRICS + LOSS ── -->
<div class="section">
  <div class="section-title">Evaluation &amp; Training</div>
  <div class="grid-2">
    <div class="chart-box">{to_div(fig_metrics)}</div>
    <div class="chart-box">{to_div(fig_loss)}</div>
  </div>
</div>

<footer>
  AI-Based Stock Market Trend Prediction System &nbsp;|&nbsp;
  <span>Linear Regression · LSTM · ARIMA</span> &nbsp;|&nbsp;
  Data: Yahoo Finance &nbsp;|&nbsp; Currency: INR (Rs.) &nbsp;|&nbsp; Built with Python + Plotly
</footer>

</body>
</html>"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[VIZ] Dashboard saved -> {OUTPUT_HTML}")
    return OUTPUT_HTML
