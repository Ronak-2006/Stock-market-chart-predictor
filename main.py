"""
main.py
───────
Entry-point for the AI-Based Stock Market Trend Prediction System.

Run:  python main.py
"""

import time
import webbrowser
import os

from data_loader import fetch_data, add_features
from models      import run_linear_regression, run_lstm, run_arima
from visualize   import build_report
from config      import OUTPUT_HTML


def main():
    print("=" * 60)
    print("  AI-Based Stock Market Trend Prediction System")
    print("=" * 60)

    t0 = time.time()

    # ── 1. Data
    df = fetch_data()
    df = add_features(df)
    print(f"[DATA] Feature matrix: {df.shape}")

    # ── 2. Models
    lr_res    = run_linear_regression(df)
    lstm_res  = run_lstm(df)
    arima_res = run_arima(df)

    # ── 3. Report
    out = build_report(df, lr_res, lstm_res, arima_res)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s - opening {out} ...")

    # Auto-open in default browser
    webbrowser.open(os.path.abspath(out))


if __name__ == "__main__":
    main()
