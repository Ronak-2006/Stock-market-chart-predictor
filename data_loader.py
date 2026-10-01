"""
data_loader.py
──────────────
Download & engineer features from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from config import TICKER, PERIOD


# ─── Download ────────────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    """Download OHLCV data and return a cleaned DataFrame."""
    print(f"[DATA] Downloading {TICKER} ({PERIOD}) …")
    df = yf.download(TICKER, period=PERIOD, progress=False, auto_adjust=True)
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"[DATA] {len(df)} rows fetched  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used as ML features."""
    close = df["Close"].squeeze()

    # ── Moving Averages
    df["MA_10"]  = close.rolling(10).mean()
    df["MA_30"]  = close.rolling(30).mean()
    df["MA_50"]  = close.rolling(50).mean()

    # ── Bollinger Bands (20-day, 2 σ)
    rolling20    = close.rolling(20)
    df["BB_mid"] = rolling20.mean()
    df["BB_up"]  = df["BB_mid"] + 2 * rolling20.std()
    df["BB_lo"]  = df["BB_mid"] - 2 * rolling20.std()

    # ── RSI (14-day)
    delta        = close.diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    rs           = gain / loss
    df["RSI"]    = 100 - (100 / (1 + rs))

    # ── MACD
    ema12        = close.ewm(span=12, adjust=False).mean()
    ema26        = close.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── Daily Return & Volatility
    df["Return"]     = close.pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    # ── Volume change
    volume           = df["Volume"].squeeze()
    df["Vol_change"] = volume.pct_change()

    # Replace any inf/-inf values (e.g. from zero-volume days) with NaN, then drop
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df
