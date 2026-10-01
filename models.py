"""
models.py
─────────
Three prediction models:
  1. Linear Regression  (sklearn)
  2. LSTM               (TensorFlow / Keras)
  3. ARIMA              (statsmodels)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings

from config import (
    LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH, LSTM_UNITS,
    ARIMA_ORDER, FORECAST_DAYS
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(actual, predicted, label=""):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)
    if label:
        print(f"  [{label}] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

def run_linear_regression(df: pd.DataFrame):
    """Predict Close price using technical indicators as features."""
    print("\n[LR] Training Linear Regression …")

    feature_cols = ["MA_10", "MA_30", "RSI", "MACD", "Signal",
                    "BB_up", "BB_lo", "Volatility", "Vol_change"]
    target_col   = "Close"

    data = df[feature_cols + [target_col]].dropna()
    X    = data[feature_cols].values
    y    = data[target_col].values.ravel()

    # 80 / 20 chronological split
    split  = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = MinMaxScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    model  = LinearRegression()
    model.fit(X_tr, y_tr)

    train_pred = model.predict(X_tr)
    test_pred  = model.predict(X_te)

    metrics = _metrics(y_te, test_pred, "LR-Test")

    # Simple future forecast: repeat last known features FORECAST_DAYS times
    last_features = scaler.transform(data[feature_cols].values[-1:])
    future_pred   = np.full(FORECAST_DAYS, model.predict(last_features)[0])

    return {
        "model":       model,
        "train_idx":   data.index[:split],
        "test_idx":    data.index[split:],
        "train_pred":  train_pred,
        "test_pred":   test_pred,
        "actual":      y,
        "full_idx":    data.index,
        "future_pred": future_pred,
        "metrics":     metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM
# ─────────────────────────────────────────────────────────────────────────────

def _make_sequences(arr: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i])
        y.append(arr[i])
    return np.array(X), np.array(y)


def run_lstm(df: pd.DataFrame):
    """Sequence-to-one LSTM on scaled Close prices."""
    print("\n[LSTM] Building & training LSTM …")

    # Import here to avoid slow TF startup when not needed
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    close  = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y   = _make_sequences(scaled, LSTM_LOOKBACK)
    split  = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True,
             input_shape=(LSTM_LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(LSTM_UNITS // 2, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    history = model.fit(
        X_tr, y_tr,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        validation_data=(X_te, y_te),
        callbacks=[early_stop],
        verbose=0,
    )
    print(f"  [LSTM] Stopped at epoch {len(history.history['loss'])}")

    train_pred = scaler.inverse_transform(model.predict(X_tr, verbose=0))
    test_pred  = scaler.inverse_transform(model.predict(X_te, verbose=0))
    actual_tr  = scaler.inverse_transform(y_tr.reshape(-1, 1))
    actual_te  = scaler.inverse_transform(y_te.reshape(-1, 1))

    metrics = _metrics(actual_te.ravel(), test_pred.ravel(), "LSTM-Test")

    # Multi-step future forecast (recursive)
    last_seq    = scaled[-LSTM_LOOKBACK:].reshape(1, LSTM_LOOKBACK, 1)
    future_scaled = []
    seq         = last_seq.copy()
    for _ in range(FORECAST_DAYS):
        p   = model.predict(seq, verbose=0)[0, 0]
        future_scaled.append(p)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = p
    future_pred = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)).ravel()

    idx_all = df.index[LSTM_LOOKBACK:]
    return {
        "train_idx":   idx_all[:split],
        "test_idx":    idx_all[split:],
        "train_pred":  train_pred.ravel(),
        "test_pred":   test_pred.ravel(),
        "actual_tr":   actual_tr.ravel(),
        "actual_te":   actual_te.ravel(),
        "future_pred": future_pred,
        "metrics":     metrics,
        "history":     history.history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def run_arima(df: pd.DataFrame):
    """Fit ARIMA on Close price series."""
    print("\n[ARIMA] Fitting ARIMA model …")

    close = df["Close"].squeeze()
    split = int(len(close) * 0.8)
    train = close.iloc[:split]
    test  = close.iloc[split:]

    model  = ARIMA(train, order=ARIMA_ORDER)
    fitted = model.fit()

    # In-sample
    in_sample = fitted.fittedvalues

    # Walk-forward forecast on test set
    history    = list(train)
    test_preds = []
    for t in range(len(test)):
        m = ARIMA(history, order=ARIMA_ORDER).fit()
        yhat = m.forecast(steps=1)[0]
        test_preds.append(yhat)
        history.append(test.iloc[t])

    test_preds = np.array(test_preds)
    metrics    = _metrics(test.values, test_preds, "ARIMA-Test")

    # Future forecast — fit on .values to avoid DatetimeIndex issues with get_forecast
    final_model  = ARIMA(close.values, order=ARIMA_ORDER).fit()
    forecast_res = final_model.get_forecast(steps=FORECAST_DAYS)
    future_pred  = np.asarray(forecast_res.predicted_mean)
    conf_int     = np.asarray(forecast_res.conf_int())      # shape (FORECAST_DAYS, 2)

    return {
        "train_idx":    train.index,
        "test_idx":     test.index,
        "in_sample":    in_sample.values,
        "test_pred":    test_preds,
        "actual_train": train.values,
        "actual_test":  test.values,
        "future_pred":  future_pred,
        "conf_lo":      conf_int[:, 0],
        "conf_hi":      conf_int[:, 1],
        "metrics":      metrics,
    }
