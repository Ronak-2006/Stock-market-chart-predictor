# ─────────────────────────────────────────────
#  Project Configuration
# ─────────────────────────────────────────────

TICKER       = "RELIANCE.NS"   # Reliance Industries (NSE) on Yahoo Finance
CURRENCY     = "INR (Rs.)"     # Currency label for charts
PERIOD       = "2y"            # Historical data period
FORECAST_DAYS = 30             # Days to predict ahead

# LSTM hyper-parameters
LSTM_LOOKBACK  = 60            # Time-steps used as input window
LSTM_EPOCHS    = 50
LSTM_BATCH     = 32
LSTM_UNITS     = 64

# ARIMA order
ARIMA_ORDER    = (5, 1, 0)     # (p, d, q)

# Output paths
OUTPUT_HTML    = "report.html" # Final dashboard
