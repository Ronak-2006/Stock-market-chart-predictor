# Stock Market Chart Predictor

An AI-assisted stock market trend prediction system that downloads historical market data, engineers technical indicators, trains three forecasting models, and generates an interactive HTML dashboard.

> **Disclaimer:** This project is for educational and research purposes only. Its predictions are not financial advice and should not be used as the sole basis for investment decisions.

## What the project does

The application uses Yahoo Finance data for a configurable stock ticker and performs the following workflow:

1. Downloads historical OHLCV data.
2. Calculates technical indicators and market features.
3. Trains and evaluates three prediction models:
   - **Linear Regression** using technical indicators.
   - **LSTM** neural network using historical closing-price sequences.
   - **ARIMA** time-series model.
4. Produces a multi-day forecast.
5. Builds and opens an interactive Plotly HTML report in the default browser.

The default configuration analyzes **RELIANCE.NS** over the previous **two years** and forecasts the next **30 business days**.

## Dashboard contents

The generated `report.html` dashboard includes:

- Interactive candlestick chart with OHLCV data.
- 10-, 30-, and 50-day moving averages.
- Bollinger Bands.
- RSI, MACD, signal line, volatility, and volume-change features used for modeling.
- Historical test predictions and future forecasts for all three models.
- ARIMA confidence interval.
- LSTM training and validation loss.
- Test-set MAE, RMSE, and R² metrics.

## Project structure

```text
.
├── config.py          # Ticker, forecast horizon, and model configuration
├── data_loader.py     # Yahoo Finance download and feature engineering
├── models.py          # Linear Regression, LSTM, and ARIMA models
├── visualize.py       # Plotly charts and HTML dashboard generation
├── main.py            # Application entry point
├── requirements.txt   # Python dependencies
└── report.html        # Generated interactive report
```

## Requirements

- Python 3.13 or a compatible Python version supported by the dependencies.
- Internet access to download data from Yahoo Finance.
- TensorFlow-compatible system support for running the LSTM model.

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Ronak-2006/Stock-market-chart-predictor.git
cd Stock-market-chart-predictor

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python main.py
```

The program will download the configured data, train the models, generate `report.html`, and attempt to open the report in your default browser.

You can also open the generated report manually:

```bash
# macOS
open report.html

# Linux
xdg-open report.html

# Windows
start report.html
```

## Configuration

Edit `config.py` before running the pipeline:

```python
TICKER = "RELIANCE.NS"  # Yahoo Finance symbol
CURRENCY = "INR (Rs.)"
PERIOD = "2y"
FORECAST_DAYS = 30
```

The file also exposes the LSTM settings and ARIMA order:

```python
LSTM_LOOKBACK = 60
LSTM_EPOCHS = 50
LSTM_BATCH = 32
LSTM_UNITS = 64
ARIMA_ORDER = (5, 1, 0)
```

For another company, replace `TICKER` with its Yahoo Finance symbol. For example, `AAPL` can be used for Apple or `^NSEI` for the NIFTY 50 index.

## Modeling approach

### Linear Regression

Uses engineered technical indicators such as moving averages, RSI, MACD, Bollinger Bands, volatility, and volume change as input features. The data is split chronologically into 80% training and 20% testing sets.

### LSTM

Uses a sequence-to-one recurrent neural network trained on scaled closing prices. The model uses a 60-day lookback window by default and performs recursive multi-step forecasting.

### ARIMA

Fits an ARIMA model to the closing-price series. It uses walk-forward predictions for test evaluation and provides a forecast confidence interval for future values.

## Evaluation

Each model is evaluated on a chronological test set using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

The metrics are displayed in the generated dashboard for side-by-side comparison.

## Data source

Market data is retrieved through [`yfinance`](https://github.com/ranaroussi/yfinance), which accesses Yahoo Finance data. Availability, accuracy, and licensing of data are subject to the source provider's terms.

## Technologies

- Python
- pandas and NumPy
- scikit-learn
- TensorFlow / Keras
- statsmodels
- yfinance
- Plotly
- HTML/CSS

## Limitations

- Financial markets are affected by news, macroeconomic events, liquidity, and other factors not captured by these models.
- Forecast quality depends on the selected ticker, data period, feature quality, and model hyperparameters.
- The simple linear-regression future forecast repeats the latest feature vector.
- The LSTM and ARIMA forecasts can accumulate error over multiple future steps.
- Past performance and model metrics do not guarantee future results.

## License

No license has currently been specified for this repository. Add a license file if you intend to define terms for using, modifying, or distributing the project.
