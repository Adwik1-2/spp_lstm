# LSTM Stock Forecast

A simple Streamlit app that downloads six years of Yahoo Finance data, trains an
LSTM on daily return/volume features, and produces a recursive price forecast.
Past forecasts are stored in SQLite and scored when actual prices become available.

> Educational project only — not financial advice.

## Highlights

- US, NSE (`.NS`), and other Yahoo Finance tickers
- Future forecasts and leakage-safe historical backtests
- Correct next-session target alignment
- Per-dataset model caching with stale-cache invalidation
- Deterministic forecast paths with bounded recursive returns
- Optional feedback correction learned from resolved forecasts
- Separate Forecast and History views with CSV export
- Automated unit tests and GitHub Actions CI

## Run locally

Requires Python 3.13.

```bash
git clone https://github.com/Adwik1-2/spp_lstm.git
cd spp_lstm
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, enter a ticker and target date, then select
**Generate forecast**.

## How it works

1. Downloads and validates adjusted close and volume data.
2. Builds return, volume-change, SMA-distance, and volatility features.
3. Trains the LSTM using chronological validation and early stopping.
4. Recursively forecasts up to 260 business days and saves the result.

Historical backtests train only on information available before the selected date.
Weekend and known historical holiday dates use the previous available session.
Feedback activates after at least three resolved predictions for the ticker and uses
a clipped median log-return error.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py forecasting.py tests
```

## Project structure

```text
app.py                     Streamlit UI, model training, storage, and charts
forecasting.py             Testable feature and forecasting helpers
tests/test_forecasting.py  Unit tests for core invariants
.github/workflows/ci.yml   CI checks
requirements.txt           Pinned runtime dependencies
```

`predictions.db` is created automatically and ignored by Git. Set
`SPP_LSTM_DB_PATH` to use a different SQLite location. On ephemeral hosting,
use durable external storage if prediction history must survive redeployments.
