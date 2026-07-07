# 📈 LSTM Stock Price Predictor

An interactive **Streamlit** web app that forecasts stock prices with an **LSTM
(Long Short-Term Memory) neural network** built in TensorFlow/Keras. It pulls
6 years of market data from Yahoo Finance, engineers return/volatility features,
trains an LSTM on the fly, and produces a **recursive multi-step forecast** up to
any target date — visualised against real prices.

It also **remembers** and **learns**: every prediction is persisted to a local
SQLite database, scored against reality once the date arrives, and the model's
systematic error is fed back to sharpen future forecasts.

---

## ✨ Features

- **Live data** — 6 years of adjusted OHLCV pulled from Yahoo Finance (`yfinance`).
- **Feature engineering** — daily return, volume change, distance from the 10-day
  SMA, and 10-day rolling volatility.
- **LSTM forecasting** — a 2-layer LSTM (BatchNorm + Dropout) predicts the *next-day
  return*, applied recursively to project a full price path to the target date.
- **Early stopping** — training runs up to 50 epochs but stops when validation loss
  plateaus and restores the best weights (better generalisation, no over-training).
- **Backtesting view** — pick a past date and see predicted vs. **actual** price with
  a live error margin.
- **⚡ SQLite prediction cache** — results are saved and reused, so repeat runs skip
  the expensive LSTM training entirely.
- **⏳ Live status panel** — shows a "Calculating…" spinner while training and a
  per-step progress bar during forecasting, so long runs never look frozen.
- **📜 History & Accuracy panel** — every saved prediction is scored against the real
  close: average error %, directional hit-rate, CSV export, and clear-history.
- **🔁 Feedback loop (adaptive bias correction)** — learns the model's *systematic*
  per-day error from resolved predictions and corrects future forecasts. Improves as
  more predictions resolve (sidebar toggle).
- **Global tickers** — US equities plus Indian NSE stocks (append `.NS`).

---
<img width="1918" height="910" alt="image" src="https://github.com/user-attachments/assets/48de81ca-9256-4710-9f66-93ca695d3e5a" />


## 🧠 How it works

```
Yahoo Finance ─▶ Feature engineering ─▶ LSTM training ─▶ Recursive forecast ─▶ Chart
                        ▲                                         │
                        │                                         ▼
                  Feedback bias  ◀────  score vs. real close  ◀── SQLite store
                  (learn from            (History panel)          (cache + history)
                   past errors)
```

1. **`load_data(ticker)`** — downloads 6 years of prices (cached in memory).
2. **`preprocess_data(df)`** — builds the 4 model features + the next-day-return target.
3. **`train_model(...)`** — scales features, windows them into 60-day sequences, and
   trains the LSTM (with early stopping) to predict a standardised next-day return.
4. **Recursive forecast** — starting from the last 60 days, the model predicts one
   step, feeds that prediction back into the input window, and repeats until the
   target date. Small random market noise is added on multi-step paths to mimic
   realistic volatility.
5. **Persistence** — before predicting, the app checks the SQLite cache; on a miss it
   trains, forecasts, and **saves** the result.
6. **Feedback correction** — see below.
<img width="1918" height="903" alt="image" src="https://github.com/user-attachments/assets/f02b42df-4736-40f5-b12f-9a889a0845b3" />


### 🗄️ The prediction store (SQLite)

Each forecast is stored in `predictions.db`:

| Column            | Meaning                                             |
|-------------------|-----------------------------------------------------|
| `ticker`          | e.g. `AAPL`, `RELIANCE.NS`                           |
| `target_date`     | date being forecast                                 |
| `model_version`   | bump to invalidate old rows when the model changes  |
| `predicted_price` | final forecast price                                |
| `last_close`      | baseline close the forecast was made against        |
| `pred_dates` / `pred_prices` | full forecast path (JSON)                |
| `generated_at`    | when it was computed                                |

**Cache key = `(ticker, target_date, model_version)`.** The logic is simply:

```python
result = get_cached_prediction(ticker, target_date)
if result is None:            # cache miss
    result = train_and_forecast(...)
    save_prediction(...)      # store for next time
# cache hit → reuse, no retraining
```

> Why `sqlite3`? It ships with Python (no server, no extra dependency), the DB is a
> single file, and it stays fully queryable — ideal for the History & Accuracy panel.

### 🔁 The feedback loop (adaptive bias correction)

The model has a *systematic* tendency to over- or under-predict. Instead of
retraining (past actuals are already in the freshly-downloaded training data), the
app **measures that bias and corrects for it**:

1. For every stored prediction of the ticker whose date has already **resolved** (real
   close now exists), it compares predicted vs. actual and computes a **per-day return
   error**.
2. It averages those errors into a single `bias_per_day`, **excluding the date being
   predicted** (no leakage) and **clipping each sample to ±2%/day** (robustness).
3. Each step of the next forecast is nudged: `pred_ret += bias_per_day`.

Rules: it activates only with **≥ 3 resolved predictions** for that ticker, is
**per-ticker**, and uses **all** resolved predictions (not just 3). Checking whether
past predictions resolved is automatic on every run — you never re-run old dates.

---

## 🚀 Getting started

### Prerequisites
- Python **3.13** (see `runtime.txt` / `.python-version`)

### Installation

```bash
git clone https://github.com/Adwik1-2/spp_lstm.git
cd spp_lstm

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Enter a ticker (e.g. `AAPL`), pick a target
date, choose whether to apply the feedback correction, and click **🚀 Predict Stock
Price**. `predictions.db` is created automatically on the first run.

---

## 📂 Project structure

```
spp_lstm/
├── app.py            # Streamlit app: data, LSTM, forecast, cache, history, feedback
├── requirements.txt  # pinned Python dependencies
├── runtime.txt       # Python version (for deployment)
├── .python-version   # Python version (pyenv)
├── .gitignore        # ignores predictions.db, __pycache__, venv
└── README.md
```

---

## ⚙️ Configuration (constants in `app.py`)

| Setting                | Default            | Note                                          |
|------------------------|--------------------|-----------------------------------------------|
| `DB_PATH`              | `predictions.db`   | SQLite store location                         |
| `MODEL_VERSION`        | `v2`               | bump to invalidate all cached predictions     |
| `EPOCHS`               | `50`               | max training epochs (early stopping halts sooner) |
| `EARLY_STOP_PATIENCE`  | `6`                | epochs of no val-loss improvement before stop |
| `MIN_FEEDBACK_SAMPLES` | `3`                | resolved predictions needed before correcting |
| `MAX_DAILY_BIAS`       | `0.02`             | cap on the feedback correction (±2%/day)      |
| Lookback window        | `time_step = 60`   | past days fed to the LSTM per step            |
| History depth          | `period="6y"`      | data downloaded per ticker                    |

---

## ⚠️ Notes & limitations

- **Educational project — not financial advice.** Stock returns are close to random;
  do not trade on these forecasts. Large error margins are expected.
- **The app runs only while open.** Streamlit executes on interaction; it does not
  predict in the background. The feedback pool grows as *you* make predictions — there
  is no 24/7 scheduler (that would need an external cron + a hosted DB).
- **Deployment & the store:** on ephemeral hosts (Streamlit Cloud, Heroku) the local
  filesystem is wiped on redeploy, so `predictions.db` may reset. For durable storage,
  point it at a hosted database (e.g. Postgres).
- **Multi-step forecasts** add random noise, so a fresh future forecast is
  non-deterministic; with the feedback toggle on, it recomputes to reflect the latest
  learning (the LSTM itself stays cached, so this is cheap).

---

## 🛠️ Tech stack

Python 3.13 · Streamlit · TensorFlow/Keras · scikit-learn · pandas · NumPy · Plotly ·
yfinance · SQLite
