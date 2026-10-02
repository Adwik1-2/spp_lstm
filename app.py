import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.utils import set_random_seed

from forecasting import (
    FEATURE_COLUMNS,
    apply_log_bias,
    build_sequences,
    data_fingerprint,
    display_currency,
    engineer_features,
    format_volume,
    normalize_ticker,
    per_step_log_error,
)


st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Prediction cache (SQLite)
# ---------------------------------------------------------------------------
# Persists each prediction so a repeat run for the same ticker + target date
# returns the stored result instead of retraining the LSTM.
# Cache key = (ticker, target_date, MODEL_VERSION). Bump MODEL_VERSION whenever
# the model / features change so old rows are treated as stale automatically.
DB_PATH = Path(os.environ.get("SPP_LSTM_DB_PATH", Path(__file__).with_name("predictions.db")))
MODEL_VERSION = "v4"          # bounded recursion + compound feedback correction
EPOCHS = 50                   # max epochs; EarlyStopping usually halts sooner
EARLY_STOP_PATIENCE = 6       # stop if val_loss doesn't improve for N epochs
MIN_FEEDBACK_SAMPLES = 3      # min resolved predictions before applying correction
MAX_DAILY_BIAS = 0.02         # cap the feedback correction at ±2%/day (robustness)
MAX_DAILY_RETURN = 0.20       # guardrail for pathological recursive model outputs
MAX_FORECAST_STEPS = 260      # approximately one trading year


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ticker           TEXT,
            target_date      TEXT,
            model_version    TEXT,
            predicted_price  REAL,
            last_close       REAL,
            pred_dates       TEXT,   -- JSON list of ISO date strings
            pred_prices      TEXT,   -- JSON list of floats
            generated_at     TEXT,
            data_fingerprint TEXT,
            PRIMARY KEY (ticker, target_date, model_version)
        )
    """)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "data_fingerprint" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN data_fingerprint TEXT")
        conn.commit()
    return conn


def get_cached_prediction(ticker, target_date, model_version=MODEL_VERSION):
    conn = get_conn()
    row = conn.execute(
        """SELECT predicted_price, last_close, pred_dates, pred_prices, generated_at,
                  data_fingerprint
           FROM predictions WHERE ticker=? AND target_date=? AND model_version=?""",
        (ticker, str(target_date), model_version),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "predicted_price": row[0],
        "last_close": row[1],
        "pred_dates": [pd.to_datetime(d) for d in json.loads(row[2])],
        "pred_prices": json.loads(row[3]),
        "generated_at": row[4],
        "data_fingerprint": row[5],
    }


def save_prediction(
    ticker, target_date, predicted_price, last_close, pred_dates, pred_prices,
    model_version=MODEL_VERSION, training_fingerprint=None,
):
    conn = get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO predictions
           (ticker, target_date, model_version, predicted_price, last_close,
            pred_dates, pred_prices, generated_at, data_fingerprint)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            ticker,
            str(target_date),
            model_version,
            float(predicted_price),
            float(last_close),
            json.dumps([pd.to_datetime(d).strftime("%Y-%m-%d") for d in pred_dates]),
            json.dumps([float(p) for p in pred_prices]),
            datetime.now().isoformat(timespec="seconds"),
            training_fingerprint,
        ),
    )
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = get_conn()
    rows = conn.execute(
        """SELECT ticker, target_date, predicted_price, last_close, generated_at,
                  model_version
           FROM predictions WHERE model_version LIKE ? ORDER BY generated_at DESC""",
        (f"{MODEL_VERSION}%",),
    ).fetchall()
    conn.close()
    return rows


def clear_predictions():
    conn = get_conn()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def get_ticker_predictions(ticker):
    conn = get_conn()
    rows = conn.execute(
        """SELECT target_date, predicted_price, last_close, pred_prices
           FROM predictions WHERE ticker=? AND model_version LIKE ?
           ORDER BY generated_at DESC""",
        (ticker, f"{MODEL_VERSION}%"),
    ).fetchall()
    conn.close()
    # A target may have both raw and feedback-corrected forecasts. Use only the
    # latest one so that date is not overweighted when estimating future bias.
    latest_by_target = {}
    for row in rows:
        latest_by_target.setdefault(row[0], row)
    return list(latest_by_target.values())

@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker):
    df = yf.download(ticker, period="6y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for column in ("Close", "Volume"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    return engineer_features(df)

@st.cache_resource(show_spinner=False)
def train_model(training_fingerprint, _data, time_step=60):
    """Train one model per unique dataset.

    Streamlit deliberately ignores underscore-prefixed parameters when hashing.
    ``training_fingerprint`` is therefore required even though the data itself is
    passed as ``_data``; without it, a model trained for one ticker can be reused
    for every other ticker.
    """
    del training_fingerprint  # its value is consumed by Streamlit's cache key
    x_df = _data[FEATURE_COLUMNS]
    y_raw = _data["target_ret_next"].values.astype(np.float32)
    X_raw = x_df.values.astype(np.float32)

    split = int(len(X_raw) * 0.9)
    scaler = MinMaxScaler()
    scaler.fit(X_raw[:split])
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    x_all, y_all = build_sequences(X_scaled, y_raw, time_step)
    validation_size = max(1, int(len(x_all) * 0.1))
    if len(x_all) - validation_size < 10:
        raise ValueError("Not enough training sequences after reserving validation data.")
    x_tr, x_val = x_all[:-validation_size], x_all[-validation_size:]
    y_tr, y_val = y_all[:-validation_size], y_all[-validation_size:]

    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_s = (y_tr - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    set_random_seed(42)
    model = Sequential([
        Input(shape=(time_step, x_tr.shape[2])),
        LSTM(32, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(16, return_sequences=False),
        BatchNormalization(), Dropout(0.3),
        Dense(8, activation="relu"), Dense(1)
    ])
    model.compile(optimizer=Adam(1e-3), loss=Huber(0.01))

    # Stop once validation loss plateaus and roll back to the best weights, so we
    # don't over-train (better generalisation than a fixed number of epochs).
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
    model.fit(
        x_tr, y_tr_s, epochs=EPOCHS, batch_size=32,
        validation_data=(x_val, y_val_s), verbose=0,
        callbacks=[early_stop], shuffle=False,
    )

    return model, scaler, y_mean, y_std


def fetch_actual_close(ticker, date_str):
    """Return the real market close for a stored prediction's target date, or
    None if that date hasn't traded yet (a future forecast) / has no data."""
    try:
        df = load_data(ticker)
    except Exception:
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    ts = pd.to_datetime(date_str).normalize()
    if ts in df.index:
        val = df.loc[ts, "Close"]
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        if pd.notna(val):
            return val
    return None


def compute_feedback_bias(ticker, exclude_target=None):
    """Estimate the model's systematic per-day return error from resolved
    predictions of `ticker`, so it can be added back to future forecasts.

    Returns (bias_per_day, n_samples). bias is 0 until at least
    MIN_FEEDBACK_SAMPLES resolved predictions exist (excluding the date being
    predicted, to avoid leakage)."""
    errors = []
    for target_date, predicted_price, last_close, pred_prices_json in get_ticker_predictions(ticker):
        if exclude_target is not None and str(target_date) == str(exclude_target):
            continue
        if not last_close or last_close <= 0:
            continue
        actual = fetch_actual_close(ticker, target_date)
        if actual is None or actual <= 0:
            continue  # not resolved yet (future date) — skip
        try:
            n_steps = max(1, len(json.loads(pred_prices_json)))
        except (TypeError, ValueError):
            n_steps = 1
        try:
            per_day_error = per_step_log_error(predicted_price, actual, n_steps)
        except ValueError:
            continue
        # Work in log-return space so multi-step corrections compound correctly.
        # Clip each sample so one wild backtest cannot dominate the average.
        errors.append(float(np.clip(per_day_error, -MAX_DAILY_BIAS, MAX_DAILY_BIAS)))

    if len(errors) < MIN_FEEDBACK_SAMPLES:
        return 0.0, len(errors)

    # A median is robust to occasional extreme forecasts and bad prints.
    bias = float(np.median(errors))
    bias = float(np.clip(bias, -MAX_DAILY_BIAS, MAX_DAILY_BIAS))
    return bias, len(errors)


def render_history():
    """History & accuracy tab — reads every saved prediction from SQLite and
    scores it against the real close once that date has traded."""
    st.subheader("Prediction history")
    st.caption(
        "Every forecast is persisted to a local SQLite database (`predictions.db`). "
        "Re-running the same ticker + date loads the stored result instead of retraining."
    )

    rows = get_all_predictions()
    if not rows:
        st.info("No saved predictions yet. Run a forecast first.")
        return

    records = []
    for ticker, tdate, pred, last_close, gen_at, model_version in rows:
        actual = fetch_actual_close(ticker, tdate)
        if actual is not None and actual > 0:
            err = abs(pred - actual) / actual * 100
            direction = "✅" if (pred >= last_close) == (actual >= last_close) else "❌"
        else:
            err, direction = None, "⏳"
        records.append({
            "Ticker": ticker,
            "Target Date": tdate,
            "Predicted": round(pred, 2),
            "Actual": round(actual, 2) if actual is not None else None,
            "Error %": round(err, 2) if err is not None else None,
            "Direction": direction,
            "Mode": "Feedback" if model_version.endswith("-feedback") else "Base",
            "Saved At": gen_at,
        })

    hist = pd.DataFrame(records)
    resolved = hist[hist["Error %"].notna()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", len(hist))
    c2.metric("Resolved", f"{len(resolved)} / {len(hist)}")
    c3.metric("Avg Error", f"{resolved['Error %'].mean():.2f}%" if len(resolved) else "—")
    if len(resolved):
        hit_rate = (resolved["Direction"] == "✅").mean() * 100
        c4.metric("Direction Hit-Rate", f"{hit_rate:.0f}%")
    else:
        c4.metric("Direction Hit-Rate", "—")

    st.dataframe(hist, hide_index=True, width="stretch")

    dl_col, clr_col = st.columns(2)
    dl_col.download_button(
        "⬇️ Download CSV", hist.to_csv(index=False),
        "prediction_history.csv", "text/csv", width="stretch",
    )
    if clr_col.button("🗑️ Clear History", width="stretch"):
        clear_predictions()
        st.rerun()


with st.sidebar:
    st.header("Forecast setup")
    with st.form("forecast_setup"):
        ticker_input = st.text_input(
            "Ticker", "AAPL", help="Examples: AAPL, RELIANCE.NS, BTC-USD"
        )
        target_date = st.date_input("Target date", pd.Timestamp.today())
        use_feedback = st.checkbox(
            "Use past-error correction", value=True,
            help="Applied only after at least three forecasts for this ticker have resolved.",
        )
        run_prediction = st.form_submit_button(
            "Generate forecast", type="primary", width="stretch"
        )

    with st.expander("Ticker examples"):
        st.caption("US")
        st.write("AAPL · MSFT · NVDA · TSLA · AMZN")
        st.caption("India (NSE)")
        st.write("RELIANCE.NS · TCS.NS · INFY.NS · HDFCBANK.NS")

    st.caption("Educational model only — not financial advice.")

st.title("Stock Forecast")
st.caption("A focused LSTM forecast with leakage-safe backtesting and saved accuracy history.")

if "page_view" not in st.session_state:
    st.session_state["page_view"] = "Forecast"
if run_prediction:
    st.session_state["page_view"] = "Forecast"
view = st.segmented_control(
    "Page", ["Forecast", "History"], label_visibility="collapsed", key="page_view",
)
if view == "History":
    render_history()
    st.stop()

if not run_prediction:
    st.info("Choose a ticker and target date in the sidebar, then generate a forecast.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Choose**\n\nEnter a Yahoo Finance ticker.")
    c2.markdown("**2 · Forecast**\n\nThe model trains on six years of data.")
    c3.markdown("**3 · Review**\n\nCompare saved forecasts as dates resolve.")
else:
    try:
        ticker = normalize_ticker(ticker_input)
    except ValueError as exc:
        st.error(f"❌ {exc}")
        st.stop()

    with st.spinner(f"Loading {ticker} market data..."):
        try:
            df = load_data(ticker)
            if df.empty:
                raise ValueError(f"Could not find data for '{ticker}'. Please verify the symbol.")
            feat, clean_data = preprocess_data(df)
        except Exception as exc:
            st.error(f"❌ Unable to load usable market data: {exc}")
            st.stop()

    requested_target_date = pd.to_datetime(target_date).normalize()
    target_date_pd = requested_target_date
    if requested_target_date.weekday() >= 5:
        target_date_pd = pd.offsets.BDay().rollback(requested_target_date)

    # For known historical holidays, use the most recent real trading session.
    # Future exchange-specific holidays cannot be known from the downloaded price
    # series, but weekends are handled above.
    if target_date_pd <= feat.index[-1] and target_date_pd not in feat.index:
        earlier_sessions = feat.index[feat.index <= target_date_pd]
        if len(earlier_sessions) == 0:
            st.error("❌ The selected date is earlier than the available market history.")
            st.stop()
        target_date_pd = earlier_sessions[-1]

    if target_date_pd != requested_target_date:
        st.info(
            f"Markets were closed on {requested_target_date.strftime('%d %b %Y')}. "
            f"Using the previous available session: {target_date_pd.strftime('%d %b %Y')}."
        )
    time_step = 60
    # Only the model-input features must be non-NaN. We deliberately do NOT drop
    # on "target_ret_next" (NaN on the most recent row), so the forecast uses data
    # right up to the latest trading day instead of stopping a few days short.
    feature_cols = FEATURE_COLUMNS

    if target_date_pd <= feat.index[-1]:
        past_data = feat[feat.index < target_date_pd].dropna(subset=feature_cols)
        dates_to_predict = pd.DatetimeIndex([target_date_pd])
    else:
        past_data = feat.dropna(subset=feature_cols)
        dates_to_predict = pd.bdate_range(start=past_data.index[-1] + pd.Timedelta(days=1), end=target_date_pd)

    # Safety net: if the target lands with no business day to forecast (e.g. a
    # weekend immediately after the last trading day), still predict that date.
    if len(dates_to_predict) == 0:
        dates_to_predict = pd.DatetimeIndex([target_date_pd])

    if len(dates_to_predict) > MAX_FORECAST_STEPS:
        st.error(
            f"❌ This horizon needs {len(dates_to_predict)} recursive steps. "
            f"Choose a date within {MAX_FORECAST_STEPS} business days; longer paths are not reliable."
        )
        st.stop()

    if len(past_data) < time_step:
        st.warning("⚠️ Need at least 60 trading days of historical data.")
    else:
        # A historical backtest must not train on the target or on anything after
        # it. The final row is also excluded because its label is the target-day
        # return, which would leak the answer into training.
        if target_date_pd <= feat.index[-1]:
            training_data = clean_data[clean_data.index < past_data.index[-1]]
        else:
            training_data = clean_data

        if len(training_data) < time_step + 20:
            st.error("❌ Not enough pre-target history to train and validate the model.")
            st.stop()

        training_fingerprint = data_fingerprint(training_data)

        # Feedback loop: learn the model's systematic per-day error from resolved
        # predictions and add it back below. When feedback is on we recompute
        # (bypass the SQLite read) so the forecast reflects the latest learning;
        # the LSTM itself stays cached, so this stays cheap.
        bias_per_day, n_fb = 0.0, 0
        if use_feedback:
            bias_per_day, n_fb = compute_feedback_bias(ticker, exclude_target=str(target_date_pd.date()))

        prediction_version = f"{MODEL_VERSION}-feedback" if use_feedback else MODEL_VERSION
        cached = None if use_feedback else get_cached_prediction(
            ticker, target_date_pd.date(), prediction_version
        )
        if cached is not None and cached["data_fingerprint"] != training_fingerprint:
            cached = None  # market history changed; replace the stale saved result

        if cached is not None:
            # Cache hit — reuse the stored prediction, skip training entirely.
            pred_prices_path = cached["pred_prices"]
            pred_dates_path = cached["pred_dates"]
            st.success(f"⚡ Loaded from cache (computed {cached['generated_at']}) — no retraining needed.")
        else:
            # Live status panel so long runs never look "hung".
            with st.status(f"🔧 Calculating forecast for {ticker}…", expanded=True) as status:
                # --- Phase 1: training (spinner only — no epoch spam) ---
                status.update(label="🧠 Calculating… training the model (please wait)")
                try:
                    model, scaler, y_mean, y_std = train_model(
                        training_fingerprint, training_data
                    )
                except Exception as exc:
                    status.update(label="Forecast failed", state="error", expanded=True)
                    st.error(f"❌ Model training failed: {exc}")
                    st.stop()

                # --- Phase 2: recursive forecast (per-step progress) ---
                n_steps = len(dates_to_predict)
                status.update(label=f"📈 Calculating… forecasting {n_steps} step(s) ahead")
                fc_bar = st.progress(0, text="Calculating…")

                current_window = past_data[["ret1", "vol_change", "sma_10_dist", "volatility_10"]].tail(time_step).values.astype(np.float32)
                last_price = past_data["Close"].iloc[-1]

                recent_prices = list(past_data["Close"].tail(10).values)
                recent_returns = list(past_data["ret1"].tail(10).values)

                pred_prices_path = []
                pred_dates_path = []

                for i, d in enumerate(dates_to_predict):
                    X_scaled = scaler.transform(current_window).reshape(1, time_step, 4)
                    raw_pred_ret = (
                        model.predict(X_scaled, verbose=0)[0][0] * y_std
                    ) + y_mean

                    # Apply feedback in compoundable log-return space and bound
                    # pathological outputs before recursively feeding them back.
                    pred_ret = apply_log_bias(
                        raw_pred_ret, bias_per_day, MAX_DAILY_RETURN
                    )

                    # Future volume is unknown. A neutral volume-change feature is
                    # deterministic and avoids presenting random noise as model
                    # intelligence; the price path is now reproducible.
                    sim_vol_change = 0.0

                    last_price = last_price * (1 + pred_ret)
                    pred_prices_path.append(last_price)
                    pred_dates_path.append(d)

                    recent_prices.append(last_price)
                    recent_prices.pop(0)
                    recent_returns.append(pred_ret)
                    recent_returns.pop(0)

                    new_row = np.array([[
                        pred_ret,
                        sim_vol_change,
                        (last_price - np.mean(recent_prices)) / np.mean(recent_prices),
                        np.std(recent_returns, ddof=1)
                    ]], dtype=np.float32)

                    current_window = np.vstack([current_window[1:], new_row])

                    fc_bar.progress(
                        int((i + 1) / n_steps * 100),
                        text=f"Calculating… step {i + 1}/{n_steps} ({d.strftime('%d %b %Y')})",
                    )

                save_prediction(
                    ticker, target_date_pd.date(),
                    pred_prices_path[-1], past_data["Close"].iloc[-1],
                    pred_dates_path, pred_prices_path,
                    prediction_version,
                    training_fingerprint,
                )
                status.update(label=f"✅ Forecast complete for {ticker}", state="complete", expanded=False)

        last_actual_date = past_data.index[-1]
        # Use the baseline close the prediction was actually made against, so a
        # cache hit's "Expected Change %" stays consistent with the stored path.
        last_actual_close = cached["last_close"] if cached is not None else past_data["Close"].iloc[-1]
        final_predicted_price = pred_prices_path[-1]
        latest_volume = format_volume(past_data["Volume"].iloc[-1])
        currency = display_currency(ticker)

        col1, col2, col3 = st.columns(3)
        
        col1.metric("Last Actual Close", f"{currency}{last_actual_close:.2f}", last_actual_date.strftime('%d %b %Y'))
        col1.caption(f"Volume: {latest_volume}")
        
        total_expected_return = ((final_predicted_price - last_actual_close) / last_actual_close) * 100
        col2.metric("Predicted Output", f"{currency}{final_predicted_price:.2f}", f"{total_expected_return:.2f}% Expected Change")
        
        actual_price = None
        if target_date_pd in feat.index:
            actual_price = feat.loc[target_date_pd, "Close"]
            error_pct = abs(final_predicted_price - actual_price) / actual_price * 100
            real_return = ((actual_price - last_actual_close) / last_actual_close) * 100
            col3.metric("Actual Reality", f"{currency}{actual_price:.2f}", f"{real_return:.2f}% Real Change", delta_color="normal")
            col3.caption(f"Absolute error: {error_pct:.2f}%")
        else:
            col3.metric("Actual Reality", "N/A", f"Step {len(dates_to_predict)} into Future", delta_color="off")
            col3.caption("Waiting for the market close")

        if use_feedback:
            if n_fb >= MIN_FEEDBACK_SAMPLES:
                st.caption(f"🔁 Feedback correction applied: {bias_per_day * 100:+.3f}%/day, learned from {n_fb} resolved prediction(s).")
            else:
                st.caption(f"🔁 Feedback ON — need ≥{MIN_FEEDBACK_SAMPLES} resolved predictions to start correcting (have {n_fb}).")

        st.markdown("---")

        plot_data = feat["Close"].loc[past_data.index[-25:]]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data.values, mode='lines+markers', name='Past Prices', line=dict(color='#1E88E5', width=2)))
        
        path_dates = [last_actual_date] + list(pred_dates_path)
        path_prices = [last_actual_close] + pred_prices_path
        
        fig.add_trace(go.Scatter(
            x=path_dates, y=path_prices, 
            mode='lines+markers', name='LSTM Prediction Path',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=8, symbol='circle')
        ))
        
        if actual_price is not None and pd.notna(actual_price):
            fig.add_trace(go.Scatter(
                x=[last_actual_date, target_date_pd], y=[last_actual_close, actual_price], 
                mode='lines+markers', name='Actual Path',
                line=dict(color='green', width=3),
                marker=dict(size=8, symbol='circle')
            ))

        fig.update_layout(hovermode="x unified", title=f"Recursive Trajectory ({len(dates_to_predict)} Steps Ahead)", yaxis_title="Price", plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig, width="stretch")
