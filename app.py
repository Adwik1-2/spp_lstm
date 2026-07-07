import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import sqlite3
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Prediction cache (SQLite)
# ---------------------------------------------------------------------------
# Persists each prediction so a repeat run for the same ticker + target date
# returns the stored result instead of retraining the LSTM.
# Cache key = (ticker, target_date, MODEL_VERSION). Bump MODEL_VERSION whenever
# the model / features change so old rows are treated as stale automatically.
DB_PATH = "predictions.db"
MODEL_VERSION = "v2"          # bumped: new training regime → invalidate old cache
EPOCHS = 50                   # max epochs; EarlyStopping usually halts sooner
EARLY_STOP_PATIENCE = 6       # stop if val_loss doesn't improve for N epochs
MIN_FEEDBACK_SAMPLES = 3      # min resolved predictions before applying correction
MAX_DAILY_BIAS = 0.02         # cap the feedback correction at ±2%/day (robustness)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
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
            PRIMARY KEY (ticker, target_date, model_version)
        )
    """)
    return conn


def get_cached_prediction(ticker, target_date):
    conn = get_conn()
    row = conn.execute(
        """SELECT predicted_price, last_close, pred_dates, pred_prices, generated_at
           FROM predictions WHERE ticker=? AND target_date=? AND model_version=?""",
        (ticker, str(target_date), MODEL_VERSION),
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
    }


def save_prediction(ticker, target_date, predicted_price, last_close, pred_dates, pred_prices):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?)",
        (
            ticker,
            str(target_date),
            MODEL_VERSION,
            float(predicted_price),
            float(last_close),
            json.dumps([pd.to_datetime(d).strftime("%Y-%m-%d") for d in pred_dates]),
            json.dumps([float(p) for p in pred_prices]),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def get_all_predictions():
    conn = get_conn()
    rows = conn.execute(
        """SELECT ticker, target_date, predicted_price, last_close, generated_at
           FROM predictions WHERE model_version=? ORDER BY generated_at DESC""",
        (MODEL_VERSION,),
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
           FROM predictions WHERE ticker=? AND model_version=?""",
        (ticker, MODEL_VERSION),
    ).fetchall()
    conn.close()
    return rows

st.markdown("""
    <style>
    .main-title { font-size: 3.8rem !important; font-weight: 800; color: #1E88E5; margin-bottom: 0px; margin-top: -20px;}
    .sub-title { font-size: 1.4rem !important; color: #8fa0b3; margin-bottom: 30px; font-weight: 500;}
    .error-text { color: #ff4b4b; font-weight: bold; font-size: 1.1rem; }
    .vol-text { color: #a3a8b8; font-size: 0.95rem; font-weight: 500; }
    .stButton>button { font-weight: bold; font-size: 1.1rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(ticker):
    df = yf.download(ticker, period="6y", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    feat = df[["Close", "Volume"]].copy()
    feat["ret1"] = feat["Close"].pct_change()
    feat["vol_change"] = feat["Volume"].pct_change()
    feat["sma_10_dist"] = (feat["Close"] - feat["Close"].rolling(10).mean()) / feat["Close"].rolling(10).mean()
    feat["volatility_10"] = feat["ret1"].rolling(10).std()
    feat["target_ret_next"] = feat["ret1"].shift(-1)
    data = feat.replace([np.inf, -np.inf], np.nan).dropna()
    return feat, data

@st.cache_resource(show_spinner=False)
def train_model(_data, time_step=60):
    x_df = _data[["ret1", "vol_change", "sma_10_dist", "volatility_10"]]
    y_raw = _data["target_ret_next"].values.astype(np.float32)
    X_raw = x_df.values.astype(np.float32)

    split = int(len(X_raw) * 0.9)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_raw[:split]).astype(np.float32)

    xs, ys = [], []
    for i in range(time_step, len(X_train)):
        xs.append(X_train[i-time_step:i, :])
        ys.append(y_raw[:split][i])
        
    x_tr = np.array(xs, dtype=np.float32)
    y_tr = np.array(ys, dtype=np.float32)

    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    y_tr_s = (y_tr - y_mean) / y_std

    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, x_tr.shape[2])),
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
    model.fit(x_tr, y_tr_s, epochs=EPOCHS, batch_size=32, validation_split=0.1, verbose=0, callbacks=[early_stop])

    return model, scaler.fit(X_raw[:split]), y_mean, y_std


def fetch_actual_close(ticker, date_str):
    """Return the real market close for a stored prediction's target date, or
    None if that date hasn't traded yet (a future forecast) / has no data."""
    try:
        df = load_data(ticker)
    except Exception:
        return None
    if df is None or df.empty or "Close" not in df.columns:
        return None
    ts = pd.to_datetime(date_str)
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
        predicted_total_ret = predicted_price / last_close - 1
        actual_total_ret = actual / last_close - 1
        per_day_error = (actual_total_ret - predicted_total_ret) / n_steps
        # clip each sample so one wild backtest can't dominate the average
        errors.append(max(-MAX_DAILY_BIAS, min(MAX_DAILY_BIAS, per_day_error)))

    if len(errors) < MIN_FEEDBACK_SAMPLES:
        return 0.0, len(errors)

    bias = float(np.mean(errors))
    bias = max(-MAX_DAILY_BIAS, min(MAX_DAILY_BIAS, bias))
    return bias, len(errors)


def render_history():
    """History & accuracy tab — reads every saved prediction from SQLite and
    scores it against the real close once that date has traded."""
    st.markdown("### 📜 Saved Predictions & Model Accuracy")
    st.caption(
        "Every forecast is persisted to a local SQLite database (`predictions.db`). "
        "Re-running the same ticker + date loads the stored result instead of retraining."
    )

    rows = get_all_predictions()
    if not rows:
        st.info("No predictions stored yet — run a forecast in the **Predictor** tab first.")
        return

    records = []
    for ticker, tdate, pred, last_close, gen_at in rows:
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

    st.dataframe(hist, hide_index=True, use_container_width=True)

    dl_col, clr_col = st.columns(2)
    dl_col.download_button(
        "⬇️ Download CSV", hist.to_csv(index=False),
        "prediction_history.csv", "text/csv", use_container_width=True,
    )
    if clr_col.button("🗑️ Clear History", use_container_width=True):
        clear_predictions()
        st.rerun()


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=70)
    st.markdown("### ⚙️ Engine Configuration")
    ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    target_date = st.date_input("Target Date", pd.Timestamp.today())
    use_feedback = st.checkbox(
        "🔁 Learn from past errors (feedback)", value=True,
        help="Corrects future forecasts using the model's average error on resolved predictions.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    run_prediction = st.button("🚀 Predict Stock Price", type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("📊 Market Ticker Directory"):
        st.markdown("**🇺🇸 US Equity Markets**")
        st.caption("Technology")
        st.code("AAPL, MSFT, NVDA, GOOGL, META, AMD, INTC", language="text")
        st.caption("Automotive / EV")
        st.code("TSLA, TM, F, GM", language="text")
        st.caption("Financials")
        st.code("JPM, V, MA, BAC, GS", language="text")
        st.caption("Consumer & Healthcare")
        st.code("AMZN, WMT, COST, JNJ, LLY", language="text")
        
        st.divider()
        
        st.markdown("**🇮🇳 Indian Equity Markets (NSE)**")
        st.caption("Append '.NS' to the symbol for Yahoo Finance")
        
        in_stocks = pd.DataFrame({
            "Sector": ["Energy/Retail", "Banking", "Banking", "Banking", "IT Services", "IT Services", "FMCG", "FMCG", "Automotive", "Automotive", "Infrastructure", "Telecom"],
            "Symbol": ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS", "TATAMOTORS.NS", "M&M.NS", "LT.NS", "BHARTIARTL.NS"]
        })
        st.dataframe(in_stocks, hide_index=True, use_container_width=True)

st.markdown('<p class="main-title">LSTM Stock Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Recursive Multi-Step Deep Learning Forecasting</p>', unsafe_allow_html=True)

if not run_prediction:
    st.info("👈 Please configure the Engine Parameters in the sidebar and click **Predict Stock Price** to generate the forecast.")
else:
    with st.spinner(f"Loading {ticker} market data..."):
        df = load_data(ticker)
        if df.empty:
            st.error(f"❌ Could not find data for '{ticker}'. Please verify the symbol.")
            st.stop()

        feat, clean_data = preprocess_data(df)

    target_date_pd = pd.to_datetime(target_date)
    time_step = 60
    # Only the model-input features must be non-NaN. We deliberately do NOT drop
    # on "target_ret_next" (NaN on the most recent row), so the forecast uses data
    # right up to the latest trading day instead of stopping a few days short.
    feature_cols = ["ret1", "vol_change", "sma_10_dist", "volatility_10"]

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

    if len(past_data) < time_step:
        st.warning("⚠️ Need at least 60 trading days of historical data.")
    else:
        # Feedback loop: learn the model's systematic per-day error from resolved
        # predictions and add it back below. When feedback is on we recompute
        # (bypass the SQLite read) so the forecast reflects the latest learning;
        # the LSTM itself stays cached, so this stays cheap.
        bias_per_day, n_fb = 0.0, 0
        if use_feedback:
            bias_per_day, n_fb = compute_feedback_bias(ticker, exclude_target=str(target_date))

        cached = None if use_feedback else get_cached_prediction(ticker, target_date)

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
                model, scaler, y_mean, y_std = train_model(clean_data)

                # --- Phase 2: recursive forecast (per-step progress) ---
                n_steps = len(dates_to_predict)
                status.update(label=f"📈 Calculating… forecasting {n_steps} step(s) ahead")
                fc_bar = st.progress(0, text="Calculating…")

                current_window = past_data[["ret1", "vol_change", "sma_10_dist", "volatility_10"]].tail(time_step).values.astype(np.float32)
                last_price = past_data["Close"].iloc[-1]

                recent_prices = list(past_data["Close"].tail(10).values)
                recent_returns = list(past_data["ret1"].tail(10).values)

                hist_ret_std = past_data["ret1"].std()
                hist_vol_std = past_data["vol_change"].std()

                pred_prices_path = []
                pred_dates_path = []

                for i, d in enumerate(dates_to_predict):
                    X_scaled = scaler.transform(current_window).reshape(1, time_step, 4)
                    pred_ret = (model.predict(X_scaled, verbose=0)[0][0] * y_std) + y_mean

                    # Feedback loop: nudge each day's return by the learned bias.
                    pred_ret += bias_per_day

                    if i > 0:
                        market_noise = np.random.normal(0, hist_ret_std * 0.8)
                        pred_ret += market_noise
                        sim_vol_change = np.random.normal(0, hist_vol_std * 0.5)
                    else:
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
                    ticker, target_date,
                    pred_prices_path[-1], past_data["Close"].iloc[-1],
                    pred_dates_path, pred_prices_path,
                )
                status.update(label=f"✅ Forecast complete for {ticker}", state="complete", expanded=False)

        last_actual_date = past_data.index[-1]
        # Use the baseline close the prediction was actually made against, so a
        # cache hit's "Expected Change %" stays consistent with the stored path.
        last_actual_close = cached["last_close"] if cached is not None else past_data["Close"].iloc[-1]
        final_predicted_price = pred_prices_path[-1]
        vol_in_lakhs = past_data["Volume"].iloc[-1] / 100000 

        col1, col2, col3 = st.columns(3)
        
        col1.metric("Last Actual Close", f"${last_actual_close:.2f}", last_actual_date.strftime('%d %b %Y'))
        col1.markdown(f'<p class="vol-text">📊 Traded Vol: {vol_in_lakhs:.2f} L</p>', unsafe_allow_html=True)
        
        total_expected_return = ((final_predicted_price - last_actual_close) / last_actual_close) * 100
        col2.metric("Predicted Output", f"${final_predicted_price:.2f}", f"{total_expected_return:.2f}% Expected Change")
        
        actual_price = None
        if target_date_pd in feat.index:
            actual_price = feat.loc[target_date_pd, "Close"]
            error_pct = abs(final_predicted_price - actual_price) / actual_price * 100
            real_return = ((actual_price - last_actual_close) / last_actual_close) * 100
            col3.metric("Actual Reality", f"${actual_price:.2f}", f"{real_return:.2f}% Real Change", delta_color="normal")
            col3.markdown(f'<p class="error-text">🎯 Error Margin: {error_pct:.2f}%</p>', unsafe_allow_html=True)
        else:
            col3.metric("Actual Reality", "N/A", f"Step {len(dates_to_predict)} into Future", delta_color="off")
            col3.markdown('<p class="vol-text">Market close data unavailable</p>', unsafe_allow_html=True)

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
        
        if actual_price:
            fig.add_trace(go.Scatter(
                x=[last_actual_date, target_date_pd], y=[last_actual_close, actual_price], 
                mode='lines+markers', name='Actual Path',
                line=dict(color='green', width=3),
                marker=dict(size=8, symbol='circle')
            ))

        fig.update_layout(hovermode="x unified", title=f"Recursive Trajectory ({len(dates_to_predict)} Steps Ahead)", yaxis_title="Price", plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Persistence showcase: every prediction ever made, scored against reality.
# Always visible so the SQLite cache is inspectable at a glance.
# ---------------------------------------------------------------------------
st.markdown("---")
with st.expander("📜 Prediction History & Model Accuracy", expanded=not run_prediction):
    render_history()
