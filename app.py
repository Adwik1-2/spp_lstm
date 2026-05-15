import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

st.set_page_config(page_title="LSTM Stock Predictor", page_icon="📈", layout="wide")

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
    model.fit(x_tr, y_tr_s, epochs=15, batch_size=32, validation_split=0.1, verbose=0)
    
    return model, scaler.fit(X_raw[:split]), y_mean, y_std

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933116.png", width=70)
    st.markdown("### ⚙️ Engine Configuration")
    ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    target_date = st.date_input("Target Date", pd.Timestamp.today())
    
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
    with st.spinner(f"Analyzing {ticker} Market Data & Training LSTM..."):
        df = load_data(ticker)
        if df.empty:
            st.error(f"❌ Could not find data for '{ticker}'. Please verify the symbol.")
            st.stop()
            
        feat, clean_data = preprocess_data(df)
        model, scaler, y_mean, y_std = train_model(clean_data)

    target_date_pd = pd.to_datetime(target_date)
    time_step = 60

    if target_date_pd <= feat.index[-1]:
        past_data = feat[feat.index < target_date_pd].dropna()
        dates_to_predict = [target_date_pd]
    else:
        past_data = feat.dropna()
        dates_to_predict = pd.bdate_range(start=past_data.index[-1] + pd.Timedelta(days=1), end=target_date_pd)

    if len(past_data) < time_step:
        st.warning("⚠️ Need at least 60 trading days of historical data.")
    else:
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

        last_actual_date = past_data.index[-1]
        last_actual_close = past_data["Close"].iloc[-1]
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