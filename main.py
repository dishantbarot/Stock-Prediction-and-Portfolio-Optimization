import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import BDay

# =========================
# Load Model and Scaler
# =========================
@st.cache_resource
def load_model_and_scaler():
    with open("gradient_boosting_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# =========================
# Data Download & Processing
# =========================
def download_stock_data(ticker):
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None
    return data

def preprocess_and_engineer_features(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df['Returns'] = df['Close'].pct_change()

    lags = [1, 5, 14, 21, 50, 100, 200]
    for lag in lags:
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()
    df['rolling_mean_50'] = df['Close'].rolling(window=50).mean()

    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    df.dropna(inplace=True)
    return df

# =========================
# Plot SMA and EMA + Crossovers
# =========================
def plot_smas_emas(df, ticker):
    for ma in [21, 100, 200]:
        df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'EMA_{ma}'] = df['Close'].ewm(span=ma, adjust=False).mean()

    # Closing price with SMA
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)
    ax.plot(df.index, df['SMA_21'], label='SMA 21', color='blue')
    ax.plot(df.index, df['SMA_100'], label='SMA 100', color='green')
    ax.plot(df.index, df['SMA_200'], label='SMA 200', color='red')
    ax.set_title(f"{ticker} - Close with SMA 21/100/200")
    ax.legend()
    st.pyplot(fig)

    # Crossovers
    cross_pairs = [
        ('SMA', 21, 100), ('SMA', 100, 200),
        ('EMA', 21, 100), ('EMA', 100, 200)
    ]
    for ma_type, short, long in cross_pairs:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df['Close'], label='Close', color='grey', alpha=0.6)
        ax.plot(df.index, df[f'{ma_type}_{short}'], label=f'{ma_type} {short}', color='orange')
        ax.plot(df.index, df[f'{ma_type}_{long}'], label=f'{ma_type} {long}', color='purple')
        ax.set_title(f"{ticker} - {ma_type} {short} vs {ma_type} {long} Crossover")
        ax.legend()
        st.pyplot(fig)

# =========================
# Predict Tomorrow’s Price
# =========================
def predict_tomorrow(df, model, scaler):
    features = [col for col in df.columns if col not in ['Close', 'Adj Close', 'Returns']]
    latest = df.tail(1)[features]
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]

    # Next business day
    tomorrow_date = df.index[-1] + BDay(1)
    return pred, tomorrow_date

# =========================
# Benchmark Comparison
# =========================
def compare_with_nifty(stock_df, stock_ticker):
    nifty = '^NSEI'
    nifty_df = download_stock_data(nifty)
    if nifty_df is None:
        st.warning("Could not download Nifty 50 data.")
        return

    stock_df['Returns'] = stock_df['Close'].pct_change()
    nifty_df['Returns'] = nifty_df['Close'].pct_change()
    stock_df['Cumulative Returns'] = (1 + stock_df['Returns']).cumprod() - 1
    nifty_df['Cumulative Returns'] = (1 + nifty_df['Returns']).cumprod() - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(stock_df.index, stock_df['Cumulative Returns'], label=f'{stock_ticker} Cumulative Returns', color='blue')
    ax.plot(nifty_df.index, nifty_df['Cumulative Returns'], label='Nifty 50 Cumulative Returns', color='orange')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f"Cumulative Returns: {stock_ticker} vs. Nifty 50")
    ax.legend()
    st.pyplot(fig)

    stock_final = stock_df['Cumulative Returns'].iloc[-1]
    nifty_final = nifty_df['Cumulative Returns'].iloc[-1]

    st.markdown("### Investment Recommendation")
    st.write(f"{stock_ticker} Return: **{stock_final:.2%}**")
    st.write(f"Nifty 50 Return: **{nifty_final:.2%}**")
    if stock_final > nifty_final:
        st.success(f"**{stock_ticker} outperformed Nifty 50 → Consider BUY.**")
    else:
        st.info(f"**Nifty 50 outperformed {stock_ticker} → Consider avoiding.**")

# =========================
# Streamlit App Layout
# =========================
st.title("📈 Stock Prediction and Benchmark App")
st.write("Enter an NSE stock ticker (e.g., RELIANCE.NS, SBIN.NS). "
         "The app will plot SMAs/EMAs & crossovers, predict tomorrow’s close, "
         "and compare performance vs Nifty 50.")

ticker = st.text_input("Enter Stock Ticker", value="RELIANCE.NS")

if st.button("Analyze"):
    df = download_stock_data(ticker)
    if df is None or df.empty:
        st.error("Could not download data for the entered ticker.")
    else:
        df_pre = preprocess_and_engineer_features(df.copy())

        st.subheader("📊 SMAs, EMAs and Crossovers")
        plot_smas_emas(df_pre.copy(), ticker)

        st.subheader("🔮 Tomorrow’s Forecast")
        pred, tomorrow_date = predict_tomorrow(df_pre, model, scaler)
        st.metric(label=f"Predicted Close for {tomorrow_date.date()}", value=f"{pred:.2f}")

        st.subheader("📌 Compare with Nifty 50")
        compare_with_nifty(df_pre.copy(), ticker)
