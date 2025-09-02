import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
from datetime import datetime, timedelta

# ==============================
# Load Model and Scaler
# ==============================
with open("gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Must match features used during training (20 features)
TRAIN_FEATURES = [
    'lag_1', 'lag_5', 'lag_14', 'lag_21',
    'lag_50', 'lag_100', 'lag_200',
    'SMA_21', 'SMA_100', 'SMA_200',
    'EMA_21', 'EMA_100', 'EMA_200',
    'rolling_mean_20', 'rolling_std_20',
    'day', 'month', 'year', 'weekofyear', 'dayofweek'
]

# ==============================
# Feature Engineering
# ==============================
def preprocess_and_engineer_features(df):
    df['Date'] = df.index
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['dayofweek'] = df.index.dayofweek

    # Lags
    for lag in [1, 5, 14, 21, 50, 100, 200]:
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    # Moving averages
    for win in [21, 100, 200]:
        df[f'SMA_{win}'] = df['Close'].rolling(window=win).mean()
        df[f'EMA_{win}'] = df['Close'].ewm(span=win, adjust=False).mean()

    # Rolling stats
    df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()

    df = df.dropna()
    return df

# ==============================
# Prediction
# ==============================
def predict_tomorrow(df):
    latest = df.tail(1)[TRAIN_FEATURES]
    latest_scaled = scaler.transform(latest)
    return model.predict(latest_scaled)[0]

# ==============================
# Streamlit App
# ==============================
st.title("📈 Stock Price Forecasting & SMA Crossovers")

# User input
ticker = st.text_input("Enter stock ticker (e.g., RELIANCE.NS):", "RELIANCE.NS")
start_date = st.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.date_input("End Date", datetime.today())

if st.button("Run Analysis"):
    # Load stock data
    stock = yf.download(ticker, start=start_date, end=end_date)
    if stock.empty:
        st.error("No data found for the selected stock.")
    else:
        df = preprocess_and_engineer_features(stock)

        # ==============================
        # Forecast Tomorrow
        # ==============================
        tomorrow_price = predict_tomorrow(df)
        st.subheader(f"🔮 Predicted Closing Price for Tomorrow: {tomorrow_price:.2f} INR")

        # Plot Tomorrow Forecast
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label="Historical Closing Price", alpha=0.7)
        plt.scatter(df.index[-1] + timedelta(days=1), tomorrow_price,
                    color="red", label="Tomorrow's Forecast", s=100, marker="*")
        plt.title(f"{ticker} Closing Price & Tomorrow's Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        st.pyplot(plt)

        # ==============================
        # SMA Crossovers (21 & 50)
        # ==============================
        df['SMA_21'] = df['Close'].rolling(window=21).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label="Closing Price", alpha=0.7)
        plt.plot(df.index, df['SMA_21'], label="21 SMA", linewidth=1.5)
        plt.plot(df.index, df['SMA_50'], label="50 SMA", linewidth=1.5)
        plt.title(f"{ticker}: Closing Price with 21 & 50 SMA Crossovers")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.legend()
        st.pyplot(plt)

        # ==============================
        # Compare with NIFTY
        # ==============================
        nifty = yf.download("^NSEI", start=start_date, end=end_date)['Close']
        compare_df = pd.DataFrame({
            ticker: df['Close'],
            'NIFTY': nifty
        }).dropna()

        plt.figure(figsize=(10, 5))
        plt.plot(compare_df.index, compare_df[ticker], label=ticker, alpha=0.7)
        plt.plot(compare_df.index, compare_df['NIFTY'], label="NIFTY 50", alpha=0.7)
        plt.title(f"{ticker} vs NIFTY 50")
        plt.xlabel("Date")
        plt.ylabel("Closing Price (INR)")
        plt.legend()
        st.pyplot(plt)
