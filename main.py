import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load trained model and scaler
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

# Feature engineering function
def preprocess_and_engineer_features(df):
    df['lag_1'] = df['Close'].shift(1)
    df['lag_5'] = df['Close'].shift(5)
    df['lag_14'] = df['Close'].shift(14)
    df['lag_21'] = df['Close'].shift(21)
    df['lag_50'] = df['Close'].shift(50)
    df['lag_100'] = df['Close'].shift(100)
    df['lag_200'] = df['Close'].shift(200)

    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()

    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week
    df['dayofweek'] = df.index.dayofweek

    df = df.dropna()
    return df

# Prediction function (fixed)
def predict_tomorrow(df):
    latest = df.tail(1)[TRAIN_FEATURES]   # ✅ only use training features
    latest_scaled = scaler.transform(latest)
    return model.predict(latest_scaled)[0]

# Streamlit UI
st.title("📈 Stock Price Forecasting & SMA Crossover App")

ticker = st.text_input("Enter Stock Ticker:", "RELIANCE.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Run Analysis"):
    # Fetch stock data
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error("⚠️ No data found for the given ticker and date range.")
    else:
        df = preprocess_and_engineer_features(df)

        # Show stock chart with SMA crossovers
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label="Stock Closing Price")
        plt.plot(df['SMA_21'], label="21-day SMA")
        plt.plot(df['SMA_50'], label="50-day SMA")
        plt.title(f"{ticker} Closing Price with 21 & 50 SMA Crossovers")
        plt.legend()
        st.pyplot(plt)

        # Predict tomorrow's price
        forecast = predict_tomorrow(df)
        st.subheader(f"🔮 Tomorrow's Forecasted Price: {forecast:.2f} INR")

        # Plot forecast point
        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label="Stock Closing Price")
        plt.scatter(df.index[-1] + pd.Timedelta(days=1), forecast, color="red", label="Forecasted Price")
        plt.title(f"{ticker} with Tomorrow's Forecast")
        plt.legend()
        st.pyplot(plt)

        # Compare with NIFTY 50
        try:
            nifty = yf.download("^NSEI", start=start_date, end=end_date)
            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'] / df['Close'].iloc[0], label=f"{ticker} Normalized")
            plt.plot(nifty['Close'] / nifty['Close'].iloc[0], label="NIFTY 50 Normalized")
            plt.title(f"{ticker} vs NIFTY 50")
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"⚠️ Error fetching NIFTY 50: {e}")
