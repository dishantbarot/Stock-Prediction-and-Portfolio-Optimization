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
    """
    Loads the pre-trained model and scaler from pkl files.
    
    NOTE: As the model and scaler files are not available, this function
    is providing a placeholder to make the app runnable.
    """
    try:
        with open("gradient_boosting_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        st.success("Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError:
        st.warning("Model or scaler files not found. Using dummy placeholders.")
        
        # --- Placeholder Model and Scaler for demonstration ---
        # A simple linear model for demonstration
        class DummyModel:
            def predict(self, X):
                return np.array([X.mean() * 1.05]) # A simple mock prediction

        # A simple scaler for demonstration
        class DummyScaler:
            def transform(self, X):
                return X.values / X.values.mean(axis=0)

        return DummyModel(), DummyScaler()


model, scaler = load_model_and_scaler()

# =========================
# Data Download & Processing
# =========================
def download_stock_data(ticker):
    """Downloads historical stock data for a given ticker."""
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None
    return data

def preprocess_and_engineer_features(df):
    """Preprocesses data and engineers features for the model."""
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    df['Returns'] = df['Close'].pct_change()
    
    lags = [1, 5, 14, 21, 50, 100, 200]
    for lag in lags:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    
    # Add features from the original notebook
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()
    df['rolling_mean_50'] = df['Close'].rolling(window=50).mean()

    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    # FIX: Added the missing dayofweek feature
    df['dayofweek'] = df.index.dayofweek

    # Drop rows with NaN values after feature creation
    df.dropna(inplace=True)
    return df

# =========================
# Plot SMA and EMA + Crossovers
# =========================
def plot_smas_emas(df, ticker):
    """Generates and plots SMAs, EMAs, and their crossovers."""
    
    # Plot SMA Crossovers
    st.subheader("SMA Crossovers 21 vs 50 ")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)
    ax.plot(df.index, df['SMA_21'], label='SMA 21', color='blue')
    ax.plot(df.index, df['SMA_50'], label='SMA 50', color='orange')
    ax.set_title(f"{ticker} - SMA 21 vs SMA 50 Crossover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("SMA Crossovers 100 vs 200 ")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)
    ax.plot(df.index, df['SMA_100'], label='SMA 100', color='green')
    ax.plot(df.index, df['SMA_200'], label='SMA 200', color='red')
    ax.set_title(f"{ticker} - SMA 100 vs SMA 200 Crossover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Plot EMA Crossovers
    st.subheader("EMA Crossovers 21 vs 50 ")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)
    ax.plot(df.index, df['EMA_21'], label='EMA 21', color='blue')
    ax.plot(df.index, df['EMA_50'], label='EMA 50', color='orange')
    ax.set_title(f"{ticker} - EMA 21 vs EMA 50 Crossover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("EMA Crossovers 100 vs 200 ")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)
    ax.plot(df.index, df['EMA_100'], label='EMA 100', color='green')
    ax.plot(df.index, df['EMA_200'], label='EMA 200', color='red')
    ax.set_title(f"{ticker} - EMA 100 vs EMA 200 Crossover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)


# =========================
# Predict Tomorrow’s Price
# =========================
# Features used during training
TRAIN_FEATURES = [
    'lag_1', 'lag_5', 'lag_14', 'lag_21',
    'lag_50', 'lag_100', 'lag_200',
    'SMA_21', 'SMA_100', 'SMA_200',
    'EMA_21', 'EMA_100', 'EMA_200',
    'rolling_mean_20', 'rolling_std_20',
    'day', 'month', 'year', 'weekofyear', 'dayofweek'
]

def predict_tomorrow(df):
    """
    Predicts the close price for the next business day and returns the prediction and date.
    """
    if df.empty:
        return None, None
        
    latest_data = df.tail(1)[TRAIN_FEATURES]
    
    # Scale the latest data using the pre-loaded scaler
    latest_scaled = scaler.transform(latest_data)
    
    # Predict the price using the pre-loaded model
    prediction = model.predict(latest_scaled)[0]
    
    # Calculate the next business day
    last_date = df.index[-1]
    tomorrow_date = last_date + BDay(1)
    
    return prediction, tomorrow_date

def plot_forecast_and_history(df, pred, tomorrow_date, ticker):
    """Plots the historical close prices and the predicted price for tomorrow."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Historical Close', color='blue', linewidth=2)
    
    # Plot the predicted point
    ax.scatter(tomorrow_date, pred, color='red', s=100, zorder=5, label='Predicted Close')
    
    ax.set_title(f"{ticker} - Historical Close and Tomorrow's Predicted Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# =========================
# Benchmark Comparison
# =========================
def compare_with_nifty(stock_df, stock_ticker):
    """Compares the stock's cumulative returns with the Nifty 50 benchmark."""
    nifty = '^NSEI'
    nifty_df = download_stock_data(nifty)
    if nifty_df is None:
        st.warning("Could not download Nifty 50 data.")
        return

    # Ensure both dataframes cover the same time period
    start_date = max(stock_df.index.min(), nifty_df.index.min())
    end_date = min(stock_df.index.max(), nifty_df.index.max())

    stock_df_aligned = stock_df.loc[start_date:end_date]
    nifty_df_aligned = nifty_df.loc[start_date:end_date]

    # Calculate returns and cumulative returns
    stock_df_aligned['Returns'] = stock_df_aligned['Close'].pct_change()
    nifty_df_aligned['Returns'] = nifty_df_aligned['Close'].pct_change()

    stock_df_aligned['Cumulative Returns'] = (1 + stock_df_aligned['Returns']).cumprod() - 1
    nifty_df_aligned['Cumulative Returns'] = (1 + nifty_df_aligned['Returns']).cumprod() - 1

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(stock_df_aligned.index, stock_df_aligned['Cumulative Returns'], label=f'{stock_ticker} Cumulative Returns', color='blue')
    ax.plot(nifty_df_aligned.index, nifty_df_aligned['Cumulative Returns'], label='Nifty 50 Cumulative Returns', color='orange')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_title(f"Cumulative Returns: {stock_ticker} vs. Nifty 50")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    st.pyplot(fig)

    stock_final = stock_df_aligned['Cumulative Returns'].iloc[-1]
    nifty_final = nifty_df_aligned['Cumulative Returns'].iloc[-1]

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
st.write(f"This app analyzes historical data from **2015-01-01** to **{datetime.today().strftime('%Y-%m-%d')}**.")
st.write("Enter an NSE stock ticker (e.g., RELIANCE.NS, SBIN.NS). "
         "The app will plot SMAs/EMAs & crossovers, predict tomorrow’s close, "
         "and compare performance vs Nifty 50.")

ticker = st.text_input("Enter Stock Ticker", value="RELIANCE.NS")

if st.button("Analyze"):
    if not model or not scaler:
        st.error("Model or scaler is not loaded. Please check file paths.")
        st.stop()

    df = download_stock_data(ticker)
    if df is None or df.empty:
        st.error("Could not download data for the entered ticker. Please check the ticker symbol.")
    else:
        df_pre = preprocess_and_engineer_features(df.copy())
        
        st.subheader("📊 SMAs, EMAs and Crossovers")
        plot_smas_emas(df_pre.copy(), ticker)

        st.subheader("🔮 Tomorrow’s Forecast")
        pred, tomorrow_date = predict_tomorrow(df_pre)
        st.metric(label=f"Predicted Close for {tomorrow_date.date()}", value=f"{pred:.2f}")
        plot_forecast_and_history(df_pre, pred, tomorrow_date, ticker)

        st.subheader("📌 Compare with Nifty 50")
        compare_with_nifty(df.copy(), ticker)
