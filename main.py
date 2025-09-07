import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
from datetime import datetime, timedelta

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    # Load the XGBoost model
    model = xgb.XGBRegressor()
    model.load_model('xgb_model.json')

    # Load the scaler
    with open('scaler (1).pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_and_scaler()

# Function to download stock data
def download_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, group_by="column")
        if data.empty:
            st.error(f"No data found for ticker {ticker}. Please check the symbol and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"An error occurred while downloading data for {ticker}: {e}")
        return None

# Function for feature engineering
def preprocess_and_engineer_features(df):
    df.index = pd.to_datetime(df.index)
    
    # Use only single level column names
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Fill missing values
    raw_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[raw_cols] = df[raw_cols].fillna(method='ffill').fillna(method='bfill')

    # Feature Engineering
    df['Returns'] = df['Close'].pct_change()
    for lag in [7, 9, 21, 50, 100, 200]:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    for window in [7, 9, 21, 50, 100, 200]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    return df

# Streamlit App
st.title('Stock Price Prediction and Analysis App')

st.markdown("""
**Disclaimer:**

Predicted stock prices are estimates based on historical data. Markets are volatile, and these predictions may not reflect actual future prices. Use for informational purposes only; always do your own research and manage risk.
""")

# User input for ticker
ticker_input = st.text_input("Enter the Stock Ticker (e.g., INFY, RELIANCE):", 'INFY').strip().upper()

if st.button('Analyze'):
    if ticker_input:
        ticker = ticker_input + ".NS"
        start_date = "2015-01-01"
        end_date = datetime.today().strftime('%Y-%m-%d')

        # Download data
        with st.spinner(f'Downloading data for {ticker_input}...'):
            stock_data = download_stock_data(ticker, start_date, end_date)

        if stock_data is not None:
            # --- 1. Crossover Plots ---
            st.header("Crossover Plots")
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
            stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
            stock_data['EMA_200'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
            
            # Plotting
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # SMA Crossover Plot
            st.subheader("SMA Crossover (50-day vs 200-day)")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
            ax.plot(stock_data.index, stock_data['SMA_50'], label='50-Day SMA')
            ax.plot(stock_data.index, stock_data['SMA_200'], label='200-Day SMA')
            ax.set_title(f"{ticker_input} - SMA Crossover")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # EMA Crossover Plot
            st.subheader("EMA Crossover (50-day vs 200-day)")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
            ax.plot(stock_data.index, stock_data['EMA_50'], label='50-Day EMA')
            ax.plot(stock_data.index, stock_data['EMA_200'], label='200-Day EMA')
            ax.set_title(f"{ticker_input} - EMA Crossover")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # --- 2. Price Prediction ---
            st.header("Tomorrow's Price Prediction")
            with st.spinner('Predicting tomorrow\'s price...'):
                # Preprocess data for prediction
                processed_data = preprocess_and_engineer_features(stock_data.copy())
                processed_data = processed_data.iloc[-1:].drop(columns=['Close', 'High', 'Low', 'Open', 'Volume', 'Returns'])
                
                # Check for NaN values and handle them
                if processed_data.isnull().values.any():
                    st.warning("Could not make a prediction due to insufficient historical data for feature engineering.")
                else:
                    # Scale the features
                    scaled_features = scaler.transform(processed_data)
                    
                    # Predict
                    prediction = model.predict(scaled_features)
                    predicted_price = prediction[0]

                    # Plotting prediction
                    last_close = stock_data['Close'].iloc[-1]
                    st.write(f"Last Close Price: **{last_close:.2f}**")
                    st.write(f"Predicted Tomorrow's Price: **{predicted_price:.2f}**")

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(stock_data.index[-100:], stock_data['Close'][-100:], label='Historical Prices')
                    ax.plot(stock_data.index[-1], last_close, 'ro', label=f'Last Close: {last_close:.2f}')
                    ax.plot(stock_data.index[-1] + timedelta(days=1), predicted_price, 'go', label=f'Prediction: {predicted_price:.2f}')
                    ax.set_title(f"{ticker_input} - Price Prediction")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price")
                    ax.legend()
                    st.pyplot(fig)
            
            # --- 3. Benchmark Comparison ---
            st.header("Comparison with Nifty50")
            with st.spinner('Comparing with Nifty50...'):
                nifty_data = download_stock_data('^NSEI', start_date, end_date)
                if nifty_data is not None:
                    comparison_df = pd.DataFrame()
                    comparison_df[ticker_input] = stock_data['Close']
                    comparison_df['Nifty50'] = nifty_data['Close']
                    
                    # Calculate cumulative returns
                    cumulative_returns = (comparison_df / comparison_df.iloc[0]) - 1

                    # Plotting comparison
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(cumulative_returns.index, cumulative_returns[ticker_input], label=f'{ticker_input} Cumulative Returns')
                    ax.plot(cumulative_returns.index, cumulative_returns['Nifty50'], label='Nifty50 Cumulative Returns')
                    ax.set_title(f"{ticker_input} vs. Nifty50 Cumulative Returns")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Cumulative Returns")
                    ax.legend()
                    st.pyplot(fig)

            # --- 4. Recommendation ---
            st.header("Recommendation")
            if 'predicted_price' in locals():
                if predicted_price > last_close:
                    st.success(f"**Buy** - The predicted price ({predicted_price:.2f}) is higher than the last close price ({last_close:.2f}).")
                else:
                    st.error(f"**Don't Buy** - The predicted price ({predicted_price:.2f}) is not higher than the last close price ({last_close:.2f}).")
