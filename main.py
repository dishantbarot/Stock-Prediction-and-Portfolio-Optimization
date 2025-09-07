# =========================================================================
# Corrected and Unified Streamlit App Code
# =========================================================================

import subprocess
import sys
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from datetime import datetime, timedelta

# ===========================
# Disclaimer
# ===========================
st.info("""
**Disclaimer:** This tool is for educational and informational purposes only. The predictions and recommendations are based on historical data and a machine learning model. They should not be considered as financial advice. Past performance is not indicative of future results. Please consult with a qualified financial advisor before making any investment decisions.
""")


# ===========================
# Load Model and Scaler
# ===========================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = xgb.XGBRegressor()
        model.load_model('xgb_model.json')
        with open('scaler_1.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure 'xgb_model.json' and 'scaler_1.pkl' are in the same directory. {e}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading the model or scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()

# ===========================
# Functions
# ===========================
def download_stock_data(ticker, start_date, end_date):
    """Downloads stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for ticker {ticker}.")
            return None
        return data
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        return None

def preprocess_features(df):
    """Preprocesses the DataFrame to create features for the model."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df['Returns'] = df['Close'].pct_change()
    
    # Lagged features
    for lag in [7, 9, 21, 50, 100, 200]:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    
    # Rolling window features
    for window in [7, 9, 21, 50, 100, 200]:
        df[f'rolling_mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['Close'].rolling(window=window).std()
    
    # Date-related features
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    # Convert 'week' to a numeric type, as isocalendar can return a multi-index
    df['weekofyear'] = df.index.isocalendar().week.astype(int)

    # Moving averages
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df.dropna(inplace=True)
    return df

def plot_crossover(df, ticker):
    """Plots various SMA and EMA crossovers."""
    
    # SMA Crossovers
    st.subheader(f"{ticker} SMA Crossovers")
    crossover_pairs_sma = [(21, 50), (50, 200)]
    for short, long in crossover_pairs_sma:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close Price')
        ax.plot(df.index, df[f'SMA_{short}'], label=f'SMA {short}')
        ax.plot(df.index, df[f'SMA_{long}'], label=f'SMA {long}')
        ax.set_title(f"{ticker} {short} SMA and {long} SMA Crossover")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
    
    # EMA Crossovers
    st.subheader(f"{ticker} EMA Crossovers")
    crossover_pairs_ema = [(21, 50), (50, 200)]
    for short, long in crossover_pairs_ema:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label='Close Price')
        ax.plot(df.index, df[f'EMA_{short}'], label=f'EMA {short}')
        ax.plot(df.index, df[f'EMA_{long}'], label=f'EMA {long}')
        ax.set_title(f"{ticker} {short} EMA and {long} EMA Crossover")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

def predict_tomorrow(df):
    """Predicts tomorrow's closing price."""
    features = [
        'High', 'Low', 'Open', 'Volume', 'Returns',
        'lag_7', 'lag_9', 'lag_21', 'lag_50', 'lag_100', 'lag_200',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_9', 'rolling_std_9',
        'rolling_mean_21', 'rolling_std_21', 'rolling_mean_50', 'rolling_std_50',
        'rolling_mean_100', 'rolling_std_100', 'rolling_mean_200', 'rolling_std_200',
        'day', 'month', 'year', 'weekofyear',
        'SMA_7', 'SMA_14', 'SMA_21', 'SMA_50', 'SMA_100', 'SMA_200',
        'EMA_7', 'EMA_14', 'EMA_21', 'EMA_50', 'EMA_100', 'EMA_200'
    ]

    last_data = df[features].iloc[-1].values.reshape(1, -1)
    scaled_data = scaler.transform(last_data)
    prediction = model.predict(scaled_data)[0]
    return prediction

def compare_with_benchmark(stock_df, stock_ticker, benchmark_ticker, start_date, end_date):
    """
    Compares the stock's average annual returns against a benchmark index (Nifty 50)
    and plots the comparison.
    """
    st.header("Comparison with Nifty50")

    # Download benchmark data
    benchmark_df = download_stock_data(benchmark_ticker, start_date, end_date)
    if benchmark_df is None:
        st.error("Benchmark data not available.")
        return

    # Align date ranges
    start = max(stock_df.index.min(), benchmark_df.index.min())
    end = min(stock_df.index.max(), benchmark_df.index.max())
    stock_aligned = stock_df.loc[start:end].copy()
    benchmark_aligned = benchmark_df.loc[start:end].copy()

    # Compute daily returns
    stock_aligned['Returns'] = stock_aligned['Close'].pct_change()
    benchmark_aligned['Returns'] = benchmark_aligned['Close'].pct_change()

    # Add Year column
    stock_aligned['Year'] = stock_aligned.index.year
    benchmark_aligned['Year'] = benchmark_aligned.index.year

    # Compute average annual returns
    avg_stock_returns = stock_aligned.groupby('Year')['Returns'].mean()
    avg_benchmark_returns = benchmark_aligned.groupby('Year')['Returns'].mean()

    # Combine into one DataFrame and drop missing years
    returns_comparison = pd.DataFrame({
        'Stock': avg_stock_returns,
        'Benchmark': avg_benchmark_returns
    }).dropna()
    
    # Plot average annual returns
    st.subheader(f'Average Annual Returns: {stock_ticker} vs Nifty50')
    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(returns_comparison.index, returns_comparison['Stock'], marker='o', color='#1565c0', linewidth=2, label=stock_ticker)
    ax.plot(returns_comparison.index, returns_comparison['Benchmark'], marker='o', color='#ffa726', linewidth=2, label=benchmark_ticker)
    ax.set_title(f'Average Annual Returns: {stock_ticker} vs Nifty50', weight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Returns')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(returns_comparison.index)
    ax.set_xticklabels(returns_comparison.index.astype(str), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Cumulative returns for conclusion
    stock_cum_return = (1 + stock_aligned['Returns']).cumprod().iloc[-1] - 1
    benchmark_cum_return = (1 + benchmark_aligned['Returns']).cumprod().iloc[-1] - 1

    st.subheader("Investment Conclusion")
    st.write(f"Over the period from **{start.date()}** to **{end.date()}**:")
    st.write(f"Total Cumulative Return for **{stock_ticker}**: **{stock_cum_return:.2%}**")
    st.write(f"Total Cumulative Return for **Nifty50**: **{benchmark_cum_return:.2%}**")

    # Recommendation
    if stock_cum_return > benchmark_cum_return:
        st.success(f"Recommendation: **BUY**")
        st.write(f"Investing in **{stock_ticker}** would have been more beneficial than investing in **Nifty50** during this period. The analysis suggests that the stock has outperformed the benchmark.")
    else:
        st.warning(f"Recommendation: **HOLD/AVOID**")
        st.write(f"Investing in **Nifty50** would have been more beneficial than investing in **{stock_ticker}** during this period. The analysis suggests that the stock has underperformed the benchmark.")


# ===========================
# Streamlit App
# ===========================
st.title("Stock Price Forecasting and Analysis App")
st.markdown("---")

ticker = st.text_input("Enter a stock ticker (e.g., RELIANCE.NS for Reliance Industries)", "RELIANCE.NS").strip().upper()
extension = ".NS"
ticker = ticker+extension if not ticker.endswith(extension) else ticker
# Hardcoded dates
start_date = datetime(2015, 1, 1)
end_date = datetime.today()

if st.button("Analyze Stock"):
    if not ticker:
        st.warning("Please enter a stock ticker.")
    elif model is None or scaler is None:
        st.error("Model or scaler failed to load. Please check the file paths and try again.")
    else:
        st.spinner("Analyzing data...")
        stock_data = download_stock_data(ticker, start_date, end_date)
        if stock_data is not None:
            # 1. Preprocessing and Crossover Plots
            st.header("Crossover Analysis")
            preprocessed_data = preprocess_features(stock_data)
            if not preprocessed_data.empty:
                plot_crossover(preprocessed_data, ticker)
            else:
                st.warning("Not enough data to perform feature engineering and crossover analysis.")
            
            # 2. Tomorrow's Price Prediction
            st.header("Tomorrow's Price Prediction")
            if not preprocessed_data.empty:
                prediction = predict_tomorrow(preprocessed_data)
                st.write(f"Predicted Tomorrow's Close Price: **₹{prediction:.2f}**")
                
                # Plotting the prediction
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(stock_data['Close'][-100:], label='Historical Close')
                tomorrow_date = stock_data.index[-1] + timedelta(days=1)
                ax.plot(tomorrow_date, prediction, 'go', markersize=10, label='Prediction')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (₹)")
                ax.legend()
                ax.set_title(f"Last 100 Days Close Price and Tomorrow's Prediction for {ticker}")
                st.pyplot(fig)
            else:
                st.warning("Cannot predict tomorrow's price without sufficient historical data.")
            
            # 3. Benchmark Comparison and Recommendation
            compare_with_benchmark(stock_data, ticker, "^NSEI", start_date, end_date)
            
            st.success("Analysis complete!")
# ===========================