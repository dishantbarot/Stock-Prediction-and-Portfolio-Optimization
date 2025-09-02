
# =========================
# Load Model and Scaler
# =========================
@st.cache_resource
def load_model_and_scaler():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()


# =========================
# Data Download & Processing
# =========================
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None
    return data

def preprocess_and_engineer_features(df):
    # Forward-fill/back-fill for missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    # Returns
    df['Returns'] = df['Close'].pct_change()
    # Lag Features
    lags = [1, 5, 14, 21, 50, 100, 200]
    for lag in lags:
        df[f'lag_{lag}'] = df['Close'].shift(lag)
    # Rolling Features
    df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()
    df['rolling_mean_50'] = df['Close'].rolling(window=50).mean()
    # Date Features
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    # Drop initial rows with NaNs
    df.dropna(inplace=True)
    return df

# =========================
# Plot SMA and EMA
# =========================
def plot_smas_emas(df, ticker):
    # Calculate SMAs/EMAs
    for ma in [21, 100, 200]:
        df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()
        df[f'EMA_{ma}'] = df['Close'].ewm(span=ma, adjust=False).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=2)
    ax.plot(df.index, df['SMA_21'], label='SMA 21', color='blue')
    ax.plot(df.index, df['SMA_100'], label='SMA 100', color='green')
    ax.plot(df.index, df['SMA_200'], label='SMA 200', color='red')
    ax.set_title(f"{ticker} - Close with SMA 21/100/200")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
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
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

# =========================
# Predict Next Price
# =========================
def predict_next_price(df, model, scaler):
    features = [col for col in df.columns if col not in ['Close', 'Adj Close', 'Returns']]
    latest = df.tail(1)[features]
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    return pred

# =========================
# Benchmark Comparison
# =========================
def compare_with_nifty(stock_df, stock_ticker, start_date, end_date):
    nifty = '^NSEI'
    nifty_df = download_stock_data(nifty, start_date, end_date)
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
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.legend()
    st.pyplot(fig)

    stock_final = stock_df['Cumulative Returns'].iloc[-1]
    nifty_final = nifty_df['Cumulative Returns'].iloc[-1]

    st.markdown("### Investment Comparison")
    st.write(f"Total Cumulative Return for {stock_ticker}: **{stock_final:.2%}**")
    st.write(f"Total Cumulative Return for Nifty 50: **{nifty_final:.2%}**")
    if stock_final > nifty_final:
        st.success(f"**{stock_ticker} outperformed Nifty 50 in this period!**")
    else:
        st.info(f"**Nifty 50 outperformed {stock_ticker} in this period.**")

# =========================
# Streamlit App Layout
# =========================
st.title("Stock Prediction and Benchmark Comparison App")
st.write(
    "Enter an NSE stock ticker (e.g., RELIANCE.NS, SBIN.NS) and analyze its returns, plot SMAs and EMAs, "
    "predict the next close price using a pre-trained XGBoost model, and compare its returns with Nifty 50."
)

ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")
start_date = st.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.date_input("End Date", value=datetime.today())

if st.button("Analyze"):
    df = download_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    if df is None or df.empty:
        st.error("Could not download data for the entered ticker.")
    else:
        st.subheader("Stock Returns Plot")
        df_pre = preprocess_and_engineer_features(df.copy())
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_pre.index, df_pre['Returns'], color='teal', label='Daily Returns')
        ax.set_title(f"{ticker} - Daily Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.legend()
        st.pyplot(fig)

        st.subheader("SMAs, EMAs and Crossovers")
        plot_smas_emas(df_pre, ticker)

        st.subheader("Forecast Next Day's Closing Price")
        pred = predict_next_price(df_pre, model, scaler)
        st.metric(label="Predicted Next Close Price", value=f"{pred:.2f}")

        st.subheader("Compare with Nifty 50")
        compare_with_nifty(df_pre.copy(), ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
