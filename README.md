# 📈 Stock Price Forecasting and Portfolio Optimization

A machine learning–driven application for forecasting stock prices, analyzing technical indicators, and comparing stock performance with benchmarks such as the Nifty 50.  
This project leverages **XGBoost**, **technical analysis**, and **portfolio optimization techniques** to provide data-driven insights for investors and researchers.

---

## 🚀 Features

- **Stock Data Download**  
  Fetches historical stock data directly from Yahoo Finance (`yfinance`).

- **Feature Engineering**  
  - Daily returns, lagged prices, rolling means, rolling volatility.  
  - SMA (Simple Moving Averages) and EMA (Exponential Moving Averages).  
  - Calendar-based features (day, month, year, week-of-year).

- **Machine Learning Forecasting**  
  - Trains an **XGBoost Regressor** on engineered features.  
  - Predicts next-day stock closing price.  
  - Supports GPU acceleration (`gpu_hist`).

- **Crossover Analysis**  
  - Visualizes SMA and EMA crossovers (21 vs 50, 50 vs 200).  
  - Detects short- and long-term trading signals.

- **Benchmark Comparison**  
  - Compares stock daily and cumulative returns against **Nifty 50**.  
  - Line charts for return analysis.

- **Investment Recommendation**  
  - Suggests **BUY** if the stock outperforms the benchmark.  
  - Suggests **HOLD/AVOID** if it underperforms.

- **Interactive Dashboard**  
  - Built with **Streamlit** for ease of use.  
  - User-friendly plots and insights.  

---

## 🛠️ Tech Stack

- **Programming Language:** Python 3.x  
- **Libraries:**  
  - `pandas`, `numpy`, `matplotlib`  
  - `yfinance` (stock data)  
  - `xgboost` (ML model)  
  - `scikit-learn` (scaling, preprocessing)  
  - `streamlit` (interactive dashboard)  

---

## 📂 Project Structure
├── main.py # Main Streamlit app
├── xgb_model.json # Trained XGBoost model
├── scaler_1.pkl # Saved Scaler
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/dishantbarot/Stock-Prediction-and-Portfolio-Optimization.git
   cd Stock-Prediction-and-Portfolio-Optimization

2. **Create Virtual Environment**
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. **Install Dependencies**
pip install -r requirements.txt

4. **Run the app**
streamlit run main.py


**📊 Example Outputs**

SMA & EMA Crossovers

Returns Comparison

Tomorrow’s Price Prediction
Predicted Tomorrow's Close Price: ₹ 2,456.80


**⚠️ Disclaimer**

This tool is for educational and informational purposes only.
Predictions and recommendations are based on historical data and machine learning models.
They should not be considered financial advice. Please consult a qualified financial advisor before making investment decisions.
