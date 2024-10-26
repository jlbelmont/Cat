import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame
import schedule
import time

# Constants
API_KEY = 'YOUR_ALPACA_API_KEY'
API_SECRET = 'YOUR_ALPACA_API_SECRET'
BASE_URL = 'https://paper-api.alpaca.markets'

# Connect to Alpaca
alpaca = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Stocks and other constants
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
start_date = "2023-01-01"
end_date = "2024-01-01"

# Step 1: Download Real-Time S&P 500 Data
def download_sp500_data():
    data = {}
    for ticker in tqdm(sp500_tickers, desc="Downloading S&P 500 data"):
        try:
            bars = alpaca.get_bars(ticker, TimeFrame.Minute, limit=1000).df  # Pull latest minute data (limit as per need)
            data[ticker] = bars['close']  # Only store close prices
        except Exception as e:
            print(f"Error retrieving data for {ticker}: {e}")
    return pd.DataFrame(data).dropna()

# Step 2: Define Mixed Model Strategies
def calculate_signals(data):
    signals = pd.DataFrame(index=data.index)
    
    # Moving Average Crossover
    sma50 = data.rolling(window=50).mean()
    sma200 = data.rolling(window=200).mean()
    signals['MA_Signal'] = np.where(sma50 > sma200, 1, -1)

    # Mean Reversion
    rolling_mean = data.rolling(window=20).mean()
    rolling_std = data.rolling(window=20).std()
    upper_band = rolling_mean + rolling_std
    lower_band = rolling_mean - rolling_std
    signals['MR_Signal'] = np.where(data < lower_band, 1, np.where(data > upper_band, -1, 0))

    # Momentum
    returns = data.pct_change()
    signals['Momentum_Signal'] = np.where(returns > 0.01, 1, -1)

    # Breakout
    high_20 = data.rolling(window=20).max()
    low_20 = data.rolling(window=20).min()
    signals['Breakout_Signal'] = np.where(data > high_20, 1, np.where(data < low_20, -1, 0))

    # Pair Trading Example (Use SPY as the index for simplicity)
    spy_data = yf.download("SPY", start=start_date, end=end_date)['Close']
    
    # Align indices to ensure both series have the same dates
    data, spy_data = data.align(spy_data, join='inner')  # Keep only dates present in both
    
    # Drop NaN values after alignment
    data = data.dropna()
    spy_data = spy_data.dropna()

    # Check if we have enough data points after cleaning
    if data.empty or spy_data.empty or len(data) < 2:
        print("Insufficient data for regression after alignment and cleaning.")
        signals['Pair_Signal'] = 0  # Default to no signal if data is insufficient
        return signals

    # Reshape data to 2D arrays for OLS
    data_reshaped = data.values.reshape(-1, 1)  # Convert to 2D array
    spy_data_reshaped = spy_data.values.reshape(-1, 1)  # Convert to 2D array

    # Perform regression
    hedge_ratio = sm.OLS(data_reshaped, spy_data_reshaped).fit().params[0]
    spread = data - hedge_ratio * spy_data
    spread_mean = spread.mean()
    spread_std = spread.std()
    signals['Pair_Signal'] = np.where(spread < spread_mean - spread_std, 1, np.where(spread > spread_mean + spread_std, -1, 0))

    return signals



# Step 3: Optimize the Weights
def optimize_weights(signals, returns):
    initial_weights = np.array([0.20, 0.25, 0.15, 0.25, 0.15])

    def objective_function(weights):
        mixed_signal = sum(weight * signals[col] for weight, col in zip(weights, signals.columns))
        mixed_returns = mixed_signal.shift(1) * returns
        sharpe_ratio = np.mean(mixed_returns) / np.std(mixed_returns)
        return -sharpe_ratio  # Negate for maximization

    bounds = [(0, 1) for _ in initial_weights]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    return result.x

# Step 4: Run the Model with Real-Time Data
def run_model():
    sp500_data = download_sp500_data()
    returns = sp500_data.pct_change().dropna()

    performance_metrics = []
    for stock in tqdm(sp500_data.columns, desc="Running mixed model"):
        stock_data = sp500_data[stock]
        signals = calculate_signals(stock_data)
        optimal_weights = optimize_weights(signals, stock_data.pct_change().dropna())
        
        # Apply mixed model
        mixed_signal = sum(weight * signals[col] for weight, col in zip(optimal_weights, signals.columns))
        mixed_returns = mixed_signal.shift(1) * returns[stock]

        sharpe_ratio = np.mean(mixed_returns) / np.std(mixed_returns)
        cumulative_return = (1 + mixed_returns).prod() - 1
        max_drawdown = np.min(np.minimum.accumulate((1 + mixed_returns).cumprod()) - (1 + mixed_returns).cumprod())

        performance_metrics.append({
            'Stock': stock,
            'Sharpe Ratio': sharpe_ratio,
            'Cumulative Return': cumulative_return,
            'Max Drawdown': max_drawdown,
            'Optimal Weights': optimal_weights
        })

    return pd.DataFrame(performance_metrics)

# Scheduler to Run Model Periodically
def schedule_model():
    results = run_model()
    print(results)
    
    # Optionally: Visualize or save the results here

# Run the model every 5 minutes (or any interval as required)
schedule.every(5).minutes.do(schedule_model)

# Start the Scheduler
if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)  # Prevents CPU overload
