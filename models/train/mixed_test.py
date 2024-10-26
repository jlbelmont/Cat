###########
# IMPORTS #
###########

import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Constants
start_date = "2023-01-01"
end_date = "2024-01-01"

# Directory to save results (current script directory)
save_dir = os.path.dirname(os.path.abspath(__file__))

# Step 1: Download S&P 500 Data
def download_sp500_data():
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    data = {ticker: yf.download(ticker, start=start_date, end=end_date)['Close'] for ticker in tqdm(sp500_tickers, desc="Downloading S&P 500 data")}
    return pd.DataFrame(data).dropna()  # Drop rows with any NaN values

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
    data, spy_data = data.align(spy_data, join='inner')  # Align data and spy_data

    # Perform regression after aligning
    hedge_ratio = sm.OLS(data, spy_data).fit().params[0]
    spread = data - hedge_ratio * spy_data
    spread_mean = spread.mean()
    spread_std = spread.std()
    signals['Pair_Signal'] = np.where(spread < spread_mean - spread_std, 1, np.where(spread > spread_mean + spread_std, -1, 0))

    return signals

# Step 3: Optimize the Weights
def optimize_weights(signals, returns):
    # Initial weight guess
    initial_weights = np.array([0.20, 0.25, 0.15, 0.25, 0.15])

    # Define objective function for maximizing Sharpe Ratio
    def objective_function(weights):
        mixed_signal = sum(weight * signals[col] for weight, col in zip(weights, signals.columns))
        mixed_returns = mixed_signal.shift(1) * returns
        sharpe_ratio = np.mean(mixed_returns) / np.std(mixed_returns)
        return -sharpe_ratio  # Negate for maximization

    # Define bounds and constraints
    bounds = [(0, 1) for _ in initial_weights]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    # Run optimization
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    return result.x  # Optimal weights

# Step 4: Run the Model on S&P 500 Data
def run_model():
    sp500_data = download_sp500_data()
    returns = sp500_data.pct_change().dropna()

    # Store results
    performance_metrics = []
    for stock in tqdm(sp500_data.columns, desc="Running mixed model"):
        stock_data = sp500_data[stock]
        signals = calculate_signals(stock_data)
        optimal_weights = optimize_weights(signals, stock_data.pct_change().dropna())
        
        # Apply the mixed signal with optimized weights
        mixed_signal = sum(weight * signals[col] for weight, col in zip(optimal_weights, signals.columns))
        mixed_returns = mixed_signal.shift(1) * returns[stock]  # Apply mixed model returns

        # Calculate and store performance metrics
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

# Step 5: Run and Display Results
if __name__ == "__main__":
    results = run_model()
    print(results)

    # Save the results DataFrame as a CSV file in the same directory as the script
    results_csv_path = os.path.join(save_dir, "model_results.csv")
    results.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")

    # Visualize performance metrics
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Sharpe Ratio distribution
    axs[0].hist(results['Sharpe Ratio'], bins=30)
    axs[0].set_title("Sharpe Ratio Distribution")
    axs[0].set_xlabel("Sharpe Ratio")
    axs[0].set_ylabel("Frequency")

    # Cumulative Return distribution
    axs[1].hist(results['Cumulative Return'], bins=30)
    axs[1].set_title("Cumulative Return Distribution")
    axs[1].set_xlabel("Cumulative Return")
    axs[1].set_ylabel("Frequency")

    # Max Drawdown distribution
    axs[2].hist(results['Max Drawdown'], bins=30)
    axs[2].set_title("Max Drawdown Distribution")
    axs[2].set_xlabel("Max Drawdown")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()

    # Save the plot as a PNG file in the same directory as the script
    plot_path = os.path.join(save_dir, "performance_metrics.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Show plot
    plt.show()
