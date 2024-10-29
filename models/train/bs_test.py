###########
# IMPORTS #
###########

import sys
import os
import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
from hmmlearn import hmm
import yfinance as yf
import statsmodels.api as sm
import openai

# Configure paths and API keys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from bot2.CONFIG.config_imports import *
from bot2.CONFIG.config_vars import *
openai.api_key = OAI_KEY

start_date = "2023-01-01"
end_date = "2024-01-01"
save_dir = os.path.dirname(os.path.abspath(__file__))

###########
# DATA PIPELINE #
###########

# Download and clean S&P 500 data
def download_sp500_data(tickers):
    data, failed_stocks = {}, []
    for ticker in tqdm(tickers, desc="Downloading data"):
        attempts = 3
        for attempt in range(attempts):
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
                if stock_data.empty:
                    raise ValueError(f"No data for {ticker}")
                data[ticker] = stock_data
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                time.sleep(1)
                if attempt == attempts - 1:
                    failed_stocks.append(ticker)
    return pd.DataFrame(data).dropna(), failed_stocks

# Generate indicators for Mixed Model Strategies
def generate_indicators(data):
    signals = pd.DataFrame(index=data.index)
    signals['MA_Signal'] = np.where(data.rolling(window=50).mean() > data.rolling(window=200).mean(), 1, -1)
    rolling_mean = data.rolling(window=20).mean()
    rolling_std = data.rolling(window=20).std()
    upper_band, lower_band = rolling_mean + rolling_std, rolling_mean - rolling_std
    signals['MR_Signal'] = np.where(data < lower_band, 1, np.where(data > upper_band, -1, 0))
    signals['Momentum_Signal'] = np.where(data.pct_change() > 0.01, 1, -1)
    high_20, low_20 = data.rolling(window=20).max(), data.rolling(window=20).min()
    signals['Breakout_Signal'] = np.where(data > high_20, 1, np.where(data < low_20, -1, 0))
    return signals.dropna()

###########
# MODELING FUNCTIONS #
###########

# Hidden Markov Model
def hmm_model(data):
    X = data.pct_change().dropna().values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
    model.fit(X)
    return pd.Series(model.predict(X), index=data.index)

# Random Forest Model for Price Prediction
def random_forest_model(data):
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().dropna()
    y = data['Close'].shift(-1).dropna()
    X, y = X.align(y, join='inner')
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return pd.Series(model.predict(X), index=X.index)

# ChatGPT-based Sentiment
def chatgpt_sentiment(stock_name):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"What is the market sentiment on {stock_name} stock?",
        max_tokens=50
    )
    sentiment = response['choices'][0]['text'].strip()
    return 1 if "positive" in sentiment.lower() else -1 if "negative" in sentiment.lower() else 0

###########
# STRATEGY & EVALUATION #
###########

def calculate_signals(data, text_data, stock_name):
    signals = generate_indicators(data)
    signals['HMM_Signal'] = hmm_model(data)
    signals['RF_Prediction'] = random_forest_model(data)
    signals['ChatGPT_Sentiment'] = chatgpt_sentiment(stock_name)
    return signals

# Optimize model weights
def optimize_weights(signals, returns):
    initial_weights = np.array([0.20, 0.25, 0.15, 0.25, 0.15])

    def objective_function(weights):
        mixed_signal = sum(weight * signals[col] for weight, col in zip(weights, signals.columns))
        mixed_returns = mixed_signal.shift(1) * returns
        sharpe_ratio = np.mean(mixed_returns) / np.std(mixed_returns)
        return -sharpe_ratio

    bounds = [(0, 1) for _ in initial_weights]
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    result = minimize(objective_function, initial_weights, bounds=bounds, constraints=constraints)
    return result.x

# Run and evaluate the model
def run_model():
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    sp500_data, failed = download_sp500_data(sp500_tickers)

    returns = sp500_data.pct_change().dropna()
    performance_metrics, cumulative_returns = [], []

    min_data_points = 50
    for stock in tqdm(sp500_data.columns, desc="Running model"):
        stock_data = sp500_data[stock].fillna(method='ffill').fillna(method='bfill')
        text_data = ["The stock is performing well.", "Concerns about recent performance."]

        if stock_data.count() < min_data_points:
            print(f"Skipping {stock}: insufficient data.")
            continue

        try:
            signals = calculate_signals(stock_data, text_data, stock)
            optimal_weights = optimize_weights(signals, stock_data.pct_change().dropna())
            mixed_signal = sum(weight * signals[col] for weight, col in zip(optimal_weights, signals.columns))
            mixed_returns = mixed_signal.shift(1) * returns[stock]
            cumulative_return = (1 + mixed_returns).cumprod()
            cumulative_returns.append(cumulative_return)

            sharpe_ratio = np.mean(mixed_returns) / np.std(mixed_returns)
            total_return = cumulative_return.iloc[-1] - 1
            max_drawdown = np.min(np.minimum.accumulate((1 + mixed_returns).cumprod()) - (1 + mixed_returns).cumprod())

            performance_metrics.append({
                'Stock': stock,
                'Sharpe Ratio': sharpe_ratio,
                'Cumulative Return': total_return,
                'Max Drawdown': max_drawdown,
                'Optimal Weights': optimal_weights
            })

        except Exception as e:
            print(f"Error processing {stock}: {e}")
            continue

    results_df = pd.DataFrame(performance_metrics)
    results_df.to_csv(os.path.join(save_dir, "model_results.csv"), index=False)

    # Visualize cumulative returns
    fig, ax = plt.subplots(figsize=(12, 8))
    for cumulative_return in cumulative_returns:
        ax.plot(cumulative_return.index, cumulative_return.values)
    ax.set_title("Cumulative Returns of Stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()

    # Visualize Sharpe Ratio and Return Distributions
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].hist(results_df['Sharpe Ratio'], bins=30)
    axs[0].set_title("Sharpe Ratio Distribution")
    axs[0].set_xlabel("Sharpe Ratio")
    axs[0].set_ylabel("Frequency")
    axs[1].hist(results_df['Cumulative Return'], bins=30)
    axs[1].set_title("Cumulative Return Distribution")
    axs[1].set_xlabel("Cumulative Return")
    axs[1].set_ylabel("Frequency")
    axs[2].hist(results_df['Max Drawdown'], bins=30)
    axs[2].set_title("Max Drawdown Distribution")
    axs[2].set_xlabel("Max Drawdown")
    axs[2].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "performance_metrics.png"))
    plt.show()

# Execute
if __name__ == "__main__":
    run_model()
