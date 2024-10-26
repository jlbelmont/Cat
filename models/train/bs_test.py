###########
# IMPORTS #
###########

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bot2.CONFIG.config_imports import *
from bot2.CONFIG.config_vars import * 

# Set up OpenAI API key
openai.api_key = OAI_KEY

# Constants
start_date = "2023-01-01"
end_date = "2024-01-01"
save_dir = os.path.dirname(os.path.abspath(__file__))

#########
# FUNCS #
#########

# Step 1: Download Stock Data
def download_sp500_data():
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
    data = {}
    failed_stocks = []

    for ticker in tqdm(sp500_tickers, desc="Downloading S&P 500 data"):
        attempts = 3
        for attempt in range(attempts):
            try:
                stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
                if stock_data.empty:
                    raise ValueError(f"No data for {ticker}")

                data[ticker] = stock_data
                break  # Exit retry loop on successful download

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                time.sleep(1)  # Short delay before retrying

                if attempt == attempts - 1:
                    print(f"Skipping {ticker} after {attempts} attempts.")
                    failed_stocks.append(ticker)

    if not data:
        print("Error: No data was successfully downloaded. Please check your connection or data source.")
        return pd.DataFrame()  # Return an empty DataFrame if all downloads fail

    # Log failed stocks
    if failed_stocks:
        print(f"Stocks with failed downloads: {failed_stocks}")

    return pd.DataFrame(data).dropna()  # Drop rows with any NaN values

# Hidden Markov Model Signal
def hmm_model(data):
    X = data.pct_change().dropna().values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
    model.fit(X)
    hidden_states = model.predict(X)
    return pd.Series(hidden_states, index=data.index)

# Random Forest Model for Price Prediction Signal
def random_forest_model(data):
    X = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().dropna()
    y = data['Close'].shift(-1).dropna()
    X, y = X.align(y, join='inner')
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return pd.Series(model.predict(X), index=X.index)

# Natural Language Processing Sentiment Signal
def nlp_sentiment_model(text_data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    svd = TruncatedSVD(n_components=10)
    pipeline = make_pipeline(tfidf, svd)
    sentiment_scores = pipeline.fit_transform(text_data)
    return sentiment_scores.mean(axis=1)

# ChatGPT-based Sentiment Signal
def chatgpt_sentiment(stock_name):
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"What is the market sentiment on {stock_name} stock?",
        max_tokens=50
    )
    sentiment = response['choices'][0]['text'].strip()
    return 1 if "positive" in sentiment.lower() else -1 if "negative" in sentiment.lower() else 0

# Step 2: Define Mixed Model Strategies
def calculate_signals(data, text_data, stock_name):
    signals = pd.DataFrame(index=data.index)

    sma50 = data.rolling(window=50).mean()
    sma200 = data.rolling(window=200).mean()
    signals['MA_Signal'] = np.where(sma50 > sma200, 1, -1)
    
    rolling_mean = data.rolling(window=20).mean()
    rolling_std = data.rolling(window=20).std()
    upper_band = rolling_mean + rolling_std
    lower_band = rolling_mean - rolling_std
    signals['MR_Signal'] = np.where(data < lower_band, 1, np.where(data > upper_band, -1, 0))
    
    returns = data.pct_change()
    signals['Momentum_Signal'] = np.where(returns > 0.01, 1, -1)
    
    high_20 = data.rolling(window=20).max()
    low_20 = data.rolling(window=20).min()
    signals['Breakout_Signal'] = np.where(data > high_20, 1, np.where(data < low_20, -1, 0))
    
    spy_data = yf.download("SPY", start=start_date, end=end_date)['Close']
    data, spy_data = data.align(spy_data, join='inner')
    if not data.empty and not spy_data.empty:
        hedge_ratio = sm.OLS(data, spy_data).fit().params[0]
        spread = data - hedge_ratio * spy_data
        spread_mean = spread.mean()
        spread_std = spread.std()
        signals['Pair_Signal'] = np.where(spread < spread_mean - spread_std, 1, np.where(spread > spread_mean + spread_std, -1, 0))

    # Advanced Model Signals
    signals['HMM_Signal'] = hmm_model(data)
    signals['RF_Prediction'] = random_forest_model(data)
    signals['NLP_Sentiment'] = nlp_sentiment_model(text_data)
    signals['ChatGPT_Sentiment'] = chatgpt_sentiment(stock_name)
    
    return signals

# Step 3: Optimize the Weights
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

# Step 4: Run the Model on S&P 500 Data with Progress Tracking
# Update run_model to use corrected access patterns and ffill/bfill
def run_model():
    sp500_data = download_sp500_data()
    
    if sp500_data.empty:
        print("Error: No data downloaded for the S&P 500 stocks.")
        return

    returns = sp500_data.pct_change().dropna()

    performance_metrics = []
    cumulative_returns = []

    min_data_points = 50

    for stock in tqdm(sp500_data.columns, desc="Running mixed model"):
        stock_data = sp500_data[stock]

        if stock_data.count() < min_data_points:
            print(f"Skipping {stock}: insufficient data.")
            continue

        stock_data = stock_data.ffill().bfill()

        text_data = ["The stock is performing well.", "There is concern about recent performance."]

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

    if not performance_metrics:
        print("No performance metrics were generated.")
        return

    results_df = pd.DataFrame(performance_metrics)
    print(results_df)
    results_df.to_csv(os.path.join(save_dir, "model_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    for cumulative_return in cumulative_returns:
        ax.plot(cumulative_return.index, cumulative_return.values)
    ax.set_title("Cumulative Returns of Stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    plt.tight_layout()
    plt.show()

# Step 5: Run and Display Results
if __name__ == "__main__":
    run_model()
