###########
# IMPORTS #
###########

import sys
import os

# Configure paths and API keys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from bot2.CONFIG.config_imports import *
from bot2.CONFIG.config_vars import *
openai.api_key = OAI_KEY

# Constants for data range
start_date = "2023-01-01"
end_date = "2024-01-01"

###########
# HMM MODEL TRAINING AND PREDICTION #
###########

# Download and preprocess stock data
def download_stock_data(ticker):
    stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
    return stock_data

# Preprocess data for HMM
def preprocess_data(data):
    returns = np.log(data / data.shift(1)).dropna()  # Log returns for stationarity
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns.values.reshape(-1, 1))
    return returns, returns_scaled

# Train HMM model
def train_hmm(data_scaled, n_components=2, n_iter=100):
    model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter, random_state=42)
    model.fit(data_scaled)
    return model

# Predict hidden states using HMM
def predict_hidden_states(model, data_scaled):
    hidden_states = model.predict(data_scaled)
    return hidden_states

# Evaluate hidden states over time
def plot_hidden_states(hidden_states, returns, title="HMM Hidden States"):
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(hidden_states.max() + 1):
        idx = hidden_states == i
        ax.plot(returns.index[idx], returns[idx], 'o', label=f"State {i}")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Returns")
    ax.legend()
    plt.show()

########################
# END-TO-END EXECUTION #
########################

# Run the HMM model on a specified stock
def run_hmm_model(ticker, n_components=2):
    # Download data
    data = download_stock_data(ticker)
    
    # Preprocess data
    returns, returns_scaled = preprocess_data(data)
    
    # Train HMM model
    hmm_model = train_hmm(returns_scaled, n_components=n_components)
    
    # Predict hidden states
    hidden_states = predict_hidden_states(hmm_model, returns_scaled)
    
    # Plot hidden states
    plot_hidden_states(hidden_states, returns)

# Example usage
if __name__ == "__main__":
    run_hmm_model("AAPL", n_components=3)  # Run HMM on Apple stock data with 3 hidden states
