#########
# GOALS #
#########

#> scrape RT data, feed into stat methods

###########
# IMPORTS #
###########

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bot2.CONFIG.config_imports import *
from bot2.CONFIG.config_vars import * 

def scrape_rt_data(symbols, start_time, end_time):
    """
    Scrapes real-time data for the specified symbols within the given time frame.

    Parameters:
        symbols (list): List of stock symbols to scrape data for.
        start_time (datetime): The start time for the data request.
        end_time (datetime): The end time for the data request.

    Returns:
        pd.DataFrame: DataFrame containing the stock data for all symbols.
    """
    try:
        # Set up the request for historical stock bars
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_time,
            end=end_time
        )
        
        # Fetch stock bars from Alpaca using the already initialized data_client
        stock_bars = data_client.get_stock_bars(request_params)

        # Initialize empty DataFrame to store the data
        data_frames = []

        # Iterate through the response and build DataFrame for each symbol
        for symbol in symbols:
            # Get bars for the symbol
            bars = stock_bars[symbol]
            # Extract data from each bar
            rows = [(bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in bars]
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['symbol'] = symbol  # Add symbol column to the DataFrame
            data_frames.append(df)

        # Concatenate all the DataFrames
        result = pd.concat(data_frames, ignore_index=True)
        return result

    except Exception as e:
        print(f"Error while scraping data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error


def get_row_at_timestamp(df, timestamp):
    """
    Returns the row(s) of stock data at a specific timestamp.

    Parameters:
        df (pd.DataFrame): The DataFrame containing stock data.
        timestamp (datetime): The timestamp to filter the data on.

    Returns:
        pd.DataFrame: A DataFrame with the row(s) matching the timestamp.
    """
    # Ensure the timestamp is in pandas-compatible format
    timestamp = pd.to_datetime(timestamp)

    # Filter DataFrame to get rows matching the timestamp
    row = df[df['timestamp'] == timestamp]

    return row


def analyze_trend(df, trend_timestamps):
    """
    Analyze stock data based on specific timestamps.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing stock data.
        trend_timestamps (list): List of timestamps to analyze.
    
    Returns:
        pd.DataFrame: A DataFrame containing data for the specified timestamps.
    """
    # Initialize an empty DataFrame to store the results
    trend_data = pd.DataFrame()
    
    # Loop through each timestamp and get the data at that point
    for timestamp in trend_timestamps:
        row = get_row_at_timestamp(df, timestamp)
        trend_data = pd.concat([trend_data, row], ignore_index=True)
    
    return trend_data


def get_most_recent_stock_value_alpaca(symbol):
    """
    Fetches the most recent stock value for a given symbol from Alpaca's API.

    Parameters:
        symbol (str): Stock symbol (e.g., "AAPL").
    
    Returns:
        dict: The most recent stock data (open, high, low, close, volume).
    """
    try:
        # Request the latest 1-minute bar for the given symbol
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            limit=1  # Only get the most recent bar
        )
        
        # Fetch the most recent stock data using the already initialized data_client
        stock_bars = data_client.get_stock_bars(request_params)
        
        # Assuming stock_bars[symbol] returns a list of bars
        most_recent_bar = stock_bars[symbol][0]
        
        return {
            'timestamp': most_recent_bar.timestamp,
            'open': most_recent_bar.open,
            'high': most_recent_bar.high,
            'low': most_recent_bar.low,
            'close': most_recent_bar.close,
            'volume': most_recent_bar.volume
        }

    except Exception as e:
        print(f"Error fetching most recent stock value: {e}")
        return None