############
# IMPORTS  #
############

import sys
import os

# Adding the path for custom imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bot2.CONFIG.config_imports import *
from bot2.CONFIG.config_vars import *

# Define a variable to control the loop
is_running = False

# Define a function to start or stop real-time data display
def toggle_realtime_data(symbol="AAPL"):
    print("Running RT data scraping...")
    global is_running
    is_running = not is_running  # Toggle the running state
    print(f"Real-time data display {'started' if is_running else 'stopped'} for {symbol}")

    while is_running:
        try:
            # Prepare the request for the latest bar data
            now = datetime.now()
            # print(now)
            bar_request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=now - timedelta(minutes=1),  # 1 minute ago for latest bar
                end=now
            )
            bars = data_client.get_stock_bars(bar_request)
            
            # Access the latest bar data
            latest_bar = bars[symbol][0]  # Fetch the most recent data
            price = latest_bar.close
            timestamp = latest_bar.timestamp

            # Display the real-time stock data in the console
            print(f"Symbol: {symbol} | Price: ${price:.2f} | Time: {timestamp}")

            # Update frequency (e.g., every 5 seconds)
            time.sleep(5)
        except Exception as e:
            print(f"Error fetching data: {e}")
            is_running = False  # Stop on error
            break

# Example of starting the toggle function
# To toggle, call `toggle_realtime_data()` again
toggle_realtime_data()  # Starts the real-time display
# Call toggle_realtime_data() again to stop
