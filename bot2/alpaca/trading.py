###########
# IMPORTS #
###########

from alpaca_realtime import *

###########
# TRADING #
###########

def place_market_order(symbol, qty, side):
    """
    Places a market order for a given symbol with specified quantity and side.
    """
    try:
        # Create market order request
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY
        )
        
        # Place order using TradingClient
        order = trading_client.submit_order(order_data)
        print(f"Order placed: {order}")
        return order

    except Exception as e:
        print(f"Error placing order: {e}")
        return None

##########
# RUNNER #
##########

def main():
    # Define symbols, start time, and end time
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_time = datetime.now() - timedelta(minutes=15)  # Last 15 minutes
    end_time = datetime.now()

    # Scrape real-time data
    stock_data = scrape_rt_data(symbols, start_time, end_time)
    print("Scraped Real-Time Data:")
    print(stock_data)

    # Analyze trend at specific timestamps
    trend_timestamps = [start_time + timedelta(minutes=5), start_time + timedelta(minutes=10)]
    trend_analysis = analyze_trend(stock_data, trend_timestamps)
    print("\nTrend Analysis:")
    print(trend_analysis)

    # Get most recent stock value for AAPL
    most_recent_aapl = get_most_recent_stock_value_alpaca("AAPL")
    print("\nMost Recent AAPL Stock Value:")
    print(most_recent_aapl)

    # Place a market order (example)
    order_response = place_market_order("AAPL", 1, "buy")
    print("\nOrder Response:")
    print(order_response)

if __name__ == "__main__":
    main()