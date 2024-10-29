###########
# IMPORTS #
###########

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from bot2.CONFIG.config_imports import *
from bot2.CONFIG.config_vars import * 

# print(dir(data_client))
while True:
        try:
            # Use the initialized data_client to fetch the latest trade for the given symbol
            trade = data_client.get_stock_latest_trade("AAPL")
            price = trade.price
            timestamp = trade.timestamp

            # Display the real-time stock data in the console
            print(f"Symbol: AAPL | Price: ${price:.2f} | Time: {timestamp}")

            # Update frequency (e.g., every 5 seconds)
            time.sleep(5)
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

alp_acct = api.get_account()

test_AAPL = api.get_barset('AAPL', 'day', limit = 100)
# api.submit_order(symbol='AAPL', qty=1, side='buy', type='market', time_in_force='day')
# api.submit_order('TSLA', 1, 'sell', 'market', 'day')
# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import MarketOrderRequest
# from alpaca.trading.enums import OrderSide, TimeInForce

# trading_client = TradingClient('api-key', 'secret-key', paper=True)

# # preparing market order
# market_order_data = MarketOrderRequest(
#                     symbol="SPY",
#                     qty=0.023,
#                     side=OrderSide.BUY,
#                     time_in_force=TimeInForce.DAY
#                     )

# # Market order
# market_order = trading_client.submit_order(
#                 order_data=market_order_data
#                )

# # preparing limit order
# limit_order_data = LimitOrderRequest(
#                     symbol="BTC/USD",
#                     limit_price=17000,
#                     notional=4000,
#                     side=OrderSide.SELL,
#                     time_in_force=TimeInForce.FOK
#                    )

# # Limit order
# limit_order = trading_client.submit_order(
#                 order_data=limit_order_data
#               )

# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import MarketOrderRequest
# from alpaca.trading.enums import OrderSide, TimeInForce

# trading_client = TradingClient('api-key', 'secret-key', paper=True)

# # preparing orders
# market_order_data = MarketOrderRequest(
#                     symbol="SPY",
#                     qty=1,
#                     side=OrderSide.SELL,
#                     time_in_force=TimeInForce.GTC
#                     )

# # Market order
# market_order = trading_client.submit_order(
#                 order_data=market_order_data
#                )

# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import MarketOrderRequest
# from alpaca.trading.enums import OrderSide, TimeInForce

# trading_client = TradingClient('api-key', 'secret-key', paper=True)

# # preparing orders
# market_order_data = MarketOrderRequest(
#                     symbol="SPY",
#                     qty=0.023,
#                     side=OrderSide.BUY,
#                     time_in_force=TimeInForce.DAY,
#                     client_order_id='my_first_order',
#                     )

# # Market order
# market_order = trading_client.submit_order(
#                 order_data=market_order_data
#                )

# # Get our order using its Client Order ID.
# my_order = trading_client.get_order_by_client_id('my_first_order')
# print('Got order #{}'.format(my_order.id))

# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest
# from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# trading_client = TradingClient('api-key', 'secret-key', paper=True)

# # preparing bracket order with both stop loss and take profit
# bracket__order_data = MarketOrderRequest(
#                     symbol="SPY",
#                     qty=5,
#                     side=OrderSide.BUY,
#                     time_in_force=TimeInForce.DAY,
#                     order_class=OrderClass.BRACKET,
#                     take_profit=TakeProfitRequest(limit_price=400),
#                     stop_loss=StopLossRequest(stop_price=300)
#                     )

# bracket_order = trading_client.submit_order(
#                 order_data=bracket__order_data
#                )

# # preparing oto order with stop loss
# oto_order_data = LimitOrderRequest(
#                     symbol="SPY",
#                     qty=5,
#                     limit_price=350,
#                     side=OrderSide.BUY,
#                     time_in_force=TimeInForce.DAY,
#                     class=OrderClass.OTO,
#                     stop_loss=StopLossRequest(stop_price=300)
#                     )

# # Market order
# oto_order = trading_client.submit_order(
#                 order_data=oto_order_data
#                )

# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import TrailingStopOrderRequest
# from alpaca.trading.enums import OrderSide, TimeInForce

# trading_client = TradingClient('api-key', 'secret-key', paper=True)


# trailing_percent_data = TrailingStopOrderRequest(
#                     symbol="SPY",
#                     qty=1,
#                     side=OrderSide.SELL,
#                     time_in_force=TimeInForce.GTC,
#                     trail_percent=1.00 # hwm * 0.99
#                     )

# trailing_percent_order = trading_client.submit_order(
#                 order_data=trailing_percent_data
#                )


# trailing_price_data = TrailingStopOrderRequest(
#                     symbol="SPY",
#                     qty=1,
#                     side=OrderSide.SELL,
#                     time_in_force=TimeInForce.GTC,
#                     trail_price=1.00 # hwm - $1.00
#                     )

# trailing_price_order = trading_client.submit_order(
#                 order_data=trailing_price_data
#                )

# from alpaca.trading.client import TradingClient
# from alpaca.trading.requests import GetOrdersRequest
# from alpaca.trading.enums import QueryOrderStatus

# trading_client = TradingClient('api-key', 'secret-key', paper=True)

# # Get the last 100 closed orders
# get_orders_data = GetOrdersRequest(
#     status=QueryOrderStatus.CLOSED,
#     limit=100,
#     nested=True  # show nested multi-leg orders
# )

# trading_client.get_orders(filter=get_orders_data)

# from alpaca.trading.stream import TradingStream

# stream = TradingStream('api-key', 'secret-key', paper=True)


# @conn.on(client_order_id)
# async def on_msg(data):
#     # Print the update to the console.
#     print("Update for {}. Event: {}.".format(data.event))

# stream.subscribe_trade_updates(on_msg)
# # Start listening for updates.
# stream.run()

# Get stock position for Apple

# aapl_position = api.get_position('AAPL')

# print(aapl_position)

# # Get a list of all of our positions.
# portfolio = api.list_positions()

# # Print the quantity of shares for each position.
# for position in portfolio:
#     print("{} shares of {}".format(position.qty, position.symbol))

from alpaca.trading.stream import TradingStream

stream = TradingStream('api-key', 'secret-key', paper=True)


@conn.on(client_order_id)
async def on_msg(data):
    # Print the update to the console.
    print("Update for {}. Event: {}.".format(data.event))

stream.subscribe_trade_updates(on_msg)
# Start listening for updates.
stream.run()